import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Run the TensorFlow/Keras segmentation model on CPU so MedGemma gets the whole GPU
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import io, re, json, time, datetime, zipfile
from xml.sax.saxutils import escape

import numpy as np
import cv2
import requests
import streamlit as st
from PIL import Image, ImageOps
try:  # optional: only needed for the local MedGemma (GPU) mode
    import torch
    from transformers import pipeline
    HAS_GPU = torch.cuda.is_available()
except Exception:
    torch = pipeline = None
    HAS_GPU = False
IMG = 256
from huggingface_hub import hf_hub_download

# Segmentation weights are hosted on Hugging Face (too large for GitHub)
REPO_ID = "hanyawael/glaucoma-segmentation"
CONFIG_PATH = hf_hub_download(REPO_ID, "config/config.json")
WEIGHTS_PATH = hf_hub_download(REPO_ID, "weights/model.weights.h5")
DISC_ID, CUP_ID = 1, 2            # class IDs predicted by the model (0 = background)
PREPROCESS = "scale01"            # "scale01" = RGB / 255 ; "caffe" = VGG16 preprocess_input style
DEFAULT_THRESHOLD = 0.60
MAX_UPLOAD_MB = 15
MAX_PHOTOS = 5                    # photographs per batch
MIN_SIDE_PX = 128
FRAGMENT_LIMIT = 0.15             # max share of disc pixels allowed outside the largest connected region
MEDGEMMA_ID = "google/medgemma-1.5-4b-it"
MEDGEMMA_ATTEMPTS = 3             # attempts with validation feedback
ALLOW_FALLBACK = True             # False = only MedGemma writes the report; True = Gemini/Groq may take over if it fails
DISCLAIMER = ("Research and screening-support prototype. This is NOT a diagnostic tool and must not be used "
              "to make clinical decisions on its own. Every result must be reviewed by a qualified eye-care professional.")

# =====================================================================
# SYSTEM PROMPT
# =====================================================================
SYSTEM_PROMPT = """You are the report-writing component of a glaucoma-screening support tool used in eye clinics.

CONTEXT
A deep-learning model has automatically outlined the optic disc and the optic cup on a retinal fundus photograph (in the report images: green outline = optic disc, red outline = optic cup). Software then measured the vertical cup-to-disc ratio (vCDR = vertical height of the cup divided by vertical height of the disc), the horizontal CDR (hCDR) and the cup-to-disc area ratio. The treating doctor chose a vCDR threshold above which a case should be flagged.
You write ONE report that serves two readers at the same time:
(A) the doctor - an ophthalmologist or other clinician who wants concise, technical, accurate content;
(B) the patient - a person (or a parent) with no medical training, who wants to understand what the picture and the numbers mean without being frightened and without being falsely reassured.

INPUT (a JSON object)
- patient_age: years
- doctor_threshold_vcdr: the cut-off chosen by the doctor
- measurements: vertical_cdr, horizontal_cdr, area_cdr (cup area / disc area), disc_area_px, cup_area_px (pixel counts on a 256x256 version of the photo; they are only meaningful relative to each other, never convert them to millimetres)
- status: BELOW_THRESHOLD, NEAR_THRESHOLD (within 0.05 below the threshold), AT_OR_ABOVE_THRESHOLD, or UNRELIABLE
- quality_flags: automatic checks that failed (may be empty)
- facts: values already computed by software (comparison_sentence, difference_from_threshold, allowed_numbers, problems_in_plain_words)
The status was computed by software and is authoritative. Never contradict it and never recompute it.

NON-NEGOTIABLE RULES
1. This is a screening aid, never a diagnosis. Never write that the patient "has" or "does not have" glaucoma. Use wording such as "may", "is a screening finding", "needs confirmation by a clinician".
2. Use only the facts supplied. Never invent measurements, never mention intraocular pressure, visual fields, family history, symptoms, the other eye or any test result that was not supplied. Quote numbers exactly as given (two decimals).
3. If status is UNRELIABLE or quality_flags is not empty: state clearly that the automatic result should not be relied on, name the problem in plain words, recommend a new photograph and/or manual review by a clinician, and do not interpret the CDR numbers as meaningful. For the patient, stay calm: the problem is with the picture or the software, not necessarily with the eye.
4. Recommend only: review by a qualified eye doctor and the usual confirmatory work-up (intraocular pressure measurement, gonioscopy, central corneal thickness, visual field test, optical coherence tomography of the nerve fibre layer and optic nerve head, comparison with the other eye and with earlier photographs). Never advise treatment, medication, or stopping or starting anything.
5. Age: glaucoma becomes more common with age, particularly after 40 - mention this neutrally and only when relevant. If patient_age is under 18, say that interpretation in children differs and specialist review is preferred.
6. Tone: calm, kind, honest. BELOW_THRESHOLD is reassuring but is not an all-clear: regular eye examinations still matter. NEAR_THRESHOLD and AT_OR_ABOVE_THRESHOLD mean "worth a specialist's look", never "you have glaucoma". No alarmist words, no emojis, no markdown, no headings inside the text values.

BACKGROUND KNOWLEDGE YOU MAY USE (state it as general context, not as a finding about this patient)
- A vertical CDR of about 0.6 or more is commonly regarded as a level of suspicion; some guidelines use more than 0.7 for stronger suspicion. The upper limit of normal depends on the population, because the 97.5th percentile of vCDR in healthy people differs between populations (roughly 0.6 to 0.7).
- Cup size depends on disc size: large discs normally have large cups, and a small disc with a modest cup can still be abnormal. A CDR alone, without disc size, can mislead.
- A difference of 0.2 or more in CDR between the two eyes is a recognised warning sign, but one photograph of one eye cannot show it.
- CDR does not measure neuroretinal rim thickness, rim notching, disc haemorrhage or nerve-fibre-layer loss, and it cannot replace pressure, visual-field and OCT testing.
- Automatic segmentation can be wrong, especially for the cup, because its edge is hard to see on a flat photograph. Image quality, centring, blur, and optic-disc crescents or atrophy can all affect the outlines.

WHAT TO WRITE (JSON keys)
clinical_summary: string, 90 to 140 words, technical register for a clinician. State what was segmented, the vCDR / hCDR / area ratio, how vCDR compares with the doctor's threshold (and by how much), the status, and the main caveat.
key_findings: list of 3 to 5 short strings, each containing a number or a concrete statement.
clinician_next_steps: list of 3 to 5 short strings.
patient_explanation: string, 130 to 190 words, in plain language at the reading level of a 12-year-old: short sentences, no jargon (or explain it at once). Two or three short paragraphs separated by a blank line. Paragraph 1: what the picture shows - the optic disc is the round spot where the nerve leaves the back of the eye like a cable leaving a plug, and the cup is the small pale dip in its centre like the hollow of a bowl; the program drew a green outline around the disc and a red outline around the cup. Paragraph 2: what the cup-to-disc number means for this person and how it compares with the doctor's threshold, with no frightening language. Paragraph 3: what this does not mean (it is not a diagnosis) and what usually happens next.
glossary: list of exactly 3 objects {"term": ..., "meaning": ...} for "Optic disc", "Optic cup" and "Cup-to-disc ratio", each meaning in one plain sentence.
patient_next_steps: list of 3 or 4 short, plain, practical strings.
limitations: list of 3 or 4 short strings covering segmentation error, one-eye and disc-size limits, and that a clinician must confirm.

OUTPUT FORMAT
Return ONLY one valid JSON object with exactly these keys: clinical_summary, key_findings, clinician_next_steps, patient_explanation, glossary, patient_next_steps, limitations. No text before or after it, no code fences. Write in English."""

EXTRA_RULES = """FACTS BLOCK
The input contains "facts.comparison_sentence". Copy its direction (ABOVE/BELOW) and numbers exactly. Never work out a difference yourself. Use only numbers listed in facts.allowed_numbers.

QUALITY BAR
- clinical_summary must read like a specialist's note: name the structures, give vCDR, hCDR and area ratio, give the comparison, the status, and the single most important caveat. Count your words: 90 to 140.
- patient_explanation must be 130 to 190 words in 2 or 3 paragraphs. Never answer it in one line.
- Every list item must be a complete sentence of at least 4 words (key_findings items may be 3 words minimum, e.g. "Vertical CDR is 0.67."), never a bare label like "vCDR = 0.67".
- If status is UNRELIABLE: the clinician text says the outline failed automatic checks, names the reason, says the numbers are shown for transparency only, and recommends re-photography and manual outlining. The patient text says the computer could not read this picture well, that this is usually about the photo and not necessarily the eye, and that a new picture will probably be taken.

EXAMPLE (status UNRELIABLE, shortened) of the clinical_summary style:
"Automatic segmentation of the optic disc and cup was attempted on the supplied fundus photograph. The software returned a vertical CDR of 0.67, a horizontal CDR of 0.75 and a cup/disc area ratio of 0.47, but the disc outline was fragmented into several regions, so the result is classified UNRELIABLE and these values should not be interpreted against the threshold of 0.60. Fragmented outlines typically reflect poor image quality, off-centre framing, peripapillary atrophy or a segmentation error. Manual delineation by a clinician on the original image, or a repeat photograph, is recommended before any conclusion is drawn."
"""
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("OUTPUT FORMAT", EXTRA_RULES + "\nOUTPUT FORMAT", 1)

# =====================================================================
# VALIDATION
# =====================================================================
def check_model_files():
    errs = []
    for path, ext, label in ((CONFIG_PATH, ".json", "config"), (WEIGHTS_PATH, ".h5", "weights")):
        if not os.path.isfile(path):
            hint = " (is Google Drive mounted?)" if path.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive") else ""
            errs.append(f"Model {label} file not found: {path}{hint}")
        elif not path.lower().endswith(ext):
            errs.append(f"Model {label} file must be a '{ext}' file: {path}")
        elif os.path.getsize(path) == 0:
            errs.append(f"Model {label} file is empty: {path}")
    if not errs:
        try:
            with open(CONFIG_PATH) as f:
                json.load(f)
        except Exception as e:
            errs.append(f"Model config is not valid JSON: {e}")
    return errs


def validate_image(file):
    """Returns (RGB numpy array, list of warnings). Raises ValueError with a friendly message."""
    data = file.getvalue()
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The file is larger than {MAX_UPLOAD_MB} MB.")
    try:
        Image.open(io.BytesIO(data)).verify()
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        raise ValueError("The file could not be read as an image (it may be corrupt or not an image).")
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if min(w, h) < MIN_SIDE_PX:
        raise ValueError(f"The image is too small ({w}x{h} px). The shortest side must be at least {MIN_SIDE_PX} px.")
    arr = np.array(im)
    warns = []
    gray = arr.mean(axis=2)
    if gray.mean() < 25:
        warns.append("The image is very dark; the segmentation may be unreliable.")
    elif gray.mean() > 225:
        warns.append("The image is very bright or overexposed; the segmentation may be unreliable.")
    if gray.std() < 12:
        warns.append("The image has very low contrast; it may not be a retinal fundus photograph.")
    if max(w, h) / min(w, h) > 1.6:
        warns.append(f"Unusual aspect ratio ({w}x{h}); resizing to {IMG}x{IMG} will distort the image.")
    return arr, warns

# =====================================================================
# MODEL + IMAGE PROCESSING
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    import keras
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    if isinstance(cfg, str):                        # double-encoded JSON
        cfg = json.loads(cfg)
    cfg.pop("compile_config", None)                 # training-only part
    model = keras.models.model_from_json(json.dumps(cfg))
    model.load_weights(WEIGHTS_PATH)
    ins, outs = model.input_shape, model.output_shape
    if ins[-1] != 3:
        raise ValueError(f"Expected a 3-channel RGB input, but the model input shape is {ins}.")
    if ins[1] not in (None, IMG) or ins[2] not in (None, IMG):
        raise ValueError(f"The model expects {ins[1]}x{ins[2]} input, but this app resizes to {IMG}x{IMG}.")
    if outs[-1] != 3:
        raise ValueError(f"Expected 3 output classes (background, disc, cup), but the model output shape is {outs}.")
    return model


def preprocess(img_rgb):
    x = cv2.resize(img_rgb, (IMG, IMG)).astype("float32")
    if PREPROCESS == "caffe":
        x = x[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype="float32")
    else:
        x = x / 255.0
    return x[None]


@st.cache_data(show_spinner=False)
def run_segmentation(file_bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")
    probs = load_model().predict(preprocess(np.array(img)), verbose=0)[0]
    if not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-2):     # logits -> softmax
        e = np.exp(probs - probs.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
    return probs.astype("float32")


def largest_cc(mask):
    """Returns (largest connected component, fraction of mask pixels outside that component)."""
    m8 = mask.astype("uint8")
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return mask.astype(bool), 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    discarded = float(1.0 - areas.max() / areas.sum())
    return lab == k, discarded


def postprocess(probs):
    pred = np.argmax(probs, axis=-1)
    conf = probs.max(axis=-1)
    cup_raw = pred == CUP_ID
    disc_raw = (pred == DISC_ID) | cup_raw
    disc, frag = largest_cc(disc_raw)
    cup, _ = largest_cc(cup_raw)
    cup = cup & disc
    mean_conf = float(conf[disc].mean()) if disc.any() else 0.0
    return disc, cup, frag, mean_conf


def diameters(m):
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    v = 0 if len(rows) == 0 else rows.max() - rows.min() + 1
    h = 0 if len(cols) == 0 else cols.max() - cols.min() + 1
    return int(v), int(h)


def measure(disc, cup):
    dv, dh = diameters(disc)
    cv_, ch = diameters(cup)
    da, ca = int(disc.sum()), int(cup.sum())
    return {
        "vertical_cdr": round(cv_ / dv, 2) if dv else 0.0,
        "horizontal_cdr": round(ch / dh, 2) if dh else 0.0,
        "area_cdr": round(ca / da, 2) if da else 0.0,
        "disc_area_px": da,
        "cup_area_px": ca,
    }


def quality_flags(disc, cup, frag, conf):
    fl = []
    frac = float(disc.mean())
    if disc.sum() == 0:
        fl.append("No optic disc was detected in the photograph.")
    else:
        if frac < 0.003:
            fl.append("The detected optic disc is extremely small relative to the photograph (the image may not show the optic disc clearly).")
        if frac > 0.70:
            fl.append("The detected optic disc fills most of the photograph (unusual framing or segmentation error).")
        if frag > FRAGMENT_LIMIT:
            fl.append("The disc outline is fragmented into several separate regions.")
        if conf < 0.60:
            fl.append("The segmentation model had low confidence inside the disc region.")
    if cup.sum() == 0:
        fl.append("No optic cup was detected.")
    elif disc.sum() and cup.sum() / disc.sum() > 0.95:
        fl.append("The cup fills almost the entire disc, which is anatomically implausible.")
    return fl


def classify(vcdr, thr, flags):
    if flags:
        return "UNRELIABLE"
    if vcdr >= thr:
        return "AT_OR_ABOVE_THRESHOLD"
    if vcdr >= thr - 0.05:
        return "NEAR_THRESHOLD"
    return "BELOW_THRESHOLD"


STATUS_TEXT = {
    "BELOW_THRESHOLD": "Below the doctor's threshold",
    "NEAR_THRESHOLD": "Close to the doctor's threshold",
    "AT_OR_ABOVE_THRESHOLD": "At or above the doctor's threshold",
    "UNRELIABLE": "Unreliable result - manual review needed",
}
STATUS_COLOR = {
    "BELOW_THRESHOLD": "#1e7b34",
    "NEAR_THRESHOLD": "#b7791f",
    "AT_OR_ABOVE_THRESHOLD": "#c0392b",
    "UNRELIABLE": "#6b7280",
}


def make_overlay(img, disc, cup):
    H, W = img.shape[:2]
    d = cv2.resize(disc.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
    c = cv2.resize(cup.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    fill = np.zeros_like(img)
    fill[d > 0] = (0, 200, 0)
    fill[c > 0] = (220, 0, 0)
    sel = (d > 0) | (c > 0)
    out[sel] = (0.7 * img[sel] + 0.3 * fill[sel]).astype("uint8")
    t = max(1, int(round(max(H, W) / 400)))
    for m, col in ((d, (0, 255, 0)), (c, (255, 0, 0))):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, col, t)
    return out

# =====================================================================
# FACTS COMPUTED IN CODE (so the small model never has to do comparisons)
# =====================================================================
def build_facts(met, thr, status, flags):
    v = met["vertical_cdr"]
    diff = round(abs(v - thr), 2)
    if status == "UNRELIABLE":
        comparison = ("Do NOT compare the CDR with the threshold and do NOT interpret the numbers. "
                      "State that the outline failed automatic checks and the values are shown for transparency only.")
    elif v >= thr:
        comparison = f"The vertical CDR {v:.2f} is {diff:.2f} ABOVE (at or over) the threshold {thr:.2f}."
    else:
        comparison = f"The vertical CDR {v:.2f} is {diff:.2f} BELOW the threshold {thr:.2f}."
    plain_flags = [f.rstrip(".") for f in flags]
    return {
        "comparison_sentence": comparison,
        "difference_from_threshold": diff,
        "allowed_numbers": sorted({f"{x:.2f}" for x in
                                   (met["vertical_cdr"], met["horizontal_cdr"], met["area_cdr"], thr, diff)}),
        "problems_in_plain_words": plain_flags,
    }

# =====================================================================
# LLM CHAIN: MedGemma (primary) -> Gemini / Groq (optional backup) -> built-in template
# =====================================================================
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TRANSIENT = (429, 500, 502, 503, 504)


class LLMError(Exception):
    def __init__(self, msg, retry=False):
        super().__init__(msg)
        self.retry = retry


def call_groq(key, model, system, user):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.3,
        "max_completion_tokens": 3500,
        "response_format": {"type": "json_object"},
    }
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    try:
        r = requests.post(GROQ_URL, json=body, timeout=60,
                          headers={"Authorization": f"Bearer {key}",
                                   "Content-Type": "application/json"})
    except requests.RequestException as e:
        raise LLMError(str(e), retry=True)
    if r.status_code != 200:
        retry = r.status_code in TRANSIENT or "json_validate_failed" in r.text
        raise LLMError(f"{r.status_code}: {r.text[:200]}", retry=retry)
    return r.json()["choices"][0]["message"].get("content") or ""


def call_gemini(key, model, system, user):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 3500,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }

    def post(b):
        return requests.post(url, json=b, timeout=60,
                             headers={"x-goog-api-key": key,
                                      "Content-Type": "application/json"})
    try:
        r = post(body)
        if r.status_code == 400 and "thinking" in r.text.lower():
            body["generationConfig"].pop("thinkingConfig")
            r = post(body)
    except requests.RequestException as e:
        raise LLMError(str(e), retry=True)
    if r.status_code != 200:
        raise LLMError(f"{r.status_code}: {r.text[:200]}", retry=r.status_code in TRANSIENT)
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except Exception:
        raise LLMError("unexpected Gemini response", retry=True)


LLM_CHAIN = [
    {"label": "Google Gemini (backup)", "fn": call_gemini,
     "key_env": "GEMINI_API_KEY", "model": "gemini-flash-latest"},
    {"label": "Groq - gpt-oss-120b (optional backup)", "fn": call_groq,
     "key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-120b"},
]


def parse_report(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no JSON object in reply")
    d = json.loads(t[i:j + 1])
    for k in ("clinical_summary", "patient_explanation"):
        if not isinstance(d.get(k), str) or len(d[k].strip()) < 40:
            raise ValueError(f"missing or empty '{k}'")
    for k in ("key_findings", "clinician_next_steps", "patient_next_steps", "limitations"):
        if not isinstance(d.get(k), list) or not d[k]:
            raise ValueError(f"missing or empty '{k}'")
        d[k] = [str(x).strip() for x in d[k] if str(x).strip()]
    g = d.get("glossary")
    if not isinstance(g, list) or not g:
        raise ValueError("missing 'glossary'")
    d["glossary"] = [{"term": str(x.get("term", "")).strip(),
                      "meaning": str(x.get("meaning", "")).strip()}
                     for x in g if isinstance(x, dict)]
    return d


def wc(s):
    return len(str(s).split())


def validate_report(rep, payload):
    """Returns a list of problems found in the model's report (empty list = accepted)."""
    probs, f = [], payload["facts"]
    status = payload["status"]
    v = payload["measurements"]["vertical_cdr"]
    thr = payload["doctor_threshold_vcdr"]
    cs, pe = rep["clinical_summary"], rep["patient_explanation"]

    if not 70 <= wc(cs) <= 160:
        probs.append(f"clinical_summary has {wc(cs)} words; it must have 90 to 140.")
    if not 100 <= wc(pe) <= 210:
        probs.append(f"patient_explanation has {wc(pe)} words; it must have 130 to 190.")
    for k, lo, hi in (("key_findings", 3, 5), ("clinician_next_steps", 3, 5),
                      ("patient_next_steps", 3, 4), ("limitations", 3, 4)):
        if not lo <= len(rep[k]) <= hi:
            probs.append(f"{k} must have {lo} to {hi} items.")
        min_words = 3 if k == "key_findings" else 4
        if any(wc(x) < min_words for x in rep[k]):
            probs.append(f"Every item in {k} must be a complete sentence of at least {min_words} words "
                         f"(for example: 'The vertical CDR is 0.67.').")
    if len(rep["glossary"]) != 3:
        probs.append("glossary must have exactly 3 entries.")

    text = " ".join([cs, pe] + rep["key_findings"] + rep["clinician_next_steps"]
                    + rep["patient_next_steps"] + rep["limitations"])
    bad = set(re.findall(r"\b\d\.\d{2}\b", text)) - set(f["allowed_numbers"])
    if bad:
        probs.append(f"Numbers not in the input were used: {sorted(bad)}. Use only {f['allowed_numbers']}.")

    low = text.lower()
    if status == "UNRELIABLE":
        if not re.search(r"unreliab|not be relied|should not be relied", low):
            probs.append("State clearly that the result is unreliable and should not be relied on.")
        if re.search(r"\b(above|below|exceeds?)\b[^.]{0,40}threshold", low):
            probs.append("Do not compare the CDR with the threshold when the status is UNRELIABLE.")
    elif v >= thr and re.search(r"\bbelow\b[^.]{0,40}threshold", low):
        probs.append(f"Wrong direction: the vCDR {v:.2f} is ABOVE or at the threshold {thr:.2f}.")
    elif v < thr and re.search(r"\b(above|exceeds?)\b[^.]{0,40}threshold", low):
        probs.append(f"Wrong direction: the vCDR {v:.2f} is BELOW the threshold {thr:.2f}.")

    if re.search(r"\b(has|have|diagnosed with|does not have) glaucoma\b", low):
        probs.append("Never state that the patient has or does not have glaucoma.")
    return probs


def fallback_report(p):
    m, thr, status = p["measurements"], p["doctor_threshold_vcdr"], p["status"]
    v = m["vertical_cdr"]
    verdict = {
        "BELOW_THRESHOLD": f"The vertical CDR ({v:.2f}) is below the chosen threshold of {thr:.2f}.",
        "NEAR_THRESHOLD": f"The vertical CDR ({v:.2f}) is within 0.05 below the chosen threshold of {thr:.2f}.",
        "AT_OR_ABOVE_THRESHOLD": f"The vertical CDR ({v:.2f}) is at or above the chosen threshold of {thr:.2f}.",
        "UNRELIABLE": "Automatic quality checks failed, so the measurements should not be relied on.",
    }[status]
    pat = {
        "BELOW_THRESHOLD": "The measurement is below the level your doctor chose to watch for. This is reassuring, but it is not an all-clear, so regular eye check-ups still matter.",
        "NEAR_THRESHOLD": "The measurement is close to the level your doctor chose to watch for. This does not mean you have an eye disease; it means a specialist should take a closer look.",
        "AT_OR_ABOVE_THRESHOLD": "The measurement is at or above the level your doctor chose to watch for. This is a screening finding, not a diagnosis, and a specialist should take a closer look.",
        "UNRELIABLE": "The computer could not measure this picture reliably. This may be a problem with the photo and not with your eye. Your doctor may ask for a new photo.",
    }[status]

    if status == "UNRELIABLE":
        problems = " ".join(f.rstrip(".") + "." for f in p.get("quality_flags", []))
        summ = ("Automatic segmentation of the optic disc and optic cup was attempted on the supplied fundus photograph. "
                f"The software returned a vertical CDR of {v:.2f}, a horizontal CDR of {m['horizontal_cdr']:.2f} and a cup/disc "
                f"area ratio of {m['area_cdr']:.2f}; however, the outline failed automatic quality checks ({problems}), so the result is "
                f"classified UNRELIABLE and these values should not be interpreted against the threshold of {thr:.2f}. "
                "Poor image quality, off-centre framing, peripapillary atrophy or a segmentation error are common causes. "
                "A repeat photograph and manual delineation by a clinician are recommended before drawing any conclusion.")
    else:
        summ = (f"An automatic model segmented the optic disc and optic cup. Vertical CDR is {v:.2f}, horizontal CDR "
                f"{m['horizontal_cdr']:.2f} and cup/disc area ratio {m['area_cdr']:.2f}. {verdict} CDR depends on disc size "
                "and does not capture rim thickness, notching, disc haemorrhage or nerve fibre layer loss. Confirmation "
                "with IOP, gonioscopy, visual field testing and OCT, together with clinician review of the outlines, is required.")

    return {
        "clinical_summary": summ,
        "key_findings": [f"Vertical CDR {v:.2f} (threshold {thr:.2f}).",
                         f"Horizontal CDR {m['horizontal_cdr']:.2f}.",
                         f"Cup/disc area ratio {m['area_cdr']:.2f}."],
        "clinician_next_steps": ["Review the photograph and outlines manually.",
                                 "Consider IOP, gonioscopy, visual field and OCT as clinically indicated.",
                                 "Compare with the fellow eye and earlier photographs."],
        "patient_explanation": (
            "The picture shows the back of your eye. The round bright spot is the optic disc, where the nerve leaves the eye. "
            "The small pale dip in its middle is the cup. The program drew a green line around the disc and a red line around the cup.\n\n"
            f"{pat}\n\nThis result is not a diagnosis. Only an eye doctor can decide what it means for you."),
        "glossary": [
            {"term": "Optic disc", "meaning": "The spot where the eye's nerve leaves the back of the eye."},
            {"term": "Optic cup", "meaning": "The small pale dip in the centre of the optic disc."},
            {"term": "Cup-to-disc ratio", "meaning": "How big the cup is compared with the disc."}],
        "patient_next_steps": ["Show this report to your eye doctor.",
                               "Keep your regular eye check-ups.",
                               "Ask your doctor if any extra tests are needed."],
        "limitations": ["Automatic outlines can be wrong, especially for the cup.",
                        "One photograph of one eye cannot show differences between the two eyes.",
                        "CDR must be judged together with disc size and other tests.",
                        "A clinician must confirm every result."],
    }


@st.cache_resource(show_spinner=False)
def load_medgemma():
    if not HAS_GPU:
        raise RuntimeError("MedGemma needs a GPU. In Colab: Runtime > Change runtime type > GPU.")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. MedGemma is a gated model: accept its license on "
            f"https://huggingface.co/{MEDGEMMA_ID} and set os.environ['HF_TOKEN'] before starting Streamlit."
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = dict(task="image-text-to-text", model=MEDGEMMA_ID, device="cuda", token=token)
    try:
        return pipeline(dtype=dtype, **kwargs)           # newer transformers
    except TypeError:
        return pipeline(torch_dtype=dtype, **kwargs)     # older transformers


def call_medgemma(system, user, image_rgb, feedback=""):
    pipe = load_medgemma()

    im = Image.fromarray(image_rgb).convert("RGB")
    im.thumbnail((896, 896))                              # MedGemma's native resolution

    text = ("Write the screening-support report from the JSON below. The image is visual context only. "
            "Do not change, recalculate or invent any measurement and do not override the status.\n\n"
            "INPUT JSON:\n" + user + "\n\n")
    if feedback:
        text += ("A previous draft of this report was rejected for these reasons:\n" + feedback +
                 "\nWrite a NEW, complete report that fixes all of them.\n\n")
    text += ("Reply with ONE valid JSON object only, with exactly the keys required in the system prompt "
             "(clinical_summary, key_findings, clinician_next_steps, patient_explanation, glossary, "
             "patient_next_steps, limitations). Your reply must start with { and end with }. "
             "No explanation, no apology, no text before or after it, no code fences.")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": text},
        ]},
    ]

    out = pipe(text=messages, max_new_tokens=3000, do_sample=False, return_full_text=False)
    gen = out[0]["generated_text"]

    if isinstance(gen, str):
        return gen
    if isinstance(gen, list):                              # full conversation returned
        for m in reversed(gen):
            if m.get("role") == "assistant":
                c = m["content"]
                if isinstance(c, list):
                    return "".join(p.get("text", "") for p in c if isinstance(p, dict))
                return c
    return str(gen)


def generate_report(payload, image_rgb=None):
    user = json.dumps(payload, indent=2)
    errors = []

    # 1. MedGemma: validated, with feedback on each retry
    if not HAS_GPU:
        errors.append("MedGemma skipped: no GPU available (using cloud fallback)")
    elif image_rgb is not None:
        feedback = ""
        for attempt in range(MEDGEMMA_ATTEMPTS):
            raw = ""
            try:
                raw = call_medgemma(SYSTEM_PROMPT, user, image_rgb, feedback)
                rep = parse_report(raw)
                probs = validate_report(rep, payload)
                if not probs:
                    return rep, "Google MedGemma 1.5 4B (local, validated)", errors
                errors.append(f"MedGemma (attempt {attempt + 1}) rejected: " + " | ".join(probs))
                feedback = "- " + "\n- ".join(probs)
            except Exception as e:
                errors.append(f"MedGemma (attempt {attempt + 1}): {str(e)[:300]} | raw output starts: {raw[:200]!r}")
                # keep the previous feedback; the model just needs another try at clean JSON
    else:
        errors.append("MedGemma: no image was provided")

    # 2. Optional cloud fallback
    if ALLOW_FALLBACK:
        for step in LLM_CHAIN:
            key = os.environ.get(step["key_env"], "")
            if not key:
                errors.append(f"{step['label']}: no API key set")
                continue
            for attempt in range(2):
                try:
                    raw = step["fn"](key, step["model"], SYSTEM_PROMPT, user)
                    return parse_report(raw), step["label"], errors
                except LLMError as e:
                    errors.append(f"{step['label']}: {e}")
                    if not e.retry:
                        break
                    time.sleep(2 * (attempt + 1))
                except Exception as e:
                    errors.append(f"{step['label']}: invalid answer ({str(e)[:200]})")
                    time.sleep(1)

    # 3. Built-in template
    return fallback_report(payload), "Built-in template (AI explanation unavailable)", errors

# =====================================================================
# PDF REPORT
# =====================================================================
def build_pdf(info, orig, overlay, met, status, flags, rep, llm_used):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image as RLImage, Table, TableStyle, PageBreak, KeepTogether)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    FONT, FONT_B, unicode_ok = "Helvetica", "Helvetica-Bold", False
    try:
        import matplotlib
        d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DV", os.path.join(d, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DV-B", os.path.join(d, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")
        FONT, FONT_B, unicode_ok = "DV", "DV-B", True
    except Exception:
        pass

    def clean(t):
        t = str(t)
        if not unicode_ok:
            for a, b in (("\u2265", ">="), ("\u2264", "<="), ("\u2013", "-"), ("\u2014", "-"),
                         ("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
                t = t.replace(a, b)
            t = t.encode("latin-1", "replace").decode("latin-1")
        return escape(t).replace("\n", "<br/>")

    navy = colors.HexColor("#12355b")
    H1 = ParagraphStyle("H1", fontName=FONT_B, fontSize=19, leading=23, textColor=navy, spaceAfter=2)
    SUB = ParagraphStyle("SUB", fontName=FONT, fontSize=9, textColor=colors.HexColor("#555555"), spaceAfter=8)
    H2 = ParagraphStyle("H2", fontName=FONT_B, fontSize=12.5, leading=16, textColor=navy, spaceBefore=10, spaceAfter=4)
    P = ParagraphStyle("P", fontName=FONT, fontSize=10, leading=14.5, spaceAfter=5)
    SM = ParagraphStyle("SM", fontName=FONT, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
    BUL = ParagraphStyle("BUL", parent=P, leftIndent=14, bulletIndent=3, spaceAfter=3)
    CAP = ParagraphStyle("CAP", fontName=FONT, fontSize=8.5, alignment=1, textColor=colors.HexColor("#444444"))
    CELL = ParagraphStyle("CELL", fontName=FONT, fontSize=9.5, leading=12)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName=FONT_B)
    WARN = ParagraphStyle("WARN", parent=P, fontSize=9, leading=12, backColor=colors.HexColor("#FFF4CE"),
                          borderPadding=5, textColor=colors.HexColor("#5A4500"), spaceAfter=8)

    def bullets(items):
        return [Paragraph(clean(x), BUL, bulletText="\u2022" if unicode_ok else "-") for x in items]

    def jpg(arr, maxpx=900):
        im = Image.fromarray(arr)
        im.thumbnail((maxpx, maxpx))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=85)
        b.seek(0)
        return b, im.size

    def rl_img(arr, width):
        b, (w, h) = jpg(arr)
        return RLImage(b, width=width, height=width * h / w)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1 * cm, "AI-assisted screening support - not a medical diagnosis. Must be reviewed by a qualified clinician.")
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Glaucoma Screening Support Report")
    S = []
    S.append(Paragraph("Glaucoma Screening Support Report", H1))
    S.append(Paragraph("Optic disc and cup segmentation with explanation for clinician and patient", SUB))
    S.append(Paragraph(clean(DISCLAIMER), WARN))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    t = Table([[Paragraph("Patient", CELLB), Paragraph(clean(info["name"]), CELL),
                Paragraph("Age", CELLB), Paragraph(f"{info['age']} years", CELL)],
               [Paragraph("Report date", CELLB), Paragraph(now, CELL),
                Paragraph("Doctor's vCDR threshold", CELLB), Paragraph(f"{info['threshold']:.2f}", CELL)]],
              colWidths=[2.6 * cm, 5.4 * cm, 4.2 * cm, 4.8 * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#b8c4d4")),
                           ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dce6")),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                           ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef2f7")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    S += [t, Spacer(1, 8)]

    w = 8.2 * cm
    it = Table([[rl_img(orig, w), rl_img(overlay, w)],
                [Paragraph("Original fundus photograph", CAP),
                 Paragraph("Segmentation: green = optic disc, red = optic cup", CAP)]],
               colWidths=[8.5 * cm, 8.5 * cm])
    it.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, 0), "MIDDLE")]))
    S += [it, Spacer(1, 8)]

    col = colors.HexColor(STATUS_COLOR[status])
    rows = [[Paragraph("Measurement", CELLB), Paragraph("Value", CELLB)],
            [Paragraph("Vertical cup-to-disc ratio (vCDR)", CELL), Paragraph(f"{met['vertical_cdr']:.2f}", CELLB)],
            [Paragraph("Horizontal cup-to-disc ratio (hCDR)", CELL), Paragraph(f"{met['horizontal_cdr']:.2f}", CELL)],
            [Paragraph("Cup / disc area ratio", CELL), Paragraph(f"{met['area_cdr']:.2f}", CELL)],
            [Paragraph("Disc area / cup area (pixels, 256x256)", CELL),
             Paragraph(f"{met['disc_area_px']} / {met['cup_area_px']}", CELL)],
            [Paragraph("Result against doctor's threshold", CELL),
             Paragraph(f"<font color='{STATUS_COLOR[status]}'><b>{clean(STATUS_TEXT[status])}</b></font>", CELL)]]
    mt = Table(rows, colWidths=[10 * cm, 7 * cm])
    mt.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#b8c4d4")),
                            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dce6")),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                            ("LINEBEFORE", (0, 5), (0, 5), 3, col),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    S.append(mt)
    S.append(Spacer(1, 3))
    S.append(Paragraph("The vCDR threshold is set by the clinician; it is configurable and is not a universal clinical cutoff.", SM))
    if flags:
        S.append(Spacer(1, 6))
        S.append(Paragraph("<b>Automatic quality warnings</b>", P))
        S += bullets(flags)
    else:
        S.append(Spacer(1, 4))
        S.append(Paragraph("Automatic segmentation quality checks: no problems detected.", SM))

    S.append(Paragraph("1. Summary for the clinician", H2))
    S.append(Paragraph(clean(rep["clinical_summary"]), P))
    S.append(Paragraph("<b>Key findings</b>", P))
    S += bullets(rep["key_findings"])
    S.append(Paragraph("<b>Suggested clinical next steps</b>", P))
    S += bullets(rep["clinician_next_steps"])

    S.append(PageBreak())
    S.append(Paragraph("2. Your results explained in plain language", H2))
    S.append(Paragraph(clean(rep["patient_explanation"]), P))
    S.append(Paragraph("<b>Words you may hear</b>", P))
    for g in rep["glossary"]:
        S.append(Paragraph(f"<b>{clean(g['term'])}:</b> {clean(g['meaning'])}", BUL, bulletText="\u2022" if unicode_ok else "-"))
    S.append(Paragraph("<b>What you can do next</b>", P))
    S += bullets(rep["patient_next_steps"])

    S.append(Paragraph("3. Limitations", H2))
    S += bullets(rep["limitations"])

    S.append(Spacer(1, 10))
    note = ("Method: U-Net with VGG16 encoder segmenting optic disc and optic cup on a 256x256 version of the photograph; "
            "the largest connected region of each structure is kept; CDR = vertical (or horizontal) extent of the cup divided by that of the disc "
            "(disc region includes the cup). The threshold was entered by the clinician and is not a universal cutoff. "
            f"Explanation text produced by: {llm_used}. "
            "This report supports, and does not replace, examination by a qualified eye care professional.")
    S.append(KeepTogether([Paragraph(clean(note), SM)]))

    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()

# =====================================================================
# MULTI-PHOTO TRIAGE (ranking)
# =====================================================================
import pandas as pd

# Lower tier number = reviewed earlier. Edit this dict to change the ordering policy.
TIER = {"AT_OR_ABOVE_THRESHOLD": 0, "NEAR_THRESHOLD": 1, "UNRELIABLE": 2, "BELOW_THRESHOLD": 3}
PRIORITY_TEXT = {
    "AT_OR_ABOVE_THRESHOLD": "High priority",
    "NEAR_THRESHOLD": "Medium priority",
    "UNRELIABLE": "Retake photo / manual review",
    "BELOW_THRESHOLD": "Routine follow-up",
}
PRIORITY_ICON = {
    "AT_OR_ABOVE_THRESHOLD": "\U0001F534",
    "NEAR_THRESHOLD": "\U0001F7E0",
    "UNRELIABLE": "\u26AA",
    "BELOW_THRESHOLD": "\U0001F7E2",
}
RANK_EXPLANATION = ("Order: cases at or above the doctor's threshold first (highest vertical CDR first), then cases near the "
                    "threshold, then unreliable results that need a new photo or manual review, then cases below the threshold. "
                    "Ties are broken by area ratio, then horizontal CDR. This is a scheduling aid for clinician review, "
                    "not a diagnosis and not a measure of how severe anyone's condition is.")


def priority_key(c):
    m = c["met"]
    return (TIER[c["status"]], -m["vertical_cdr"], -m["area_cdr"], -m["horizontal_cdr"])


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "patient"


def build_summary_pdf(cases, thr):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image as RLImage, Table, TableStyle)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    FONT, FONT_B, unicode_ok = "Helvetica", "Helvetica-Bold", False
    try:
        import matplotlib
        d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DV", os.path.join(d, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DV-B", os.path.join(d, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")
        FONT, FONT_B, unicode_ok = "DV", "DV-B", True
    except Exception:
        pass

    def clean(t):
        t = str(t)
        if not unicode_ok:
            t = t.encode("latin-1", "replace").decode("latin-1")
        return escape(t).replace("\n", "<br/>")

    navy = colors.HexColor("#12355b")
    H1 = ParagraphStyle("H1", fontName=FONT_B, fontSize=19, leading=23, textColor=navy, spaceAfter=2)
    SUB = ParagraphStyle("SUB", fontName=FONT, fontSize=9, textColor=colors.HexColor("#555555"), spaceAfter=8)
    P = ParagraphStyle("P", fontName=FONT, fontSize=9, leading=12.5, spaceAfter=5)
    SM = ParagraphStyle("SM", fontName=FONT, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
    CELL = ParagraphStyle("CELL", fontName=FONT, fontSize=9, leading=11.5)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName=FONT_B)
    WARN = ParagraphStyle("WARN", parent=P, fontSize=9, leading=12, backColor=colors.HexColor("#FFF4CE"),
                          borderPadding=5, textColor=colors.HexColor("#5A4500"), spaceAfter=8)

    def thumb(arr, width):
        im = Image.fromarray(arr)
        im.thumbnail((500, 500))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=80)
        b.seek(0)
        w, h = im.size
        return RLImage(b, width=width, height=width * h / w)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1 * cm, "AI-assisted screening support - not a medical diagnosis. Must be reviewed by a qualified clinician.")
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    S = [Paragraph("Glaucoma Screening - Triage Summary", H1),
         Paragraph(f"{len(cases)} photograph(s) ranked for clinician review - doctor's vCDR threshold {thr:.2f} - "
                   f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", SUB),
         Paragraph(clean(DISCLAIMER), WARN)]

    rows = [[Paragraph(h, CELLB) for h in ("#", "Overlay", "Patient", "vCDR", "hCDR", "Area", "Priority")]]
    for c in cases:
        m, info, s = c["met"], c["info"], c["status"]
        pr = f"<font color='{STATUS_COLOR[s]}'><b>{clean(PRIORITY_TEXT[s])}</b></font><br/>{clean(STATUS_TEXT[s])}"
        rows.append([Paragraph(str(c["rank"]), CELLB),
                     thumb(c["overlay"], 3.0 * cm),
                     Paragraph(f"<b>{clean(info['name'])}</b><br/>{info['age']} years<br/>{clean(c['file'])}", CELL),
                     Paragraph(f"{m['vertical_cdr']:.2f}", CELLB),
                     Paragraph(f"{m['horizontal_cdr']:.2f}", CELL),
                     Paragraph(f"{m['area_cdr']:.2f}", CELL),
                     Paragraph(pr, CELL)])
    t = Table(rows, colWidths=[1.0 * cm, 3.4 * cm, 3.8 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 4.0 * cm], repeatRows=1)
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#b8c4d4")),
                           ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dce6")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    S += [t, Spacer(1, 8), Paragraph(clean(RANK_EXPLANATION), SM)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Glaucoma Screening Triage Summary")
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def build_zip(cases, summary_pdf):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if summary_pdf:
            z.writestr("00_triage_summary.pdf", summary_pdf)
        for c in cases:
            if c.get("pdf"):
                z.writestr(f"{c['rank']:02d}_{safe_name(c['info']['name'])}.pdf", c["pdf"])
    return buf.getvalue()

# =====================================================================
# USER INTERFACE
# =====================================================================
def show_case(c):
    status, met, info, rep = c["status"], c["met"], c["info"], c["rep"]
    for w in c["warns"]:
        st.warning(w)
    msg = f"**{STATUS_TEXT[status]}** - vertical CDR {met['vertical_cdr']:.2f} vs threshold {info['threshold']:.2f}"
    {"BELOW_THRESHOLD": st.success, "NEAR_THRESHOLD": st.warning,
     "AT_OR_ABOVE_THRESHOLD": st.error, "UNRELIABLE": st.info}[status](msg)
    if status == "BELOW_THRESHOLD":
        st.caption("A result below the threshold does not rule out glaucoma.")
    if c["flags"]:
        for f in c["flags"]:
            st.warning(f)
    else:
        st.caption("\u2705 Automatic segmentation quality checks passed.")

    a, b = st.columns(2)
    a.image(c["img"], caption="Original", use_container_width=True)
    b.image(c["overlay"], caption="Green = optic disc, red = optic cup", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vertical CDR", f"{met['vertical_cdr']:.2f}")
    m2.metric("Horizontal CDR", f"{met['horizontal_cdr']:.2f}")
    m3.metric("Area ratio", f"{met['area_cdr']:.2f}")
    m4.metric("Threshold", f"{info['threshold']:.2f}")

    tab_d, tab_p = st.tabs(["For the doctor", "For the patient"])
    with tab_d:
        st.write(rep["clinical_summary"])
        st.markdown("**Key findings**")
        for x in rep["key_findings"]:
            st.markdown(f"- {x}")
        st.markdown("**Suggested next steps**")
        for x in rep["clinician_next_steps"]:
            st.markdown(f"- {x}")
        st.markdown("**Limitations**")
        for x in rep["limitations"]:
            st.markdown(f"- {x}")
    with tab_p:
        st.write(rep["patient_explanation"])
        st.markdown("**Words you may hear**")
        for g in rep["glossary"]:
            st.markdown(f"- **{g['term']}**: {g['meaning']}")
        st.markdown("**What you can do next**")
        for x in rep["patient_next_steps"]:
            st.markdown(f"- {x}")

    if c.get("pdf"):
        st.download_button("\u2B07 Download this report (PDF)", data=c["pdf"],
                           file_name=f"glaucoma_report_{c['rank']:02d}_{safe_name(info['name'])}.pdf",
                           mime="application/pdf", key=f"dl_{c['rank']}")
    st.caption(f"Explanation written by: {c['used']}")
    if c["errs"]:
        with st.expander("Why MedGemma / the AI explanation was not used"):
            for e in c["errs"]:
                st.write(e)


def analyze_batch(details, thr):
    """details = list of (uploaded_file, name, age). Returns dict with ranked cases and failures."""
    cases, failed = [], []
    n = len(details)
    prog = st.progress(0.0, text="Segmenting photographs...")
    for i, (f, nm, ag) in enumerate(details):
        label = f"{nm.strip()} ({f.name})"
        prog.progress(i / (2 * n), text=f"Segmenting photo {i + 1} of {n}: {f.name}")
        try:
            img, warns = validate_image(f)
            probs = run_segmentation(f.getvalue())
            disc, cup, frag, conf = postprocess(probs)
            met = measure(disc, cup)
            flags = quality_flags(disc, cup, frag, conf)
            status = classify(met["vertical_cdr"], float(thr), flags)
            overlay = make_overlay(img, disc, cup)
            cases.append(dict(file=f.name, img=img, overlay=overlay, met=met, flags=flags,
                              status=status, warns=warns,
                              info={"name": nm.strip(), "age": int(ag), "threshold": float(thr)}))
        except ValueError as e:
            failed.append(f"{label}: {e}")
        except Exception as e:
            failed.append(f"{label}: could not be analysed ({e})")

    cases.sort(key=priority_key)
    for k, c in enumerate(cases, 1):
        c["rank"] = k

    for k, c in enumerate(cases):
        prog.progress(0.5 + 0.5 * k / max(len(cases), 1),
                      text=f"Writing report {k + 1} of {len(cases)}: {c['info']['name']}")
        payload = {"patient_age": c["info"]["age"], "doctor_threshold_vcdr": round(float(thr), 2),
                   "measurements": c["met"], "status": c["status"], "quality_flags": c["flags"]}
        payload["facts"] = build_facts(c["met"], float(thr), c["status"], c["flags"])
        rep, used, errs = generate_report(payload, c["img"])
        c.update(rep=rep, used=used, errs=list(errs))
        try:
            c["pdf"] = build_pdf(c["info"], c["img"], c["overlay"], c["met"], c["status"],
                                 c["flags"], rep, used)
        except Exception as e:
            c["pdf"] = None
            c["errs"].append(f"PDF could not be built: {e}")
    prog.empty()

    summary_pdf, zip_bytes = None, None
    if cases:
        try:
            summary_pdf = build_summary_pdf(cases, float(thr))
        except Exception as e:
            failed.append(f"Triage summary PDF could not be built: {e}")
        zip_bytes = build_zip(cases, summary_pdf)
    return dict(cases=cases, failed=failed, summary_pdf=summary_pdf, zip=zip_bytes, thr=float(thr))


def main():
    st.set_page_config(page_title="GlaucoGuide AI", page_icon="\U0001F441",
                       layout="wide", initial_sidebar_state="collapsed")
    st.markdown("<style>#MainMenu{visibility:hidden;} footer{visibility:hidden;} "
                "[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)
    st.title("\U0001F441 GlaucoGuide AI")
    st.caption(f"Upload up to {MAX_PHOTOS} retinal photographs. Each one is outlined and measured, the cases are ranked "
               "so the ones that most need a specialist's look come first, and a report is written for the doctor and the patient.")
    st.warning(DISCLAIMER)

    files = st.file_uploader(f"Fundus photographs (up to {MAX_PHOTOS})", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    thr = st.number_input(
        "Doctor's vCDR threshold", min_value=0.30, max_value=0.95,
        value=DEFAULT_THRESHOLD, step=0.05, format="%.2f",
        help="Configurable screening-flag level, NOT a universal clinical cutoff. Cases with a vertical "
             "cup-to-disc ratio at or above this value are flagged for further review. 0.60 is a commonly "
             "used level of suspicion; some guidelines use 0.65 or 0.70. The right value depends on the "
             "population and the doctor's judgement.")

    details = []
    too_many = bool(files) and len(files) > MAX_PHOTOS
    if too_many:
        st.error(f"Please upload at most {MAX_PHOTOS} photographs (you uploaded {len(files)}).")
    elif files:
        st.markdown("**Patient details for each photograph**")
        for i, f in enumerate(files):
            kid = f"{i}_{f.name}_{f.size}"
            c1, c2 = st.columns([3, 1])
            nm = c1.text_input(f"Photo {i + 1}: patient name ({f.name})",
                               value=os.path.splitext(f.name)[0], key=f"nm_{kid}")
            ag = c2.number_input("Age (years)", min_value=0, max_value=120, value=None, step=1,
                                 placeholder="e.g. 58", key=f"ag_{kid}")
            details.append((f, nm, ag))

    go = st.button("Analyze and rank", type="primary")

    if go:
        errors = []
        if not details:
            errors.append("Please upload at least one fundus photograph.")
        for i, (f, nm, ag) in enumerate(details, 1):
            if not nm.strip():
                errors.append(f"Photo {i}: please enter the patient's name.")
            if ag is None:
                errors.append(f"Photo {i} ({f.name}): please enter the patient's age.")
        if too_many:
            errors.append(f"Too many photographs (maximum {MAX_PHOTOS}).")
        errors += check_model_files()
        if errors:
            st.session_state.pop("batch", None)
            for e in errors:
                st.error(e)
        else:
            try:
                res = analyze_batch(details, float(thr))
            except Exception as e:
                st.session_state.pop("batch", None)
                st.error(f"Could not complete the analysis: {e}")
            else:
                if res["cases"]:
                    st.session_state["batch"] = res
                else:
                    st.session_state.pop("batch", None)
                    for msg in res["failed"]:
                        st.error(msg)

    r = st.session_state.get("batch")
    if not r:
        return
    cases = r["cases"]

    for msg in r["failed"]:
        st.error(msg)

    st.subheader("Review order")
    st.caption(RANK_EXPLANATION)

    counts = {s: sum(1 for c in cases if c["status"] == s) for s in TIER}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{PRIORITY_ICON['AT_OR_ABOVE_THRESHOLD']} High priority", counts["AT_OR_ABOVE_THRESHOLD"])
    k2.metric(f"{PRIORITY_ICON['NEAR_THRESHOLD']} Medium priority", counts["NEAR_THRESHOLD"])
    k3.metric(f"{PRIORITY_ICON['UNRELIABLE']} Retake / manual review", counts["UNRELIABLE"])
    k4.metric(f"{PRIORITY_ICON['BELOW_THRESHOLD']} Routine", counts["BELOW_THRESHOLD"])

    df = pd.DataFrame([{
        "Rank": c["rank"],
        "Patient": c["info"]["name"],
        "Age": c["info"]["age"],
        "Photo": c["file"],
        "vCDR": c["met"]["vertical_cdr"],
        "hCDR": c["met"]["horizontal_cdr"],
        "Area ratio": c["met"]["area_cdr"],
        "Priority": f"{PRIORITY_ICON[c['status']]} {PRIORITY_TEXT[c['status']]}",
        "Status": STATUS_TEXT[c["status"]],
    } for c in cases])
    st.dataframe(df, hide_index=True, use_container_width=True)

    d1, d2 = st.columns(2)
    if r.get("summary_pdf"):
        d1.download_button("\u2B07 Triage summary (PDF)", data=r["summary_pdf"],
                           file_name="glaucoma_triage_summary.pdf", mime="application/pdf", type="primary")
    if r.get("zip"):
        d2.download_button("\u2B07 All reports (ZIP)", data=r["zip"],
                           file_name="glaucoma_reports.zip", mime="application/zip")

    st.subheader("Individual cases")
    for c in cases:
        title = (f"{PRIORITY_ICON[c['status']]} #{c['rank']} - {c['info']['name']} - "
                 f"vCDR {c['met']['vertical_cdr']:.2f} - {PRIORITY_TEXT[c['status']]}")
        with st.expander(title, expanded=(c["rank"] == 1)):
            show_case(c)


if __name__ == "__main__":
    main()
