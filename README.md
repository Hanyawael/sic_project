# Glaucoma Segmentation: Optic Disc & Optic Cup from Retinal Fundus Images

Deep learning project that segments the **Optic Disc (OD)** and **Optic Cup (OC)** in retinal fundus images and uses the masks to compute the **vertical Cup-to-Disc Ratio (vCDR)**, a key clinical indicator used in glaucoma screening.

> **Disclaimer:** This project is for educational and screening-support purposes only. It is not a medical device and must not replace diagnosis by a qualified ophthalmologist.

## Live Demo & Resources

| Resource | Link |
|---|---|
| Streamlit web app | https://glaucoma-segmentation.streamlit.app/ |
| Trained models (Hugging Face) | https://huggingface.co/hanyawael/glaucoma-segmentation |
| Source code (GitHub) | https://github.com/Hanyawael/sic_project |

> The `models/` folder in this repository is intentionally empty. The trained weights are hosted on Hugging Face because of their size.

## Overview

Glaucoma damages the optic nerve and can cause irreversible vision loss. An enlarged optic cup relative to the optic disc (high vCDR) is one of the main signs. This project:

1. Trains U-Net segmentation models with three different encoders and compares them.
2. Segments each fundus image into **background / optic disc / optic cup**.
3. Computes the **vCDR** from the predicted masks and flags images above a threshold of **0.5** as high risk.
4. Serves the best model through a **Streamlit** app.

## Datasets

All three datasets come from the Kaggle collection [`arnavjain1/glaucoma-datasets`](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets), using the `Images_Square` / `Masks_Square` versions:

- **G1020**
- **ORIGA**
- **REFUGE** (1,200 images)

**Mask labels:** `0` = background, `1` = optic disc (rim), `2` = optic cup.

### Data split

| Split | Images | Source |
|---|---|---|
| Train | 2,136 | G1020 + ORIGA + 1,000 REFUGE |
| Validation | 534 | 80/20 split of the 2,670 train+val images |
| Test | 200 | Held-out REFUGE images (never seen in training/validation) |

The 200 test images were drawn randomly from REFUGE with `random_state=42`.

## Preprocessing & Augmentation

- Images resized to **256 × 256** (bilinear for images, nearest-neighbour for masks so labels stay exactly {0, 1, 2}).
- Pixel values scaled to [0, 1]; masks one-hot encoded into 3 channels.
- Batch size of 16.
- Augmentation with **Albumentations**, applied to the training set only:
  - Geometric (image + mask together): horizontal/vertical flip, affine (scale, translate, rotate ±25°)
  - Pixel-level (image only): brightness/contrast, gamma, CLAHE, hue/saturation/value, Gaussian blur

## Models

U-Net built with the [`segmentation-models`](https://github.com/qubvel/segmentation_models) library (TensorFlow/Keras backend), ImageNet-pretrained encoders, softmax output over 3 classes.

| Encoder | Parameters |
|---|---|
| ResNet34 | 24.5 M |
| VGG16 | 23.8 M |
| MobileNetV2 | 8.0 M |

**Training setup**

- Loss: Dice loss
- Optimizer: Adam, learning rate `1e-4`
- Metrics: IoU, Dice coefficient
- Up to 100 epochs with callbacks:
  - `ModelCheckpoint` (best `val_iou_score`)
  - `ReduceLROnPlateau` (factor 0.5, patience 3)
  - `EarlyStopping` (patience 6, restore best weights)

## Results

### Validation set

| Model | Epochs trained | Val IoU | Val Dice |
|---|---|---|---|
| **VGG16** | 49 | **0.8457** | **0.9971** |
| MobileNetV2 | 51 | 0.8362 | 0.9969 |
| ResNet34 | 49 | 0.5518 | 0.9940 |

### Held-out test set (REFUGE, 200 images)

| Model | Test Loss | Test IoU | Test Dice |
|---|---|---|---|
| **VGG16** | 0.0026 | **0.8703** | **0.9974** |
| MobileNetV2 | 0.0028 | 0.8598 | 0.9972 |
| ResNet34 | 0.0062 | 0.5508 | 0.9938 |

> **Note:** Dice and IoU are computed over the full image including the large background class, so Dice values are very high. The disc/cup-specific results below are a more informative view of segmentation quality.

### Per-structure results (VGG16, sample images)

| Sample | Disc Dice | Cup Dice | True vCDR | Predicted vCDR | Error |
|---|---|---|---|---|---|
| T0204 | 0.9622 | 0.9175 | 0.5641 | 0.5250 | 0.0391 |
| T0267 | 0.9667 | 0.9390 | 0.6389 | 0.6053 | 0.0336 |
| T0191 | 0.9382 | 0.8714 | 0.5500 | 0.5263 | 0.0237 |

VGG16 achieved the best results on both validation and test sets, with MobileNetV2 a close second at about a third of the size.

## Clinical Analysis (vCDR)

From the predicted mask:

- **Disc** = pixels labelled 1 or 2
- **Cup** = pixels labelled 2
- **vCDR** = vertical height of cup / vertical height of disc
- **Risk flag:** vCDR > 0.5 → *High Risk*, otherwise *Normal*

The notebook also produces heatmap overlays and a full visualization (original image, predicted masks, vCDR and risk status).

## Repository Structure

```
sic_project/
├── .devcontainer/      # Dev container configuration
├── models/             # Empty: weights are hosted on Hugging Face
├── notebooks/          # Training and evaluation notebook
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Getting Started

### Run the web app locally

```bash
git clone https://github.com/Hanyawael/sic_project.git
cd sic_project
pip install -r requirements.txt
streamlit run app.py
```

Or just open the hosted version: https://glaucoma-segmentation.streamlit.app/

### Use the models

Download the trained weights from Hugging Face: https://huggingface.co/hanyawael/glaucoma-segmentation

### Reproduce the training

Open the notebook in the `notebooks/` folder on Kaggle (with the `arnavjain1/glaucoma-datasets` dataset attached and a GPU enabled) and run all cells.

## Tech Stack

Python · TensorFlow / Keras · segmentation-models · Albumentations · OpenCV · NumPy · Pandas · Matplotlib · Streamlit · Hugging Face Hub

## Acknowledgements

- Datasets: G1020, ORIGA, REFUGE (via Kaggle `arnavjain1/glaucoma-datasets`)
