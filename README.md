# Breast Ultrasound Tumor Segmentation & Classification

A **multi-task deep learning** framework that simultaneously segments breast tumors and classifies tissue type (benign / malignant / normal) from ultrasound images.

Two architectures are trained and compared on the publicly available **BUSI dataset**:

| Model | Classification Accuracy | Dice Coefficient | Mean IoU |
|-------|:-----------------------:|:----------------:|:--------:|
| Multi-Task U-Net | 65% | 0.5522 | 0.6504 |
| **BiFPN Multi-Task U-Net** | **72%** | **0.5094** | **0.6358** |

---

## Overview

Medical image analysis often requires both **localising** a lesion and **characterising** it. This project tackles both simultaneously using a single shared-encoder network. The encoder learns general visual features that benefit both tasks, while two specialised heads decode those features into:

- A **binary segmentation mask** that delineates the tumor boundary
- A **3-class softmax prediction** (benign / malignant / normal)

---

## Image Processing Techniques

| Technique | Purpose |
|-----------|---------|
| Pixel normalisation (`÷ 255`) | Scale inputs to `[0, 1]` |
| Bilinear resize to 256 × 256 | Uniform input dimensions |
| Convolutional feature extraction | Spatial feature learning |
| Batch Normalisation | Training stability |
| Dropout (0.3) | Regularisation at bottleneck |
| Skip connections (U-Net) | Preserve fine spatial detail |
| BiFPN top-down + bottom-up fusion | Richer multi-scale features |
| Dice Loss | Handles foreground/background imbalance |
| Global Average Pooling | Compact feature vector for classification |
| Threshold @ 0.5 | Convert soft mask to binary prediction |

---

## Repository Structure

```
breast-ultrasound-segmentation/
│
├── image-processing.ipynb   ← main notebook (all code)
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── images/
│   ├── input/               ← sample ultrasound images
│   └── output/              ← predicted mask overlays
│
└── results/                 ← saved metric summaries / plots
```

---

## Dataset

**Breast Ultrasound Images Dataset (BUSI)**
- 780 images across 3 classes: benign (437), malignant (210), normal (133)
- Each image paired with at least one ground-truth segmentation mask
- Source: [Kaggle — aryashah2k/breast-ultrasound-images-dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

> Download the dataset and set `DATASET_DIR` in the **Configuration** cell before running.

---

## Installation

```bash
git clone https://github.com/<your-username>/breast-ultrasound-segmentation.git
cd breast-ultrasound-segmentation
pip install -r requirements.txt
```

Python 3.10+ and a CUDA-capable GPU are recommended for training (the notebook was developed on Kaggle with dual Tesla T4).

---

## Usage

Open the notebook and run cells top-to-bottom:

```bash
jupyter notebook image-processing.ipynb
```

**Key sections:**

| Section | Description |
|---------|-------------|
| 2. Configuration | Set `DATASET_DIR`, image size, batch size, epochs |
| 3. Dataset Exploration | Mask count analysis per class |
| 4. Data Loading | Load & preprocess images and masks |
| 6. Model Architecture | U-Net and BiFPN model definitions |
| 8. Training | Train both models with early stopping |
| 9. Evaluation | Classification report + Dice / IoU metrics |
| 10. Prediction Visualization | Visual comparison of predicted vs. ground-truth masks |
| 11. Inference Demo | Run inference on a single image |

**To load a pre-trained model:**

```python
model = keras.models.load_model(
    'best_model_bifpn_multi.keras',
    custom_objects={'dice_loss': dice_loss}
)
```

---

## Example Results

Sample predictions from the BiFPN model (ground-truth | raw prediction | thresholded mask):

> *(Add screenshots from `images/output/` after running Section 10)*

---

## Technologies Used

- **TensorFlow / Keras** — model building and training
- **NumPy** — array operations
- **scikit-learn** — train/val split, classification report
- **Matplotlib** — visualisation
- **Pillow** — image I/O
- **CUDA / cuDNN** — GPU acceleration

---

## Future Improvements

- Add data augmentation (flips, brightness jitter, elastic deformation) to improve generalisation
- Replace the static 0.5 threshold with an optimal threshold tuned on the validation set
- Extend BiFPN to multiple rounds of top-down / bottom-up fusion
- Add Grad-CAM overlays to visualise which image regions drive the classification decision
- Merge secondary masks (`_mask_1`, `_mask_2`) into a union mask to capture all lesions
- Evaluate on an independent test split (currently only train/val)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The BUSI dataset is subject to its own license; please refer to the [original publication](https://doi.org/10.1016/j.dib.2019.104863) for terms of use.
