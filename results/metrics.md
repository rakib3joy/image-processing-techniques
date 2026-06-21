# Experiment Results

**Dataset:** Breast Ultrasound Images (BUSI)
**Split:** 80% train / 20% validation
**Image size:** 256x256
**Random seed:** 42

---

## Classification Metrics (Validation Set)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|:--------:|:-----------------:|:--------------:|:----------:|
| Multi-Task U-Net | 0.6474 | 0.5810 | 0.5594 | 0.5618 |
| **BiFPN Multi-Task U-Net** | **0.7244** | **0.7398** | **0.6933** | **0.6834** |

## Segmentation Metrics (Validation Set)

| Model | Dice Coefficient | Pixel Accuracy | Mean IoU |
|-------|:---------------:|:--------------:|:--------:|
| Multi-Task U-Net | 0.5522 | 0.9232 | 0.6504 |
| **BiFPN Multi-Task U-Net** | **0.5094** | **0.9298** | **0.6358** |

---

## Label Encoding
| Index | Class |
|:-----:|-------|
| 0 | benign |
| 1 | malignant |
| 2 | normal |

## Notes
- Segmentation threshold: 0.5
- Dice loss: 1 - (2*intersection + 1) / (sum_true + sum_pred + 1)
- Loss weights: segmentation x 1.0, classification x 0.5
