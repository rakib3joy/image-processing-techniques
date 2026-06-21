# Best Model - Validation Results

**Model:** BiFPN Multi-Task U-Net
**Dataset:** Breast Ultrasound Images (BUSI) - validation split (20%)
**Checkpoint:** models/best_model_bifpn_multi.keras

---

## Classification

| Metric | Value |
|--------|:-----:|
| Accuracy | 0.7244 (72.4%) |
| Precision (macro) | 0.7398 |
| Recall (macro) | 0.6933 |
| F1-score (macro) | 0.6834 |

### Per-Class Detail

              precision    recall  f1-score   support

      benign     0.7157    0.8690    0.7849        84
   malignant     0.7895    0.3488    0.4839        43
      normal     0.7143    0.8621    0.7812        29

    accuracy                         0.7244       156
   macro avg     0.7398    0.6933    0.6834       156
weighted avg     0.7358    0.7244    0.7013       156


## Segmentation

| Metric | Value |
|--------|:-----:|
| Dice Coefficient | 0.5094 |
| Pixel Accuracy | 0.9298 (93.0%) |
| Mean IoU | 0.6358 |

---

## Comparison vs Baseline (Standard U-Net)

| Metric | U-Net | BiFPN U-Net | Change |
|--------|:-----:|:-----------:|:------:|
| Accuracy | 0.6474 | 0.7244 | +0.0769 |
| Precision (macro) | 0.5810 | 0.7398 | +0.1589 |
| Recall (macro) | 0.5594 | 0.6933 | +0.1339 |
| F1 (macro) | 0.5618 | 0.6834 | +0.1215 |
| Dice | 0.5522 | 0.5094 | -0.0428 |
| Pixel Accuracy | 0.9232 | 0.9298 | +0.0066 |
| Mean IoU | 0.6504 | 0.6358 | -0.0146 |
