# Training Configuration

## Dataset
- **Name:** Breast Ultrasound Images (BUSI)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Classes:** benign, malignant, normal
- **Split:** 80% train / 20% validation
- **Random seed:** 42

## Preprocessing
- **Image size:** 256 x 256
- **Normalisation:** pixel values / 255 -> [0, 1]
- **Mask channels:** 1 (grayscale binary)

## Training
- **Batch size:** 16
- **Max epochs:** 50
- **Optimizer:** Adam (default lr = 1e-3)
- **Early stopping:** patience = 10 epochs, monitors val_loss
- **Checkpoint:** saves best model only, monitors val_loss

## Loss Functions and Weights
| Head | Loss | Weight |
|------|------|:------:|
| Segmentation | Dice Loss | 1.0 |
| Classification | Categorical Cross-Entropy | 0.5 |

## Post-processing
- **Segmentation threshold:** 0.5

## Model Checkpoints
| Model | Checkpoint |
|-------|-----------|
| Standard Multi-Task U-Net | models/best_model_unet_multi.keras |
| BiFPN Multi-Task U-Net (primary) | models/best_model_bifpn_multi.keras |
