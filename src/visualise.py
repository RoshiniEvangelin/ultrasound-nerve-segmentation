"""
predict.py
-----------
Generates segmentation predictions for ultrasound nerve images using a trained U-Net model.

Features:
- Loads best trained model from disk
- Applies identical preprocessing as during training
- Predicts segmentation mask for any single image
- Displays input, ground truth, and predicted mask
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import UNet
from albumentations.pytorch import ToTensorV2
import albumentations as A

# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"C:\Users\RE\Desktop\nerve\model_best.pth"  # Match your training script name
TEST_IMAGE_PATH = r"C:\Users\RE\Desktop\nerve\dataset\train\images\46_42.tif"
TRUE_MASK_PATH = r"C:\Users\RE\Desktop\nerve\dataset\train\masks\46_42_mask.tif"

# LOAD MODEL
model = UNet(in_channels=1, out_channels=1).to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded model from {MODEL_PATH} (using {DEVICE})")

# PREPROCESS IMAGE
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.0,), std=(1.0,)),
    ToTensorV2()
])

# Load grayscale image
image = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
true_mask = cv2.imread(TRUE_MASK_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError(f" Could not load image: {TEST_IMAGE_PATH}")
if true_mask is None:
    raise ValueError(f" Could not load mask: {TRUE_MASK_PATH}")

# Normalize and transform
augmented = transform(image=image, mask=true_mask)
image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)  # Shape: [1,1,256,256]
true_mask = augmented["mask"].cpu().numpy().squeeze()

# PREDICT MASK
with torch.no_grad():
    pred = model(image_tensor)
    pred = torch.sigmoid(pred)
    pred_mask = (pred > 0.5).float().cpu().numpy().squeeze()

# POST-PROCESS (optional cleanup)
kernel = np.ones((3, 3), np.uint8)
pred_mask = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

# VISUALIZE RESULTS
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Input Ultrasound")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(true_mask, cmap='gray')
plt.title("True Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
