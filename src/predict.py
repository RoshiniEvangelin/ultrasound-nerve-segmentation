import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import UNet
from src.data_loader import get_transforms

# =========================================
# Configuration
# =========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\RE\Desktop\nerve\model_best.pth"     # 🔹 best model checkpoint
TEST_DIR = r"C:\Users\RE\Desktop\nerve\dataset\test"          # 🔹 folder containing test images
OUTPUT_DIR = r"C:\Users\RE\Desktop\nerve\outputs\predictions" # 🔹 folder to save predicted masks

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Load Model
model = UNet(in_channels=1, out_channels=1).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded trained model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Train the model first.")

transform = get_transforms(train=False)
all_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
print(f"📂 Found {len(all_files)} test images.")

for idx, file in enumerate(all_files, 1):
    img_path = os.path.join(TEST_DIR, file)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"⚠️ Skipping unreadable file: {file}")
        continue

    orig_image = image.copy()
    image = image.astype("float32") / 255.0

    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(image_tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Post-processing (cleanup)
    kernel = np.ones((3, 3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    save_path = os.path.join(OUTPUT_DIR, file.replace(".tif", "_pred.png"))
    cv2.imwrite(save_path, pred_mask)

    print(f"[{idx}/{len(all_files)}] 🧠 Saved prediction → {save_path}")
# Optional: Visualize a few predictions
sample_files = all_files[:3]  # show first 3 test images
for file in sample_files:
    pred_path = os.path.join(OUTPUT_DIR, file.replace(".tif", "_pred.png"))
    img_path = os.path.join(TEST_DIR, file)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if img is not None and pred_mask is not None:
        overlay = cv2.addWeighted(img, 0.7, pred_mask, 0.3, 0)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title("Input"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(pred_mask, cmap='gray'); plt.title("Predicted Mask"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(overlay, cmap='gray'); plt.title("Overlay"); plt.axis('off')
        plt.show()

print("\n🎉 All predictions completed successfully!")
print(f" Masks saved in: {OUTPUT_DIR}")
