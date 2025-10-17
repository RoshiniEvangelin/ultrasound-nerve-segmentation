"""
train.py
---------
Training script for Automated Ultrasound Nerve Segmentation using U-Net.

Features:
- BCE + Dice combined loss for stable training
- Automatic checkpoint saving of best model
- Learning rate scheduler for smooth convergence
- Final Dice & loss curve visualization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.model import UNet
import importlib

# Import DataLoader dynamically
data_loader = importlib.import_module("src.data_loader")
get_loaders = data_loader.get_loaders
# Loss Functions
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class BCEDiceLoss(nn.Module):
    """Combines BCEWithLogitsLoss with Dice loss for balanced segmentation."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * dice_loss(pred, target)
# Dice Metric
def dice_coefficient(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)


# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
LEARNING_RATE = 1e-5
BATCH_SIZE = 8

# Paths
TRAIN_IMG_DIR = r"C:\Users\RE\Desktop\nerve\dataset\train\images"
TRAIN_MASK_DIR = r"C:\Users\RE\Desktop\nerve\dataset\train\masks"
MODEL_SAVE_PATH = r"C:\Users\RE\Desktop\nerve\model_best.pth"


# Main Training Function
def main():
    print(f"🚀 Training on device: {DEVICE}")

    # Load Data
    train_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, batch_size=BATCH_SIZE, num_workers=0)

    # Model, Loss, Optimizer, Scheduler
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # Resume if checkpoint exists
    best_dice = 0.0
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"🔁 Loaded existing model from {MODEL_SAVE_PATH}")
    else:
        print("🆕 Starting fresh training session!")

    # Track loss & dice
    losses, dices = [], []
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, leave=True)
        epoch_loss, epoch_dice = 0.0, 0.0

        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # Ensure masks have a channel dimension
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                dice = dice_coefficient(preds, masks)

            epoch_loss += loss.item()
            epoch_dice += dice.item()

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item(), dice=dice.item())

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        scheduler.step(avg_loss)

        losses.append(avg_loss)
        dices.append(avg_dice)

        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] | Avg Loss: {avg_loss:.4f} | Avg Dice: {avg_dice:.4f}")

        # Save Best Model
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Saved new best model (Dice: {best_dice:.4f})")
    # Training Complete
    print(f"\n🎉 Training complete! Best Dice Score: {best_dice:.4f}")
    print(f"📁 Model saved at: {MODEL_SAVE_PATH}")

    # Plot Training Progress
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Training Loss')
    plt.plot(dices, label='Dice Score')
    plt.legend()
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

# Entry Point
if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
