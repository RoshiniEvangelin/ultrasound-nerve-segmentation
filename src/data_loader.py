"""
data_loader.py
---------------
PyTorch Dataset and DataLoader utilities for the Automated Ultrasound Nerve Segmentation project.

Handles grayscale ultrasound image loading, preprocessing, and augmentation using Albumentations.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Dataset Class
class UltrasoundNerveDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        """
        Args:
            image_dir (str): Path to folder with ultrasound images.
            mask_dir (str): Path to folder with corresponding masks.
            transform (albumentations.Compose): Augmentations and preprocessing.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load grayscale ultrasound image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f" Could not read image file: {img_path}")

        # Normalize to [0,1]
        image = image.astype("float32") / 255.0

        # Load corresponding mask (if exists)
        mask = None
        if self.mask_dir:
            mask_name = img_name.replace(".tif", "_mask.tif")
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask.astype("float32") / 255.0
            else:
                mask = np.zeros_like(image, dtype="float32")

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure single-channel format for both image and mask
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        # Albumentations already returns torch.Tensor via ToTensorV2()
        return image, mask
# Transformations (augmentations + resize)
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.3),  # safe for float32 images
            A.Normalize(mean=(0.0,), std=(1.0,)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.0,), std=(1.0,)),
            ToTensorV2()
        ])

# Dataloader Builder
def get_loaders(train_img_dir, train_mask_dir, batch_size=8, num_workers=0, pin_memory=True):
    transform = get_transforms(train=True)
    dataset = UltrasoundNerveDataset(train_img_dir, train_mask_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # keep 0 on Windows to avoid multiprocessing issues
        pin_memory=pin_memory
    )
    return loader

# Test Run 
if __name__ == "__main__":
    train_img_dir = r"C:\Users\RE\Desktop\nerve\dataset\train\images"
    train_mask_dir = r"C:\Users\RE\Desktop\nerve\dataset\train\masks"

    loader = get_loaders(train_img_dir, train_mask_dir, batch_size=4)
    print("✅ DataLoader initialized successfully!")

    for i, (images, masks) in enumerate(loader):
        print("Image batch shape:", images.shape)
        print("Mask batch shape:", masks.shape)
        break
