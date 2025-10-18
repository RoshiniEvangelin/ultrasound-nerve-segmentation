<font size="2">
# **Ultrasound-Nerve-Segmentation**
This project implements a U-Net-based deep learning model to automatically segment nerves from ultrasound images.
It was trained on the Ultrasound Nerve Segmentation dataset from Kaggle, using PyTorch.

PROJECT OVERVIEW
This project implements a U-Net-based deep learning model to automatically segment nerves from ultrasound images.
It was trained on the Ultrasound Nerve Segmentation dataset from Kaggle, using PyTorch. 


**FOLDER STRUCTURE**
nerve/
├── src/
│   ├── model.py           # U-Net model architecture
│   ├── data_loader.py     # Dataset and DataLoader
│   ├── train.py           # Training loop
│   ├── predict.py         # Run model on new images
├── examples/              # Example inputs, masks, predictions
├── README.md              # Project description
├── requirements.txt       # Dependencies
└── .gitignore

**DATASET**

This project uses the Ultrasound Nerve Segmentation dataset from Kaggle.
You need to download it manually or via the Kaggle API.

**🔗 DATASET LINK:**
Kaggle: https:https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/data


**HOW TO DOWNLOAD**
Option 1 — Manually
Go to the link above and click Download All.
Extract the zip file.
Organize your data like this:
dataset/
└── train/
    ├── images/
    │   ├── 1_1.tif
    │   ├── 1_2.tif
    │   └── ...
    └── masks/
        ├── 1_1_mask.tif
        ├── 1_2_mask.tif
        └── ...


⚙️**INSTALLATION AND SETUP**
1.Clone this repository:git clone 
https://github.com/ROSHINI0211/ultrasound-nerve-segmentation.git
cd ultrasound-nerve-segmentation
2.Create a conda environment:
conda create -n nerve_seg python=3.10
conda activate nerve_seg
3.Training the Model
python -m src.train
4.Making Predictions
python -m src.predict

**MODEL DETAILS**
Architecture: U-Net
Loss: 50% Binary Cross-Entropy + 50% Dice Loss
Optimizer: Adam (lr=1e-5)
Scheduler: ReduceLROnPlateau
Metric: Dice Coefficient
Epochs: 40


**TECHNOLOGIES USED**
Python 3.10
PyTorch
OpenCV
Albumentations
Matplotlib
tqdm
</font>
