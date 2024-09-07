Lung CT Scan Classification with Transfer Learning and Custom Vision Transformer (ViT)
Project Overview 

This project classifies lung CT scans into four categories using a Kaggle dataset. The classes are:

1 .Adenocarcinoma (Left Lower Lobe) - T2 N0 M0 Ib

2 .Large Cell Carcinoma (Left Hilum) - T2 N2 M0 IIIa

3 .Normal - Healthy lung scans

4 .Squamous Cell Carcinoma (Left Hilum) - T1 N2 M0 IIIa


This project implements a custom-built Vision Transformer (ViT) model based on research papers and leverages transfer learning with six pre-trained models due to limited GPU power and data constraints.


Dataset
The dataset used is from Kaggle and contains labeled lung CT scans for classification.


[Dataset Link](https://www.google.com/url?q=http%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fmohamedhanyyy%2Fchest-ctscan-images)


Models Used

We use six pre-trained models for transfer learning and a custom Vision Transformer (ViT):

DenseNet121 - Pre-trained with weights1

ResNet50 - Pre-trained with weights2

Swin Transformer (Swin-T) - Pre-trained with weights3

VGG11 - Pre-trained with VGG11_Weights.IMAGENET1K_V1

Vision Transformer (ViT-B_16) - Pre-trained with weights5

ResNet18 - Pre-trained with weights6

Custom Vision Transformer (ViT)

A custom ViT model is built from scratch based on research papers, designed to handle the complexity of CT scan images for accurate classification.

Training Setup

Environment

Python Version: 3.x

PyTorch: 1.12+ or 2.x

Torchvision: 0.13+

CUDA: 11.x (if using GPU)

To ensure compatibility with updated APIs, you may use the following script for installing/updating the correct versions of PyTorch and Torchvision:

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[1]) >= 12 or int(torch.__version__.split(".")[0]) == 2, "torch version should be 1.12+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
    !pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Training Details

Batch size: 32

Optimizer: AdamW

Loss Function: CrossEntropyLoss

Learning rate: 0.0001

Scheduler: CosineAnnealingLR

Each model was fine-tuned for 10 epochs, with early stopping to prevent overfitting.

Results
![Screenshot 2024-09-07 222511](https://github.com/user-attachments/assets/5a3daff0-a2e1-4cfa-8d8c-7d8747852436)


![Screenshot 2024-09-07 222527](https://github.com/user-attachments/assets/785d2a60-c3ea-4e70-abf3-78ba9800fecb)




