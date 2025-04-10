import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# wandb.login(key="your_wandb_api_key_here")

EPOCHS = 25
BATCH_SIZE = 8
LR = 1e-3
NUM_CLASSES = 21  # Pascal VOC has 21 classes including background
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# wandb.init(project="segnet-efficientnet-voc", config={
#     "epochs": EPOCHS,
#     "batch_size": BATCH_SIZE,
#     "learning_rate": LR,
#     "architecture": "SegNet-EfficientNet",
#     "dataset": "PascalVOC2012"
# })

class SegNetEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNetEfficientNet, self).__init__()
        base_model = models.efficientnet_b0(pretrained=True)
        features = list(base_model.features.children())

        # Encoder: Use EfficientNet blocks
        self.encoder = nn.Sequential(*features)

        # Decoder: Up-convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
        return x

class VOCSegmentationDataset(VOCSegmentation):
    def __init__(self, root, image_set='train', transform=None, target_transform=None):
        super().__init__(root=root, year='2012', image_set=image_set, download=True)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        target = torch.as_tensor(np.array(target), dtype=torch.long)
        return img, target
if __name__ == "__main__":  
    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST)

    train_dataset = VOCSegmentationDataset("voc_data", 'train', image_transform, mask_transform)
    val_dataset = VOCSegmentationDataset("voc_data", 'val', image_transform, mask_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
