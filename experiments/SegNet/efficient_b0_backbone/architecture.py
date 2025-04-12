
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import wandb
import pandas as pd 
import os
import matplotlib.pyplot as plt
import opendatasets as opd
import zipfile

torch.manual_seed(42)
np.random.seed(42)

# wandb.login(key="your_wandb_api_key_here")

EPOCHS = 25
BATCH_SIZE = 8
LR = 1e-3
NUM_CLASSES = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# wandb.init(project="segnet-efficientnet-camvid", config={
#     "epochs": EPOCHS,
#     "batch_size": BATCH_SIZE,
#     "learning_rate": LR,
#     "architecture": "SegNet-EfficientNet",
#     "dataset": "CamVid"
# })

class SegNetEfficientNet(nn.Module):
    def __init__(self, num_classes=32):
        super(SegNetEfficientNet, self).__init__()
        base_model = models.efficientnet_b0(pretrained=True)
        features = list(base_model.features.children())

        # EfficientNet-B0 backbone (output channels gradually increase to 1280)
        self.encoder = nn.Sequential(*features)  # Output: [B, 1280, H/32, W/32]

        # Decoder blocks (mirroring encoder with ConvTranspose2d)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)  # Downsampled features from EfficientNet
        x = self.decoder(x)  # Upsampled
        x = self.classifier(x)
        x = F.interpolate(x, size=(360, 480), mode='bilinear', align_corners=False)
        
        return x

class CamVidDataset(Dataset):
    """
    CamVid dataset loader with RGB mask to class index conversion.
    Expects directory structure:
        camvid/
            train/
            train_labels/
            val/
            val_labels/
            test/
            test_labels/
    """
    def __init__(self, root, split='train', transform=None, image_size=(360, 480), target_transform=None, class_dict_path='camvid/CamVid/class_dict.csv'):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = os.path.join(root, split)
        self.label_dir = os.path.join(root, f"{split}_labels")

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.label_paths = sorted(glob.glob(os.path.join(self.label_dir, '*.png')))
        self.label_resize = transforms.Resize(image_size, interpolation=Image.NEAREST)
        self.image_resize = transforms.Resize(image_size, interpolation=Image.BILINEAR)
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels."

        # Load class_dict.csv and build color-to-class mapping
        df = pd.read_csv(class_dict_path)
        self.color_to_class = {
            (row['r'], row['g'], row['b']): idx for idx, row in df.iterrows()
        }

    def __len__(self):
        return len(self.image_paths)

    def rgb_to_class(self, mask):
        """Convert an RGB mask (PIL.Image) to a 2D class index mask."""
        mask_np = np.array(mask)
        h, w, _ = mask_np.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)

        for rgb, class_idx in self.color_to_class.items():
            matches = (mask_np == rgb).all(axis=2)
            class_mask[matches] = class_idx

        return class_mask

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('RGB')

        # Resize both to 360x480
        image = self.image_resize(image)
        label = self.label_resize(label)

        if self.transform:
            image = self.transform(image)

        label = self.rgb_to_class(label)
        label = torch.from_numpy(label).long()

        return image, label

if __name__ == "__main__":  
    dataset_url = "https://www.kaggle.com/datasets/carlolepelaars/camvid"
    opd.download(dataset_url)

    # Set dataset folder (adjust path if needed)
    dataset_folder = "camvid"
    print("Dataset directory contents:")
    print(os.listdir(dataset_folder))
    input_transform = transforms.Compose([
    transforms.Resize((360, 480)),  # Or larger if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

    def label_transform(label):
        # Resize using nearest neighbor so that labels are not interpolated
        label = label.resize((480, 360), Image.NEAREST)
        label = np.array(label, dtype=np.int64)
        return torch.from_numpy(label)

    num_classes = 32
    data_root = 'camvid/CamVid/'  # make sure this matches your structure

    # Load datasets and dataloaders (assuming CamVidDataset is already defined)
    train_dataset = CamVidDataset(root=data_root, split='train',
                                transform=input_transform, target_transform=label_transform)
    val_dataset = CamVidDataset(root=data_root, split='val',
                                transform=input_transform, target_transform=label_transform)
    test_dataset = CamVidDataset(root=data_root, split='test',
                                transform=input_transform, target_transform=label_transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
