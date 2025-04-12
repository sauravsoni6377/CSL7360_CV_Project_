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
from .architecture import SegNetEfficientNet, NUM_CLASSES, DEVICE, LR, EPOCHS, train_loader, val_loader, IMAGE_SIZE
from tqdm import tqdm

model = SegNetEfficientNet(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=255)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pixel_accuracy(preds, labels):
    _, preds = torch.max(preds, 1)
    correct = (preds == labels).float()
    acc = correct.sum() / correct.numel()
    return acc

# def mean_iou(preds, labels, num_classes=NUM_CLASSES):
#     _, preds = torch.max(preds, 1)
#     ious = []
#     for cls in range(num_classes):
#         intersection = ((preds == cls) & (labels == cls)).float().sum()
#         union = ((preds == cls) | (labels == cls)).float().sum()
#         if union > 0:
#             ious.append(intersection / union)
#     return sum(ious) / len(ious) if ious else 0

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += pixel_accuracy(outputs, masks).item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Validation
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_acc += pixel_accuracy(outputs, masks).item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    # wandb.log({
    #     "epoch": epoch + 1,
    #     "train_loss": train_loss,
    #     "train_accuracy": train_acc,
    #     "val_loss": val_loss,
    #     "val_accuracy": val_acc
    # })

    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "segnet_efficientnet_camvid.pth")
# wandb.finish()

