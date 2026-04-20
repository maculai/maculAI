"""
macul.ai — Vision Model Training
Trains EfficientNet on ophthalmic imaging data for pathology detection.

Datasets:
- RFMiD (Retinal Fundus Multi-disease Image Dataset) — Kaggle
- IDRiD (Indian Diabetic Retinopathy Image Dataset)
- OCT-C8 (8-class OCT classification)
- REFUGE (Glaucoma detection)

Usage:
  python vision_model.py --data ./data/fundus/ --task fundus
  python vision_model.py --data ./data/oct/ --task oct
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import argparse


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class OphthalmicDataset(Dataset):
    """
    Dataset loader for ophthalmic images.
    Expected folder structure:
    data/fundus/
      normal/
      diabetic_retinopathy/
      wet_amd/
      retinal_tear/
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                for img_file in os.listdir(cls_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((
                            os.path.join(cls_path, img_file),
                            self.class_to_idx[cls]
                        ))

        print(f"[Dataset] {len(self.samples)} images, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[macul.ai Vision] Training on {device}")

    train_dataset = OphthalmicDataset(os.path.join(args.data, "train"), TRAIN_TRANSFORM)
    val_dataset = OphthalmicDataset(os.path.join(args.data, "val"), VAL_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = build_model(num_classes=len(train_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("./weights", exist_ok=True)
            torch.save(model.state_dict(), f"./weights/{args.task}_best.pt")
            print(f"[macul.ai] Saved best model — {val_acc:.2f}%")

    print(f"[macul.ai] Done. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/fundus/")
    parser.add_argument("--task", default="fundus", choices=["fundus", "oct"])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args)
