#!/usr/bin/env python3
"""finetune_chexpert.py

Usage examples:
  python finetune_chexpert.py --data-dir /path/to/CheXpert-v1.0-small --csv train.csv --valid-csv valid.csv --binary
  python finetune_chexpert.py --data-dir /mnt/data/CheXpert-v1.0-small --csv train.csv --valid-csv valid.csv --epochs 10

This script supports:
- Multi-label (default) or binary (abnormal vs normal) modes
- Mixed precision training (AMP)
- Checkpoint saving
- Simple logging to stdout

Notes:
- Make sure torchvision, torch, pandas, tqdm are installed.
- Designed for RTX 3070 Ti. Default batch_size=16 and image_size=224 are safe starting points.
"""

import argparse
import os
from pathlib import Path
import time
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda import amp
from tqdm import tqdm


class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_cols=None, uncertain_to_zero=True, binary=False):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.binary = binary

        # Filter out rows whose image files don't exist
        first_col = self.data.columns[0]
        def _exists(rel_path):
            p = os.path.join(root_dir, str(rel_path))
            return os.path.exists(p)
        before = len(self.data)
        self.data = self.data[self.data[first_col].apply(_exists)].reset_index(drop=True)
        after = len(self.data)
        if before != after:
            print(f"[Dataset] Skipped {before - after} rows with missing images. Keeping {after}/{before}.")

        # If label_cols is None, try to infer (CheXpert format: first cols are path, then metadata, then labels)
        if label_cols is None:
            # Common CheXpert small: label columns start at column index 5 (0-based -> 5th col onward)
            # But we'll attempt to detect columns named like 'No Finding' or known labels
            candidates = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                          'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                          'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
            found = [c for c in candidates if c in self.data.columns]
            if found:
                self.label_cols = found
            else:
                # fallback: assume last 14 columns are labels
                self.label_cols = list(self.data.columns[-14:])
        else:
            self.label_cols = label_cols

        # Preprocess labels
        labels = self.data[self.label_cols].copy()
        # Replace NaN with 0, replace -1 uncertain as specified
        labels = labels.fillna(0)
        if uncertain_to_zero:
            labels = labels.replace(-1, 0)
        else:
            # treat uncertain as positive
            labels = labels.replace(-1, 1)

        if binary:
            # create single 'abnormal' label = 1 if any disease present except 'No Finding'
            no_find_col = None
            for c in self.label_cols:
                if 'No Finding' in c:
                    no_find_col = c
                    break
            if no_find_col is not None:
                # abnormal if No Finding == 0 and any other label == 1
                other_cols = [c for c in self.label_cols if c != no_find_col]
                abnormal = (labels[other_cols].sum(axis=1) > 0).astype(int)
            else:
                abnormal = (labels.sum(axis=1) > 0).astype(int)
            self.labels = abnormal.values.reshape(-1, 1)
            self.label_cols = ['abnormal']
        else:
            self.labels = labels.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, str(img_rel))
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size):
    train_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_t, val_t


def build_model(num_classes, pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc='Train', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with amp.autocast():
            outputs = model(images)
            # For single-output (binary) ensure outputs shape matches
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc='Valid', leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def save_checkpoint(state, filename):
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DenseNet on CheXpert')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to CheXpert directory (containing train/ valid/ and csv files)')
    parser.add_argument('--csv', type=str, default='train.csv', help='CSV filename for training (located under data-dir)')
    parser.add_argument('--valid-csv', type=str, default='valid.csv', help='CSV filename for validation (located under data-dir)')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--binary', action='store_true', help='Train binary abnormal vs normal')
    parser.add_argument('--uncertain-to-zero', action='store_true', help='Treat uncertain (-1) as 0 (default)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use ImageNet pretrained weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_csv = os.path.join(args.data_dir, args.csv)
    valid_csv = os.path.join(args.data_dir, args.valid_csv)
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    train_t, val_t = get_transforms(args.img_size)

    # determine num_classes
    sample_df = pd.read_csv(train_csv, nrows=5)
    # Try to detect label columns like CheXpert
    common = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    label_cols = [c for c in common if c in sample_df.columns]
    if args.binary:
        num_classes = 1
    else:
        if label_cols:
            num_classes = len(label_cols)
        else:
            # fallback: assume last 14 columns
            num_classes = 14

    train_dataset = CheXpertDataset(csv_file=train_csv, root_dir=args.data_dir, transform=train_t,
                                   label_cols=label_cols if label_cols else None,
                                   uncertain_to_zero=args.uncertain_to_zero, binary=args.binary)
    valid_dataset = CheXpertDataset(csv_file=valid_csv, root_dir=args.data_dir, transform=val_t,
                                   label_cols=label_cols if label_cols else None,
                                   uncertain_to_zero=args.uncertain_to_zero, binary=args.binary)

    # If no valid images found locally, split train 80/20
    if len(valid_dataset) == 0:
        print("[Warning] No valid images found. Auto-splitting train set 80/20 for validation.")
        from torch.utils.data import random_split
        val_size = max(1, int(0.2 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

    model = build_model(num_classes=num_classes, pretrained=args.pretrained).to(device)

    # For binary use BCEWithLogitsLoss with pos_weight if class imbalance
    criterion = nn.BCEWithLogitsLoss()

    # Use AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scaler = amp.GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate_one_epoch(model, valid_loader, criterion, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - time: {elapsed:.1f}s")
        # save checkpoint
        chkpt_path = os.path.join(args.save_dir, f'chexpert_densenet_epoch{epoch+1}.pth')
        save_checkpoint({'epoch': epoch+1, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(), 'val_loss': val_loss}, chkpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, 'chexpert_densenet_best.pth')
            save_checkpoint({'epoch': epoch+1, 'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict(), 'val_loss': val_loss}, best_path)

    print('Training completed. Best val loss:', best_val_loss)


if __name__ == '__main__':
    main()
