"""
2_train_fasterrcnn.py
Trains Faster R-CNN (MobileNetV3 backbone) on:
  - Penn-Fudan Pedestrian Dataset
  - Oxford-IIIT Pet Subset (5 breeds)
Saves model weights and training logs to results/.
"""

import os
import json
import time
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────
# Custom Dataset
# ─────────────────────────────────────────────────────
class DetectionDataset(Dataset):
    """Generic detection dataset that reads images + JSON label files."""

    def __init__(self, img_dir, label_dir, transforms=None, num_classes=2, label_offset=1):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.label_offset = label_offset  # offset labels (Faster R-CNN uses 0 = background)

        self.imgs = sorted([
            f for f in self.img_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        label_path = self.label_dir / f"{img_path.stem}.json"
        with open(label_path) as f:
            ann = json.load(f)

        boxes = torch.as_tensor(ann["boxes"], dtype=torch.float32)
        # Shift labels so 0 = background (add offset)
        labels = torch.as_tensor(
            [l + self.label_offset for l in ann["labels"]], dtype=torch.int64
        )
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(len(boxes), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


def get_transform():
    return T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])


def get_transform_augmented():
    return T.Compose([
        T.Resize((512, 512)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
    ])


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────
def build_fasterrcnn(num_classes):
    """Faster R-CNN with MobileNetV3 backbone, fine-tuned head."""
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.train()  # keep train mode to get losses
    total_loss = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss / len(data_loader)


def train_model(model, train_loader, val_loader, num_epochs, lr, save_name, device):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val_loss = float("inf")
    patience, patience_counter = 5, 0

    total_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)
        lr_scheduler.step()

        epoch_time = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(epoch_time)

        print(f"  Epoch {epoch+1}/{num_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"time={epoch_time:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), RESULTS_DIR / f"{save_name}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    total_time = time.time() - total_start
    history["total_time"] = total_time
    with open(RESULTS_DIR / f"{save_name}_history.json", "w") as f:
        json.dump(history, f)
    print(f"  Total training time: {total_time:.1f}s")
    return model, history


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # --- Penn-Fudan ---
    print("=" * 60)
    print("Training Faster R-CNN on Penn-Fudan")
    print("=" * 60)

    pf_train = DetectionDataset(
        "data/pennfudan_split/train/images",
        "data/pennfudan_split/train/labels",
        transforms=get_transform_augmented(),
        label_offset=0,  # labels are already 1 (person)
    )
    pf_val = DetectionDataset(
        "data/pennfudan_split/val/images",
        "data/pennfudan_split/val/labels",
        transforms=get_transform(),
        label_offset=0,
    )

    pf_train_loader = DataLoader(pf_train, batch_size=2, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2)
    pf_val_loader = DataLoader(pf_val, batch_size=2, shuffle=False,
                                collate_fn=collate_fn, num_workers=2)

    pf_model = build_fasterrcnn(num_classes=2)  # background + person
    pf_model, pf_hist = train_model(
        pf_model, pf_train_loader, pf_val_loader,
        num_epochs=15, lr=0.005,
        save_name="frcnn_pennfudan", device=DEVICE
    )

    # --- Oxford Pets Subset ---
    print("\n" + "=" * 60)
    print("Training Faster R-CNN on Oxford Pets (5 breeds)")
    print("=" * 60)

    pet_train = DetectionDataset(
        "data/pets_split/train/images",
        "data/pets_split/train/labels",
        transforms=get_transform_augmented(),
        label_offset=1,  # labels 0-4 → 1-5 (0 = background)
    )
    pet_val = DetectionDataset(
        "data/pets_split/val/images",
        "data/pets_split/val/labels",
        transforms=get_transform(),
        label_offset=1,
    )

    pet_train_loader = DataLoader(pet_train, batch_size=2, shuffle=True,
                                   collate_fn=collate_fn, num_workers=2)
    pet_val_loader = DataLoader(pet_val, batch_size=2, shuffle=False,
                                 collate_fn=collate_fn, num_workers=2)

    pet_model = build_fasterrcnn(num_classes=6)  # background + 5 breeds
    pet_model, pet_hist = train_model(
        pet_model, pet_train_loader, pet_val_loader,
        num_epochs=20, lr=0.005,
        save_name="frcnn_pets", device=DEVICE
    )

    print("\nFaster R-CNN training complete!")
