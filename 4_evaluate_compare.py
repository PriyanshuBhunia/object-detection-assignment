"""
4_evaluate_compare.py
Evaluates both models on both datasets:
  - mAP@0.5, Precision, Recall
  - Training time
  - Inference speed (images/sec)
Generates comparison tables and example prediction images.
"""

import os
import json
import time
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]


# ─────────────────────────────────────────────────────
# Dataset (same as training)
# ─────────────────────────────────────────────────────
class DetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None, label_offset=1):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.label_offset = label_offset
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
        labels = torch.as_tensor(
            [l + self.label_offset for l in ann["labels"]], dtype=torch.int64
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target, str(img_path)

    @staticmethod
    def collate_fn(batch):
        imgs, targets, paths = zip(*batch)
        return list(imgs), list(targets), list(paths)


def get_transform():
    return T.Compose([T.Resize((512, 512)), T.ToTensor()])


# ─────────────────────────────────────────────────────
# mAP Computation (simplified, IoU threshold 0.5)
# ─────────────────────────────────────────────────────
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_ap(precisions, recalls):
    """11-point interpolated AP."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precs_above = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(precs_above) if precs_above else 0
    return ap / 11


def evaluate_predictions(all_preds, all_gts, iou_threshold=0.5, num_classes=2):
    """
    all_preds: list of dicts with boxes, labels, scores
    all_gts: list of dicts with boxes, labels
    Returns: mAP, per-class precision, recall
    """
    per_class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "scores": [], "matches": []})

    for preds, gts in zip(all_preds, all_gts):
        gt_boxes = gts["boxes"]
        gt_labels = gts["labels"]
        pred_boxes = preds["boxes"]
        pred_labels = preds["labels"]
        pred_scores = preds["scores"]

        # Sort predictions by score descending
        if len(pred_scores) > 0:
            order = np.argsort(-np.array(pred_scores))
            pred_boxes = [pred_boxes[i] for i in order]
            pred_labels = [pred_labels[i] for i in order]
            pred_scores = [pred_scores[i] for i in order]

        matched_gt = set()

        for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
            best_iou = 0
            best_gt_idx = -1
            for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if gl != pl:
                    continue
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                per_class_stats[pl]["tp"] += 1
                per_class_stats[pl]["matches"].append((ps, True))
                matched_gt.add(best_gt_idx)
            else:
                per_class_stats[pl]["fp"] += 1
                per_class_stats[pl]["matches"].append((ps, False))

        for gi, gl in enumerate(gt_labels):
            if gi not in matched_gt:
                per_class_stats[gl]["fn"] += 1

    # Compute per-class AP, precision, recall
    aps = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for cls in sorted(per_class_stats.keys()):
        stats = per_class_stats[cls]
        total_tp += stats["tp"]
        total_fp += stats["fp"]
        total_fn += stats["fn"]

        # Sort matches by score
        matches = sorted(stats["matches"], key=lambda x: -x[0])
        precisions_list = []
        recalls_list = []
        tp_cum, fp_cum = 0, 0
        total_gt = stats["tp"] + stats["fn"]

        for score, is_tp in matches:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            p = tp_cum / (tp_cum + fp_cum)
            r = tp_cum / total_gt if total_gt > 0 else 0
            precisions_list.append(p)
            recalls_list.append(r)

        ap = compute_ap(precisions_list, recalls_list) if precisions_list else 0
        aps.append(ap)

    mAP = np.mean(aps) if aps else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    return mAP, precision, recall


# ─────────────────────────────────────────────────────
# Faster R-CNN Evaluation
# ─────────────────────────────────────────────────────
def evaluate_fasterrcnn(weights_path, test_loader, num_classes, score_thresh=0.5):
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_gts = []
    total_imgs = 0
    total_time = 0

    with torch.no_grad():
        for images, targets, paths in test_loader:
            images_gpu = [img.to(DEVICE) for img in images]

            start = time.time()
            outputs = model(images_gpu)
            total_time += time.time() - start
            total_imgs += len(images)

            for out, tgt in zip(outputs, targets):
                mask = out["scores"] >= score_thresh
                pred = {
                    "boxes": out["boxes"][mask].cpu().numpy().tolist(),
                    "labels": out["labels"][mask].cpu().numpy().tolist(),
                    "scores": out["scores"][mask].cpu().numpy().tolist(),
                }
                gt = {
                    "boxes": tgt["boxes"].numpy().tolist(),
                    "labels": tgt["labels"].numpy().tolist(),
                }
                all_preds.append(pred)
                all_gts.append(gt)

    fps = total_imgs / total_time if total_time > 0 else 0
    mAP, prec, rec = evaluate_predictions(all_preds, all_gts, num_classes=num_classes)
    return mAP, prec, rec, fps, model, all_preds


# ─────────────────────────────────────────────────────
# YOLOv8 Evaluation
# ─────────────────────────────────────────────────────
def evaluate_yolo(run_dir, yaml_path, imgsz=512):
    """Use YOLO's built-in val() and parse results."""
    from ultralytics import YOLO

    # Find best weights
    weights = Path(run_dir) / "weights" / "best.pt"
    if not weights.exists():
        weights = Path(run_dir) / "weights" / "last.pt"
    if not weights.exists():
        print(f"  [WARN] No YOLO weights found in {run_dir}")
        return 0, 0, 0, 0

    model = YOLO(str(weights))

    # Run validation on test set
    results = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=imgsz,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    mAP50 = float(results.box.map50) if hasattr(results.box, "map50") else 0
    precision = float(results.box.mp) if hasattr(results.box, "mp") else 0
    recall = float(results.box.mr) if hasattr(results.box, "mr") else 0

    # Measure inference speed
    test_imgs = list((Path(yaml_path).parent / "test" / "images").glob("*"))
    n = min(len(test_imgs), 50)
    start = time.time()
    for img_path in test_imgs[:n]:
        model.predict(str(img_path), imgsz=imgsz, verbose=False)
    elapsed = time.time() - start
    fps = n / elapsed if elapsed > 0 else 0

    return mAP50, precision, recall, fps


# ─────────────────────────────────────────────────────
# Visualization: draw predictions on sample images
# ─────────────────────────────────────────────────────
def draw_predictions(img_path, boxes, labels, scores, class_names, save_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box, lbl, score in zip(boxes, labels, scores):
        color = COLORS[int(lbl) % len(COLORS)]
        draw.rectangle(box, outline=color, width=3)
        name = class_names.get(int(lbl), f"cls{lbl}")
        text = f"{name} {score:.2f}"
        draw.text((box[0], max(0, box[1] - 15)), text, fill=color)

    img.save(save_path)


def generate_frcnn_predictions(model, test_loader, class_names, prefix, n=4):
    """Save example prediction images."""
    model.eval()
    count = 0
    with torch.no_grad():
        for images, targets, paths in test_loader:
            images_gpu = [img.to(DEVICE) for img in images]
            outputs = model(images_gpu)
            for out, path in zip(outputs, paths):
                if count >= n:
                    return
                mask = out["scores"] >= 0.5
                draw_predictions(
                    path,
                    out["boxes"][mask].cpu().numpy().tolist(),
                    out["labels"][mask].cpu().numpy().tolist(),
                    out["scores"][mask].cpu().numpy().tolist(),
                    class_names,
                    RESULTS_DIR / f"{prefix}_pred_{count}.jpg"
                )
                count += 1


def generate_yolo_predictions(run_dir, test_dir, prefix, n=4, imgsz=512):
    from ultralytics import YOLO
    weights = Path(run_dir) / "weights" / "best.pt"
    if not weights.exists():
        weights = Path(run_dir) / "weights" / "last.pt"
    model = YOLO(str(weights))
    imgs = sorted(Path(test_dir).glob("*"))[:n]
    for i, img_path in enumerate(imgs):
        results = model.predict(str(img_path), imgsz=imgsz, verbose=False)
        result_img = results[0].plot()
        Image.fromarray(result_img).save(RESULTS_DIR / f"{prefix}_pred_{i}.jpg")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    results_table = []

    # Load training histories
    def load_time(path):
        if Path(path).exists():
            with open(path) as f:
                h = json.load(f)
            return h.get("total_time", 0)
        return 0

    # ── Faster R-CNN Penn-Fudan ──
    print("=" * 60)
    print("Evaluating Faster R-CNN on Penn-Fudan (test)")
    print("=" * 60)

    pf_test = DetectionDataset(
        "data/pennfudan_split/test/images",
        "data/pennfudan_split/test/labels",
        transforms=get_transform(), label_offset=0,
    )
    pf_test_loader = DataLoader(pf_test, batch_size=2, collate_fn=DetectionDataset.collate_fn)

    frcnn_pf_weights = RESULTS_DIR / "frcnn_pennfudan_best.pth"
    if frcnn_pf_weights.exists():
        mAP, prec, rec, fps, frcnn_pf_model, _ = evaluate_fasterrcnn(
            frcnn_pf_weights, pf_test_loader, num_classes=2
        )
        train_time = load_time(RESULTS_DIR / "frcnn_pennfudan_history.json")
        results_table.append({
            "Dataset": "Penn-Fudan", "Model": "Faster R-CNN",
            "mAP@0.5": f"{mAP:.4f}", "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}", "Train Time (s)": f"{train_time:.0f}",
            "Inference (img/s)": f"{fps:.1f}"
        })
        generate_frcnn_predictions(
            frcnn_pf_model, pf_test_loader,
            {1: "person"}, "frcnn_pf", n=4
        )
        print(f"  mAP@0.5={mAP:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  FPS={fps:.1f}")
    else:
        print("  [SKIP] Weights not found.")

    # ── Faster R-CNN Pets ──
    print("\n" + "=" * 60)
    print("Evaluating Faster R-CNN on Pets (test)")
    print("=" * 60)

    pet_test = DetectionDataset(
        "data/pets_split/test/images",
        "data/pets_split/test/labels",
        transforms=get_transform(), label_offset=1,
    )
    pet_test_loader = DataLoader(pet_test, batch_size=2, collate_fn=DetectionDataset.collate_fn)

    frcnn_pet_weights = RESULTS_DIR / "frcnn_pets_best.pth"
    if frcnn_pet_weights.exists():
        breed_names = {i+1: b for i, b in enumerate(
            ["Abyssinian", "beagle", "boxer", "chihuahua", "persian"]
        )}
        mAP, prec, rec, fps, frcnn_pet_model, _ = evaluate_fasterrcnn(
            frcnn_pet_weights, pet_test_loader, num_classes=6
        )
        train_time = load_time(RESULTS_DIR / "frcnn_pets_history.json")
        results_table.append({
            "Dataset": "Pets (5 breeds)", "Model": "Faster R-CNN",
            "mAP@0.5": f"{mAP:.4f}", "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}", "Train Time (s)": f"{train_time:.0f}",
            "Inference (img/s)": f"{fps:.1f}"
        })
        generate_frcnn_predictions(
            frcnn_pet_model, pet_test_loader,
            breed_names, "frcnn_pets", n=4
        )
        print(f"  mAP@0.5={mAP:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  FPS={fps:.1f}")
    else:
        print("  [SKIP] Weights not found.")

    # ── YOLOv8n Penn-Fudan ──
    print("\n" + "=" * 60)
    print("Evaluating YOLOv8n on Penn-Fudan (test)")
    print("=" * 60)

    yolo_pf_run = RESULTS_DIR / "yolo_runs" / "pennfudan"
    pf_yolo_yaml = Path("data/pennfudan_split_yolo/pennfudan.yaml")
    if yolo_pf_run.exists() and pf_yolo_yaml.exists():
        mAP, prec, rec, fps = evaluate_yolo(yolo_pf_run, pf_yolo_yaml)
        train_time = load_time(RESULTS_DIR / "yolo_pennfudan_history.json")
        results_table.append({
            "Dataset": "Penn-Fudan", "Model": "YOLOv8n",
            "mAP@0.5": f"{mAP:.4f}", "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}", "Train Time (s)": f"{train_time:.0f}",
            "Inference (img/s)": f"{fps:.1f}"
        })
        generate_yolo_predictions(
            yolo_pf_run, "data/pennfudan_split_yolo/test/images", "yolo_pf", n=4
        )
        print(f"  mAP@0.5={mAP:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  FPS={fps:.1f}")
    else:
        print("  [SKIP] YOLO run not found.")

    # ── YOLOv8n Pets ──
    print("\n" + "=" * 60)
    print("Evaluating YOLOv8n on Pets (test)")
    print("=" * 60)

    yolo_pet_run = RESULTS_DIR / "yolo_runs" / "pets"
    pet_yolo_yaml = Path("data/pets_split_yolo/pets.yaml")
    if yolo_pet_run.exists() and pet_yolo_yaml.exists():
        mAP, prec, rec, fps = evaluate_yolo(yolo_pet_run, pet_yolo_yaml)
        train_time = load_time(RESULTS_DIR / "yolo_pets_history.json")
        results_table.append({
            "Dataset": "Pets (5 breeds)", "Model": "YOLOv8n",
            "mAP@0.5": f"{mAP:.4f}", "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}", "Train Time (s)": f"{train_time:.0f}",
            "Inference (img/s)": f"{fps:.1f}"
        })
        generate_yolo_predictions(
            yolo_pet_run, "data/pets_split_yolo/test/images", "yolo_pets", n=4
        )
        print(f"  mAP@0.5={mAP:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  FPS={fps:.1f}")
    else:
        print("  [SKIP] YOLO run not found.")

    # ── Print Comparison Table ──
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    if results_table:
        df = pd.DataFrame(results_table)
        print(df.to_string(index=False))
        df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
        print(f"\nSaved to {RESULTS_DIR / 'comparison_table.csv'}")
    else:
        print("No results to compare — make sure training completed first.")

    print("\nEvaluation complete! Check results/ for all outputs.")
