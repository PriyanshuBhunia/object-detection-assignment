"""
1_setup_datasets.py
Downloads and prepares:
  - Penn-Fudan Pedestrian Dataset
  - Oxford-IIIT Pet Dataset (subset of 5 breeds)
Splits each into 70/15/15 train/val/test.
"""

import os
import shutil
import random
import json
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

random.seed(42)

BASE_DIR = Path("data")
BASE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────
# 1. Penn-Fudan Pedestrian Dataset
# ─────────────────────────────────────────────────────
def setup_pennfudan():
    pf_dir = BASE_DIR / "PennFudan"
    if pf_dir.exists():
        print("[Penn-Fudan] Already downloaded.")
    else:
        url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
        zip_path = BASE_DIR / "PennFudanPed.zip"
        print("[Penn-Fudan] Downloading...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(BASE_DIR)
        os.rename(BASE_DIR / "PennFudanPed", pf_dir)
        zip_path.unlink()
        print("[Penn-Fudan] Extracted.")

    # Parse masks → bounding boxes
    img_dir = pf_dir / "PNGImages"
    mask_dir = pf_dir / "PedMasks"
    images = sorted(img_dir.glob("*.png"))

    annotations = {}
    for img_path in images:
        name = img_path.stem
        mask_path = mask_dir / f"{name}_mask.png"
        mask = np.array(Image.open(mask_path))
        obj_ids = np.unique(mask)[1:]  # skip background 0
        boxes = []
        for oid in obj_ids:
            pos = np.where(mask == oid)
            ymin, ymax = int(pos[0].min()), int(pos[0].max())
            xmin, xmax = int(pos[1].min()), int(pos[1].max())
            if (xmax - xmin) > 2 and (ymax - ymin) > 2:
                boxes.append([xmin, ymin, xmax, ymax])
        annotations[img_path.name] = {"boxes": boxes, "labels": [1]*len(boxes)}

    # Split
    all_imgs = list(annotations.keys())
    random.shuffle(all_imgs)
    n = len(all_imgs)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train+n_val],
        "test": all_imgs[n_train+n_val:]
    }

    out_dir = BASE_DIR / "pennfudan_split"
    for split, files in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(img_dir / f, out_dir / split / "images" / f)
            ann = annotations[f]
            # Save as JSON for Faster R-CNN
            with open(out_dir / split / "labels" / f"{Path(f).stem}.json", "w") as fp:
                json.dump(ann, fp)
            # Also save YOLO format labels
            img = Image.open(img_dir / f)
            w, h = img.size
            yolo_dir = out_dir / split / "labels_yolo"
            yolo_dir.mkdir(exist_ok=True)
            with open(yolo_dir / f"{Path(f).stem}.txt", "w") as fp:
                for box in ann["boxes"]:
                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    fp.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[Penn-Fudan] Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Create YOLO dataset.yaml
    yolo_yaml = out_dir / "pennfudan.yaml"
    abs_path = out_dir.resolve()
    with open(yolo_yaml, "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("names:\n  0: person\n")

    # Symlink yolo labels so they're next to images (YOLO convention)
    for split in ["train", "val", "test"]:
        yolo_lbl = out_dir / split / "labels_yolo"
        # YOLO expects labels/ next to images/
        # We already created labels_yolo, let's just copy them as the expected path
        target = out_dir / split / "yolo_labels"
        if not target.exists():
            shutil.copytree(yolo_lbl, target)

    print("[Penn-Fudan] Done.")
    return out_dir


# ─────────────────────────────────────────────────────
# 2. Oxford-IIIT Pet Dataset (Subset: 5 breeds)
# ─────────────────────────────────────────────────────
SELECTED_BREEDS = [
    "Abyssinian",
    "beagle",
    "boxer",
    "chihuahua",
    "persian",
]

def setup_pets():
    pets_dir = BASE_DIR / "oxford_pets"
    pets_dir.mkdir(exist_ok=True)

    images_tar = BASE_DIR / "images.tar.gz"
    annots_tar = BASE_DIR / "annotations.tar.gz"

    if not (pets_dir / "images").exists():
        print("[Pets] Downloading images...")
        urllib.request.urlretrieve(
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            images_tar
        )
        print("[Pets] Downloading annotations...")
        urllib.request.urlretrieve(
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            annots_tar
        )
        with tarfile.open(images_tar) as t:
            t.extractall(pets_dir)
        with tarfile.open(annots_tar) as t:
            t.extractall(pets_dir)
        images_tar.unlink()
        annots_tar.unlink()
        print("[Pets] Extracted.")
    else:
        print("[Pets] Already downloaded.")

    # Parse XML annotations for selected breeds
    xmlanno_dir = pets_dir / "annotations" / "xmls"
    img_dir = pets_dir / "images"

    breed_to_id = {b: i for i, b in enumerate(SELECTED_BREEDS)}
    annotations = {}

    for xml_file in sorted(xmlanno_dir.glob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        if not filename.endswith(".jpg"):
            filename += ".jpg"

        # Determine breed from filename (format: Breed_Name_123.jpg)
        name_parts = filename.rsplit("_", 1)[0]  # remove trailing number
        breed_match = None
        for breed in SELECTED_BREEDS:
            if name_parts.lower() == breed.lower() or name_parts.lower().startswith(breed.lower()):
                breed_match = breed
                break
        if breed_match is None:
            continue

        # Get bounding box
        obj = root.find("object")
        if obj is None:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        box = [
            int(bndbox.find("xmin").text),
            int(bndbox.find("ymin").text),
            int(bndbox.find("xmax").text),
            int(bndbox.find("ymax").text),
        ]
        label_id = breed_to_id[breed_match]
        annotations[filename] = {"boxes": [box], "labels": [label_id]}

    # Split
    all_imgs = list(annotations.keys())
    random.shuffle(all_imgs)
    n = len(all_imgs)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train+n_val],
        "test": all_imgs[n_train+n_val:]
    }

    out_dir = BASE_DIR / "pets_split"
    for split, files in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for f in files:
            src = img_dir / f
            if not src.exists():
                continue
            shutil.copy(src, out_dir / split / "images" / f)
            ann = annotations[f]
            with open(out_dir / split / "labels" / f"{Path(f).stem}.json", "w") as fp:
                json.dump(ann, fp)

            # YOLO format
            img = Image.open(src)
            w, h = img.size
            yolo_dir = out_dir / split / "labels_yolo"
            yolo_dir.mkdir(exist_ok=True)
            with open(yolo_dir / f"{Path(f).stem}.txt", "w") as fp:
                for box, lbl in zip(ann["boxes"], ann["labels"]):
                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    fp.write(f"{lbl} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[Pets] Selected breeds: {SELECTED_BREEDS}")
    print(f"[Pets] Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # YOLO yaml
    yolo_yaml = out_dir / "pets.yaml"
    abs_path = out_dir.resolve()
    with open(yolo_yaml, "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("names:\n")
        for i, breed in enumerate(SELECTED_BREEDS):
            f.write(f"  {i}: {breed}\n")

    # Save breed mapping
    with open(out_dir / "breed_map.json", "w") as f:
        json.dump(breed_to_id, f)

    print("[Pets] Done.")
    return out_dir


if __name__ == "__main__":
    print("=" * 60)
    print("Setting up datasets...")
    print("=" * 60)
    setup_pennfudan()
    print()
    setup_pets()
    print("\nAll datasets ready!")
