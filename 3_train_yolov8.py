"""
3_train_yolov8.py
Trains YOLOv8n (Nano) on:
  - Penn-Fudan Pedestrian Dataset
  - Oxford-IIIT Pet Subset (5 breeds)

Requires: pip install ultralytics
Before running, YOLO labels must be in place. The setup script creates them.
We need to restructure slightly so YOLO finds labels next to images.
"""

import os
import shutil
import time
import json
from pathlib import Path
from ultralytics import YOLO

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def prepare_yolo_structure(base_dir):
    """
    YOLO expects:
        split/images/img.png
        split/labels/img.txt   <-- same name, .txt
    We created labels_yolo/ — need to copy to labels/ (overwrite the JSON ones for YOLO).
    We'll create a separate yolo directory to avoid conflict.
    """
    yolo_dir = base_dir.parent / f"{base_dir.name}_yolo"
    if yolo_dir.exists():
        return yolo_dir

    for split in ["train", "val", "test"]:
        src_imgs = base_dir / split / "images"
        src_lbls = base_dir / split / "labels_yolo"

        dst_imgs = yolo_dir / split / "images"
        dst_lbls = yolo_dir / split / "labels"

        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        # Copy images
        for f in src_imgs.iterdir():
            shutil.copy(f, dst_imgs / f.name)
        # Copy YOLO labels
        if src_lbls.exists():
            for f in src_lbls.iterdir():
                shutil.copy(f, dst_lbls / f.name)

    return yolo_dir


def create_yolo_yaml(yolo_dir, name, class_names):
    yaml_path = yolo_dir / f"{name}.yaml"
    abs_path = yolo_dir.resolve()
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("names:\n")
        for i, cn in enumerate(class_names):
            f.write(f"  {i}: {cn}\n")
    return yaml_path


def train_yolo(yaml_path, epochs, save_name, batch=8, imgsz=512):
    model = YOLO("yolov8n.pt")  # loads pretrained nano weights

    start = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if __import__("torch").cuda.is_available() else "cpu",
        project=str(RESULTS_DIR / "yolo_runs"),
        name=save_name,
        patience=5,        # early stopping
        augment=True,       # built-in augmentation
        verbose=True,
    )
    total_time = time.time() - start

    # Save timing info
    history = {"total_time": total_time, "epochs": epochs, "imgsz": imgsz, "batch": batch}
    with open(RESULTS_DIR / f"yolo_{save_name}_history.json", "w") as f:
        json.dump(history, f)

    print(f"  Training time: {total_time:.1f}s")
    return model, results


if __name__ == "__main__":
    # --- Penn-Fudan ---
    print("=" * 60)
    print("Training YOLOv8n on Penn-Fudan")
    print("=" * 60)

    pf_yolo = prepare_yolo_structure(Path("data/pennfudan_split"))
    pf_yaml = create_yolo_yaml(pf_yolo, "pennfudan", ["person"])
    pf_model, pf_results = train_yolo(pf_yaml, epochs=15, save_name="pennfudan", batch=8)

    # --- Oxford Pets ---
    print("\n" + "=" * 60)
    print("Training YOLOv8n on Oxford Pets (5 breeds)")
    print("=" * 60)

    pet_yolo = prepare_yolo_structure(Path("data/pets_split"))
    breeds = ["Abyssinian", "beagle", "boxer", "chihuahua", "persian"]
    pet_yaml = create_yolo_yaml(pet_yolo, "pets", breeds)
    pet_model, pet_results = train_yolo(pet_yaml, epochs=20, save_name="pets", batch=8)

    print("\nYOLOv8n training complete!")
