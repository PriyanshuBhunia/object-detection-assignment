"""
5_plot_loss_curves.py
Plots training and validation loss curves for Faster R-CNN on both datasets.
YOLOv8 loss curves are auto-saved by Ultralytics in the run directory.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")


def plot_frcnn_loss(history_path, title, save_path):
    with open(history_path) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist["train_loss"], "b-o", label="Training Loss", markersize=5)
    plt.plot(epochs, hist["val_loss"], "r-o", label="Validation Loss", markersize=5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def copy_yolo_plots():
    """YOLO auto-saves results.png in each run folder — copy them to results/."""
    for name in ["pennfudan", "pets"]:
        src = RESULTS_DIR / "yolo_runs" / name / "results.png"
        dst = RESULTS_DIR / f"yolo_{name}_loss_curves.png"
        if src.exists():
            import shutil
            shutil.copy(src, dst)
            print(f"Copied YOLO curves: {dst}")
        else:
            print(f"[SKIP] YOLO results not found: {src}")


if __name__ == "__main__":
    print("=" * 50)
    print("Plotting loss curves")
    print("=" * 50)

    # Faster R-CNN
    pf_hist = RESULTS_DIR / "frcnn_pennfudan_history.json"
    if pf_hist.exists():
        plot_frcnn_loss(pf_hist, "Faster R-CNN — Penn-Fudan Loss Curves",
                        RESULTS_DIR / "frcnn_pennfudan_loss.png")
    else:
        print("[SKIP] Penn-Fudan Faster R-CNN history not found.")

    pet_hist = RESULTS_DIR / "frcnn_pets_history.json"
    if pet_hist.exists():
        plot_frcnn_loss(pet_hist, "Faster R-CNN — Oxford Pets Loss Curves",
                        RESULTS_DIR / "frcnn_pets_loss.png")
    else:
        print("[SKIP] Pets Faster R-CNN history not found.")

    # YOLO
    copy_yolo_plots()

    print("\nDone! Check results/ for all plots.")
