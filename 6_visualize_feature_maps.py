"""
6_visualize_feature_maps.py
Visualizes intermediate feature maps from Faster R-CNN's MobileNetV3 backbone.
Bonus: Also compares predictions at different image sizes (512 vs 640).
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────
# Feature Map Extraction Hook
# ─────────────────────────────────────────────────────
class FeatureMapExtractor:
    """Hooks into backbone layers to capture feature maps."""

    def __init__(self, model):
        self.features = {}
        self.hooks = []

        # Hook into the backbone's FPN output layers
        backbone = model.backbone

        # Register hooks on the body (MobileNetV3 layers)
        for name, layer in backbone.body.named_children():
            hook = layer.register_forward_hook(self._make_hook(f"body_{name}"))
            self.hooks.append(hook)

        # Register hooks on FPN layers
        for name, layer in backbone.fpn.named_children():
            hook = layer.register_forward_hook(self._make_hook(f"fpn_{name}"))
            self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach().cpu()
        return hook

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def visualize_feature_maps(feature_tensor, title, save_path, num_channels=16):
    """Visualize the first N channels of a feature map as a grid."""
    if feature_tensor.dim() == 4:
        feature_tensor = feature_tensor[0]  # take first image in batch

    n_channels = min(num_channels, feature_tensor.shape[0])
    cols = 4
    rows = (n_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        if i < n_channels:
            fmap = feature_tensor[i].numpy()
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Ch {i}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def run_feature_map_visualization(weights_path, img_path, num_classes, prefix):
    """Load model, run one image, visualize extracted feature maps."""
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    extractor = FeatureMapExtractor(model)

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        _ = model(img_tensor)

    # Visualize select layers
    saved_count = 0
    for name, feat in extractor.features.items():
        if saved_count >= 4:  # limit to 4 layers
            break
        if feat.dim() >= 3 and feat.shape[-1] > 1 and feat.shape[-2] > 1:
            visualize_feature_maps(
                feat,
                f"{prefix} — Layer: {name} (shape: {list(feat.shape)})",
                RESULTS_DIR / f"{prefix}_featuremap_{name}.png"
            )
            saved_count += 1

    extractor.remove_hooks()
    print(f"  Visualized {saved_count} layers for {prefix}")


# ─────────────────────────────────────────────────────
# Bonus: Compare Image Sizes (512 vs 640)
# ─────────────────────────────────────────────────────
def compare_image_sizes(weights_path, img_path, num_classes, prefix):
    """Run inference at 512 and 640 resolution, compare detections."""
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    sizes = [512, 640]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, size in enumerate(sizes):
        img = Image.open(img_path).convert("RGB")
        transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)[0]

        # Filter by confidence
        mask = output["scores"] >= 0.5
        boxes = output["boxes"][mask].cpu().numpy()
        scores = output["scores"][mask].cpu().numpy()

        # Draw on image
        img_resized = img.resize((size, size))
        draw = ImageDraw.Draw(img_resized)
        for box, score in zip(boxes, scores):
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], max(0, box[1] - 12)), f"{score:.2f}", fill="red")

        axes[idx].imshow(np.array(img_resized))
        axes[idx].set_title(f"{size}x{size} — {len(boxes)} detections", fontsize=12)
        axes[idx].axis("off")

    fig.suptitle(f"{prefix} — Image Size Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = RESULTS_DIR / f"{prefix}_size_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Feature Map Visualization + Image Size Comparison")
    print("=" * 50)

    # Find a test image for each dataset
    pf_test_imgs = list(Path("data/pennfudan_split/test/images").glob("*"))
    pet_test_imgs = list(Path("data/pets_split/test/images").glob("*"))

    # --- Penn-Fudan ---
    pf_weights = RESULTS_DIR / "frcnn_pennfudan_best.pth"
    if pf_weights.exists() and pf_test_imgs:
        print("\n[Penn-Fudan] Feature maps...")
        run_feature_map_visualization(pf_weights, pf_test_imgs[0], 2, "pennfudan")

        print("[Penn-Fudan] Image size comparison...")
        compare_image_sizes(pf_weights, pf_test_imgs[0], 2, "pennfudan")
    else:
        print("[SKIP] Penn-Fudan weights or test images not found.")

    # --- Oxford Pets ---
    pet_weights = RESULTS_DIR / "frcnn_pets_best.pth"
    if pet_weights.exists() and pet_test_imgs:
        print("\n[Pets] Feature maps...")
        run_feature_map_visualization(pet_weights, pet_test_imgs[0], 6, "pets")

        print("[Pets] Image size comparison...")
        compare_image_sizes(pet_weights, pet_test_imgs[0], 6, "pets")
    else:
        print("[SKIP] Pets weights or test images not found.")

    print("\nAll bonus visualizations complete! Check results/")
