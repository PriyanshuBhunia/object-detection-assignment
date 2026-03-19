# Object Detection Assignment — Faster R-CNN vs YOLOv8n

## Setup

```bash
pip install torch torchvision ultralytics pycocotools matplotlib pandas Pillow tqdm scikit-learn
```

## Run Order

```bash
# 1. Download & prepare both datasets
python 1_setup_datasets.py

# 2. Train Faster R-CNN on both datasets
python 2_train_fasterrcnn.py

# 3. Train YOLOv8n on both datasets
python 3_train_yolov8.py

# 4. Generate evaluation comparison table + example predictions
python 4_evaluate_compare.py

# 5. Plot training loss curves
python 5_plot_loss_curves.py

# 6. (Bonus) Visualize feature maps + compare image sizes
python 6_visualize_feature_maps.py
```

All results (metrics, plots, predictions) are saved to `results/`.

## Datasets

- **Penn-Fudan Pedestrian** — ~170 images, 1 class (person)
- **Oxford-IIIT Pet Subset** — 5 breeds (Abyssinian, Beagle, Boxer, Chihuahua, Persian)

## Models

- **Faster R-CNN** — MobileNetV3-Large-FPN backbone (two-stage detector)
- **YOLOv8n** — Nano variant (single-stage detector)

## Bonus Features

- Training loss curve visualization
- Feature map visualization from backbone layers
- Image size comparison (512 vs 640)
- Data augmentation (horizontal flip, color jitter, YOLO built-in mosaic/mixup)
