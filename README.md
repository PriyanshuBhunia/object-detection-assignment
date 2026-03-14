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
```

All results (metrics, plots, predictions) are saved to `results/`.
