# Wessley Schematic ML - Connector & Ground Detection

Phase 2: Train a detector for connectors and grounds on synthetic diagrams.

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install -e ..

# ML dependencies
pip install ultralytics opencv-python

# For PNG generation (optional but recommended)
pip install cairosvg
```

### 2. Generate Training Dataset

```bash
# Generate 1000 train + 200 val samples with base preset
python scripts/prepare_dataset.py \
    --output-dir ./dataset_yolo \
    --num-train 1000 \
    --num-val 200 \
    --preset base \
    --seed 42
```

This creates:
```
dataset_yolo/
├── images/
│   ├── train/     # 1000 PNG images
│   └── val/       # 200 PNG images
├── labels/
│   ├── train/     # 1000 YOLO label files
│   └── val/       # 200 YOLO label files
└── dataset.yaml   # YOLO dataset config
```

### 3. Train the Model

```bash
# Train YOLOv11-nano (fast, ~5 min)
python scripts/train_detector.py train \
    --dataset ./dataset_yolo/dataset.yaml \
    --model-size n \
    --epochs 100 \
    --batch 16

# Or train a larger model for better accuracy
python scripts/train_detector.py train \
    --dataset ./dataset_yolo/dataset.yaml \
    --model-size s \
    --epochs 100
```

### 4. Evaluate

```bash
# Validate on val set
python scripts/train_detector.py validate \
    --model runs/detect/harness_detector/weights/best.pt \
    --dataset ./dataset_yolo/dataset.yaml

# Visualize predictions
python scripts/visualize_predictions.py predict \
    --model runs/detect/harness_detector/weights/best.pt \
    --images ./dataset_yolo/images/val \
    --output ./visualizations \
    --num-samples 50
```

### 5. Compare with Ground Truth

```bash
python scripts/visualize_predictions.py compare \
    --model runs/detect/harness_detector/weights/best.pt \
    --images ./dataset_yolo/images/val \
    --labels ./dataset_yolo/labels/val \
    --output ./comparisons
```

## Success Criteria (Phase 2)

- **mAP50 > 0.95** on synthetic validation set
- Visually correct detections on 50+ random samples
- Model file ready for domain gap testing

## Directory Structure

```
ml/
├── README.md               # This file
├── scripts/
│   ├── prepare_dataset.py  # Generate YOLO dataset
│   ├── train_detector.py   # Train YOLOv8/11
│   └── visualize_predictions.py
├── configs/                # (future) model configs
└── runs/                   # Training outputs (gitignored)
    └── detect/
        └── harness_detector/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            └── results.csv
```

## Next Steps After This Phase

1. **Domain Gap Test**: Run on 1-2 real OEM diagrams to see how badly it fails
2. **Style Tuning**: If synthetic style is too different, adjust renderer
3. **Wire Detection**: Add wire/line segment detection
4. **Full Graph**: Reconstruct connectivity from detections
