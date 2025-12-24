#!/usr/bin/env python3
"""
Visualize model predictions on sample images.

Overlays predicted bounding boxes on images for manual inspection.
"""

import argparse
import random
from pathlib import Path


def visualize_predictions(
    model_path: Path,
    images_dir: Path,
    output_dir: Path,
    num_samples: int = 50,
    conf_threshold: float = 0.25,
    seed: int = 42,
) -> None:
    """Run model on sample images and save visualizations."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Get image files
    images_dir = Path(images_dir)
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # Sample random images
    random.seed(seed)
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)

    print(f"Running inference on {len(image_files)} images...")

    # Run predictions
    results = model.predict(
        source=image_files,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
    )

    print(f"\nPredictions saved to: {output_dir}/predictions")

    # Summary statistics
    total_connectors = 0
    total_grounds = 0
    total_images_with_detections = 0

    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            total_images_with_detections += 1
            for cls in boxes.cls:
                if int(cls) == 0:
                    total_connectors += 1
                elif int(cls) == 1:
                    total_grounds += 1

    print(f"\nSummary:")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Images with detections: {total_images_with_detections}")
    print(f"  Total connectors detected: {total_connectors}")
    print(f"  Total grounds detected: {total_grounds}")
    print(f"  Avg connectors/image: {total_connectors / len(image_files):.1f}")
    print(f"  Avg grounds/image: {total_grounds / len(image_files):.1f}")


def compare_predictions_vs_labels(
    model_path: Path,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    num_samples: int = 20,
    conf_threshold: float = 0.25,
) -> None:
    """Compare model predictions against ground truth labels."""
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install ultralytics opencv-python")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_files = sorted(images_dir.glob("*.png"))[:num_samples]

    print(f"Comparing {len(image_files)} samples...")

    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Load ground truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        gt_boxes.append((cls, x1, y1, x2, y2))

        # Run prediction
        results = model.predict(img, conf=conf_threshold, verbose=False)
        pred_boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            pred_boxes.append((cls, x1, y1, x2, y2, conf))

        # Draw comparison
        vis = img.copy()

        # Ground truth in green
        for cls, x1, y1, x2, y2 in gt_boxes:
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = "C" if cls == 0 else "G"
            cv2.putText(vis, f"GT:{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Predictions in red/blue
        for cls, x1, y1, x2, y2, conf in pred_boxes:
            color = (255, 0, 0) if cls == 0 else (0, 0, 255)  # Blue for connector, Red for ground
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = "C" if cls == 0 else "G"
            cv2.putText(vis, f"P:{label}:{conf:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save
        cv2.imwrite(str(output_dir / f"{img_path.stem}_compare.png"), vis)

    print(f"Comparison images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Run predictions and visualize")
    pred_parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    pred_parser.add_argument(
        "--images",
        "-i",
        type=Path,
        required=True,
        help="Directory with images",
    )
    pred_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./visualizations"),
        help="Output directory",
    )
    pred_parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=50,
        help="Number of samples to visualize",
    )
    pred_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )

    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Compare predictions vs ground truth")
    comp_parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    comp_parser.add_argument(
        "--images",
        "-i",
        type=Path,
        required=True,
        help="Directory with images",
    )
    comp_parser.add_argument(
        "--labels",
        "-l",
        type=Path,
        required=True,
        help="Directory with label files",
    )
    comp_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./comparisons"),
        help="Output directory",
    )
    comp_parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=20,
        help="Number of samples",
    )

    args = parser.parse_args()

    if args.command == "predict":
        visualize_predictions(
            model_path=args.model,
            images_dir=args.images,
            output_dir=args.output,
            num_samples=args.num_samples,
            conf_threshold=args.conf,
        )
    elif args.command == "compare":
        compare_predictions_vs_labels(
            model_path=args.model,
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            num_samples=args.num_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
