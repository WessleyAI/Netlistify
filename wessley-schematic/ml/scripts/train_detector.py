#!/usr/bin/env python3
"""
Train YOLOv8 detector on synthetic harness diagrams.

Phase 1: Detect connectors and grounds only.
"""

import argparse
from pathlib import Path


def train_yolo(
    dataset_yaml: Path,
    model_size: str = "n",  # nano, small, medium, large, xlarge
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "harness_detector",
    resume: bool = False,
) -> None:
    """Train YOLOv8 on the harness detection dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed.")
        print("Install with: pip install ultralytics")
        return

    # Select model
    model_name = f"yolo11{model_size}.pt"  # YOLO11 is latest
    print(f"Loading model: {model_name}")

    model = YOLO(model_name)

    # Train
    print(f"\nStarting training:")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Output: {project}/{name}")

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        resume=resume,
        # Augmentation settings (conservative for schematic diagrams)
        hsv_h=0.0,  # No hue shift (diagrams are typically B&W)
        hsv_s=0.0,  # No saturation shift
        hsv_v=0.2,  # Slight brightness variation
        degrees=0.0,  # No rotation (diagrams are axis-aligned)
        translate=0.1,  # Small translation
        scale=0.2,  # Some scale variation
        shear=0.0,  # No shear
        flipud=0.0,  # No vertical flip
        fliplr=0.0,  # No horizontal flip (component positions matter)
        mosaic=0.5,  # Mosaic augmentation
        mixup=0.0,  # No mixup
        # Early stopping
        patience=20,
        # Save settings
        save=True,
        save_period=10,
        # Logging
        verbose=True,
    )

    print(f"\nTraining complete!")
    print(f"Best model: {project}/{name}/weights/best.pt")
    print(f"Results: {project}/{name}/results.csv")

    return results


def validate_model(
    model_path: Path,
    dataset_yaml: Path,
    split: str = "val",
    imgsz: int = 640,
) -> None:
    """Validate a trained model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed.")
        return

    model = YOLO(str(model_path))
    results = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        verbose=True,
    )

    print(f"\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")

    # Per-class results
    names = results.names
    for i, (p, r, map50) in enumerate(zip(
        results.box.p, results.box.r, results.box.ap50
    )):
        print(f"  {names[i]}: P={p:.3f}, R={r:.3f}, mAP50={map50:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 detector for harness components"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        required=True,
        help="Path to dataset.yaml",
    )
    train_parser.add_argument(
        "--model-size",
        "-m",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )
    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Number of epochs",
    )
    train_parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=16,
        help="Batch size",
    )
    train_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    train_parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory",
    )
    train_parser.add_argument(
        "--name",
        type=str,
        default="harness_detector",
        help="Experiment name",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a trained model")
    val_parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="Path to trained model (.pt)",
    )
    val_parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        required=True,
        help="Path to dataset.yaml",
    )
    val_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to validate on",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_yolo(
            dataset_yaml=args.dataset,
            model_size=args.model_size,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            resume=args.resume,
        )
    elif args.command == "validate":
        validate_model(
            model_path=args.model,
            dataset_yaml=args.dataset,
            split=args.split,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
