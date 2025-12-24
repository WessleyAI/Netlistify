#!/usr/bin/env python3
"""
Generate a YOLO-format dataset for training connector + ground detection.

Creates a properly structured dataset with train/val split:
    dataset_yolo/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from wessley_schematic.synthetic import (
    GraphGeneratorConfig,
    HarnessPreset,
    generate_harness_graph,
    layout_harness,
    render_svg,
    export_labels,
)
from wessley_schematic.synthetic.config import LayoutConfig, RenderConfig


# For Phase 1: Only detect connectors and grounds
TARGET_CLASSES = {
    "connector": 0,
    "ground": 1,
}


def generate_yolo_dataset(
    output_dir: Path,
    num_train: int = 1000,
    num_val: int = 200,
    preset: str = "base",
    seed: int = 42,
    generate_png: bool = True,
) -> None:
    """Generate YOLO-format dataset with train/val split."""
    output_dir = Path(output_dir)

    # Create directory structure
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Temp directory for intermediate files
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)

    random.seed(seed)

    # Get preset config
    harness_preset = HarnessPreset(preset)

    layout_config = LayoutConfig()
    render_config = RenderConfig(
        width=640,  # Standard YOLO size
        height=640,
        generate_png=generate_png,
    )

    print(f"Generating {num_train} training samples...")
    _generate_split(
        output_dir=output_dir,
        temp_dir=temp_dir,
        split="train",
        num_samples=num_train,
        preset=harness_preset,
        layout_config=layout_config,
        render_config=render_config,
        seed=seed,
    )

    print(f"Generating {num_val} validation samples...")
    _generate_split(
        output_dir=output_dir,
        temp_dir=temp_dir,
        split="val",
        num_samples=num_val,
        preset=harness_preset,
        layout_config=layout_config,
        render_config=render_config,
        seed=seed + num_train,  # Different seed for val
    )

    # Create dataset.yaml
    yaml_content = f"""# Wessley Schematic Detection Dataset
# Auto-generated for connector + ground detection

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes (Phase 1: connectors and grounds only)
names:
  0: connector
  1: ground

# Dataset info
nc: 2
"""

    (output_dir / "dataset.yaml").write_text(yaml_content)

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nDataset created at: {output_dir}")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {num_val}")
    print(f"  Config: {output_dir / 'dataset.yaml'}")


def _generate_split(
    output_dir: Path,
    temp_dir: Path,
    split: str,
    num_samples: int,
    preset: HarnessPreset,
    layout_config: LayoutConfig,
    render_config: RenderConfig,
    seed: int,
) -> None:
    """Generate samples for a single split."""
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    temp_labels = temp_dir / "labels"
    temp_svgs = temp_dir / "svgs"
    temp_labels.mkdir(exist_ok=True)
    temp_svgs.mkdir(exist_ok=True)

    for i in range(num_samples):
        sample_name = f"{split}_{i:05d}"

        # Generate harness
        config = GraphGeneratorConfig.from_preset(preset, seed=seed + i)
        harness = generate_harness_graph(config)
        harness, layout_geometry = layout_harness(harness, layout_config)

        # Render
        svg_path = temp_svgs / f"{sample_name}.svg"
        render_geometry = render_svg(harness, layout_geometry, svg_path, render_config)

        # Export labels (to temp)
        export_labels(
            harness,
            layout_geometry,
            render_geometry,
            temp_labels,
            sample_name,
            render_config,
        )

        # Convert to YOLO format (filter to only target classes)
        _convert_to_yolo(
            temp_labels / f"{sample_name}.objects.json",
            labels_dir / f"{sample_name}.txt",
            render_config.width,
            render_config.height,
        )

        # Move image (PNG or SVG)
        if render_config.generate_png:
            png_path = svg_path.with_suffix(".png")
            if png_path.exists():
                shutil.copy(png_path, images_dir / f"{sample_name}.png")
        else:
            # Use SVG as image (some tools can handle it)
            shutil.copy(svg_path, images_dir / f"{sample_name}.svg")

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples}")


def _convert_to_yolo(
    objects_json_path: Path,
    yolo_txt_path: Path,
    img_width: int,
    img_height: int,
) -> None:
    """Convert objects.json to YOLO format, filtering to target classes only."""
    with open(objects_json_path) as f:
        data = json.load(f)

    lines = []
    for obj in data.get("objects", []):
        class_name = obj.get("class", "")
        if class_name not in TARGET_CLASSES:
            continue

        class_id = TARGET_CLASSES[class_name]
        bbox = obj.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Convert to YOLO format (center x, center y, width, height) normalized
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(yolo_txt_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO-format dataset for connector + ground detection"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./dataset_yolo"),
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--num-train",
        "-t",
        type=int,
        default=1000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num-val",
        "-v",
        type=int,
        default=200,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--preset",
        "-p",
        choices=["tiny", "base", "complex"],
        default="base",
        help="Harness complexity preset",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG generation (use SVG only)",
    )

    args = parser.parse_args()

    generate_yolo_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        preset=args.preset,
        seed=args.seed,
        generate_png=not args.no_png,
    )


if __name__ == "__main__":
    main()
