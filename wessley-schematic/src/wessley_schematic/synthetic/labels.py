"""
Label exporter for ML training.

Exports harness diagram annotations in formats suitable for:
- Object detection (YOLO/COCO-style bounding boxes)
- Wire/line segment prediction
- Full graph reconstruction
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wessley_schematic.schemas.graph import HarnessGraph
from wessley_schematic.synthetic.config import RenderConfig
from wessley_schematic.synthetic.layout import LayoutGeometry
from wessley_schematic.synthetic.renderer_svg import RenderGeometry


def export_labels(
    harness: HarnessGraph,
    layout_geometry: LayoutGeometry,
    render_geometry: RenderGeometry,
    output_dir: Path | str,
    sample_name: str,
    render_config: RenderConfig | None = None,
) -> dict[str, Path]:
    """
    Export all label files for a harness diagram.

    Args:
        harness: The harness graph
        layout_geometry: Layout geometry from layout_harness()
        render_geometry: Render geometry from render_svg()
        output_dir: Directory to write label files
        sample_name: Base name for the sample (e.g., "synthetic_0001")
        render_config: Render configuration for coordinate scaling

    Returns:
        Dict mapping label type to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if render_config is None:
        render_config = RenderConfig()

    # Calculate scaling
    scale_x = render_config.width / layout_geometry.width
    scale_y = render_config.height / layout_geometry.height

    def scale_bbox(
        bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

    def scale_point(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (x * scale_x, y * scale_y)

    output_files = {}

    # 1. Export object detection labels
    objects_path = output_dir / f"{sample_name}.objects.json"
    objects_data = _export_object_labels(
        harness, layout_geometry, scale_bbox, sample_name, render_config
    )
    with open(objects_path, "w") as f:
        json.dump(objects_data, f, indent=2)
    output_files["objects"] = objects_path

    # 2. Export wire labels
    wires_path = output_dir / f"{sample_name}.wires.json"
    wires_data = _export_wire_labels(
        harness, render_geometry, layout_geometry, scale_point
    )
    with open(wires_path, "w") as f:
        json.dump(wires_data, f, indent=2)
    output_files["wires"] = wires_path

    # 3. Export full graph JSON
    graph_path = output_dir / f"{sample_name}.graph.json"
    graph_data = _export_graph_json(harness)
    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    output_files["graph"] = graph_path

    # 4. Export YOLO-format labels (normalized coordinates)
    yolo_path = output_dir / f"{sample_name}.txt"
    _export_yolo_labels(
        harness, layout_geometry, scale_bbox, yolo_path, render_config
    )
    output_files["yolo"] = yolo_path

    # 5. Export COCO-format annotations (single image)
    coco_path = output_dir / f"{sample_name}.coco.json"
    coco_data = _export_coco_labels(
        harness, layout_geometry, scale_bbox, sample_name, render_config
    )
    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    output_files["coco"] = coco_path

    return output_files


def _export_object_labels(
    harness: HarnessGraph,
    layout_geometry: LayoutGeometry,
    scale_bbox,
    sample_name: str,
    render_config: RenderConfig,
) -> dict[str, Any]:
    """Export object detection labels in custom JSON format."""
    objects = []

    for component in harness.components:
        bbox = layout_geometry.component_bboxes.get(component.id)
        if not bbox:
            continue

        scaled_bbox = scale_bbox(bbox)
        x1, y1, x2, y2 = scaled_bbox

        objects.append(
            {
                "id": component.id,
                "class": component.type.value,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "label": component.label,
                "attributes": component.attributes,
            }
        )

    return {
        "image": f"{sample_name}.png",
        "image_width": render_config.width,
        "image_height": render_config.height,
        "objects": objects,
    }


def _export_wire_labels(
    harness: HarnessGraph,
    render_geometry: RenderGeometry,
    layout_geometry: LayoutGeometry,
    scale_point,
) -> dict[str, Any]:
    """Export wire/line segment labels."""
    wires = []

    for wire in harness.wires:
        polyline = render_geometry.wire_polylines.get(wire.id)
        if not polyline:
            # Fallback: use endpoint positions
            from_pos = layout_geometry.pin_positions.get(
                (wire.from_endpoint.component_id, wire.from_endpoint.pin_id)
            ) or layout_geometry.component_centers.get(wire.from_endpoint.component_id)

            to_pos = layout_geometry.pin_positions.get(
                (wire.to_endpoint.component_id, wire.to_endpoint.pin_id)
            ) or layout_geometry.component_centers.get(wire.to_endpoint.component_id)

            if from_pos and to_pos:
                polyline = [scale_point(from_pos), scale_point(to_pos)]
            else:
                continue

        wires.append(
            {
                "id": wire.id,
                "polyline": [[round(x, 2), round(y, 2)] for x, y in polyline],
                "from": {
                    "component_id": wire.from_endpoint.component_id,
                    "pin_id": wire.from_endpoint.pin_id,
                },
                "to": {
                    "component_id": wire.to_endpoint.component_id,
                    "pin_id": wire.to_endpoint.pin_id,
                },
                "color": wire.color,
                "gauge_mm2": wire.gauge_mm2,
            }
        )

    return {"wires": wires}


def _export_graph_json(harness: HarnessGraph) -> dict[str, Any]:
    """Export full graph as JSON (ground truth)."""
    components = []
    for comp in harness.components:
        comp_data = {
            "id": comp.id,
            "type": comp.type.value,
            "label": comp.label,
            "pins": [
                {
                    "id": pin.id,
                    "role": pin.role.value if pin.role else None,
                    "net_id": pin.net_id,
                }
                for pin in comp.pins
            ],
            "attributes": comp.attributes,
        }
        if comp.position:
            comp_data["position"] = list(comp.position)
        if comp.size:
            comp_data["size"] = list(comp.size)
        components.append(comp_data)

    wires = []
    for wire in harness.wires:
        wire_data = {
            "id": wire.id,
            "from_endpoint": {
                "component_id": wire.from_endpoint.component_id,
                "pin_id": wire.from_endpoint.pin_id,
            },
            "to_endpoint": {
                "component_id": wire.to_endpoint.component_id,
                "pin_id": wire.to_endpoint.pin_id,
            },
            "color": wire.color,
            "gauge_mm2": wire.gauge_mm2,
            "attributes": wire.attributes,
        }
        if wire.polyline:
            wire_data["polyline"] = [list(p) for p in wire.polyline]
        wires.append(wire_data)

    nets = [
        {
            "id": net.id,
            "wire_ids": net.wire_ids,
            "name": net.name,
            "attributes": net.attributes,
        }
        for net in harness.nets
    ]

    return {
        "id": harness.id,
        "components": components,
        "wires": wires,
        "nets": nets,
        "attributes": harness.attributes,
    }


# Class ID mapping for YOLO format
CLASS_ID_MAP = {
    "connector": 0,
    "fuse": 1,
    "relay": 2,
    "ground": 3,
    "splice": 4,
    "ecu": 5,
    "node": 6,
}


def _export_yolo_labels(
    harness: HarnessGraph,
    layout_geometry: LayoutGeometry,
    scale_bbox,
    output_path: Path,
    render_config: RenderConfig,
) -> None:
    """
    Export labels in YOLO format (normalized coordinates).

    Format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].
    """
    lines = []

    img_w = render_config.width
    img_h = render_config.height

    for component in harness.components:
        bbox = layout_geometry.component_bboxes.get(component.id)
        if not bbox:
            continue

        scaled_bbox = scale_bbox(bbox)
        x1, y1, x2, y2 = scaled_bbox

        # Convert to YOLO format (center x, center y, width, height) normalized
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        class_id = CLASS_ID_MAP.get(component.type.value, 6)

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _export_coco_labels(
    harness: HarnessGraph,
    layout_geometry: LayoutGeometry,
    scale_bbox,
    sample_name: str,
    render_config: RenderConfig,
) -> dict[str, Any]:
    """
    Export labels in COCO format for a single image.

    This is a simplified single-image COCO format.
    For full dataset COCO format, annotations should be aggregated.
    """
    categories = [
        {"id": 0, "name": "connector", "supercategory": "component"},
        {"id": 1, "name": "fuse", "supercategory": "component"},
        {"id": 2, "name": "relay", "supercategory": "component"},
        {"id": 3, "name": "ground", "supercategory": "component"},
        {"id": 4, "name": "splice", "supercategory": "component"},
        {"id": 5, "name": "ecu", "supercategory": "component"},
        {"id": 6, "name": "node", "supercategory": "component"},
    ]

    image = {
        "id": 1,
        "file_name": f"{sample_name}.png",
        "width": render_config.width,
        "height": render_config.height,
    }

    annotations = []
    ann_id = 1

    for component in harness.components:
        bbox = layout_geometry.component_bboxes.get(component.id)
        if not bbox:
            continue

        scaled_bbox = scale_bbox(bbox)
        x1, y1, x2, y2 = scaled_bbox
        width = x2 - x1
        height = y2 - y1

        annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": CLASS_ID_MAP.get(component.type.value, 6),
                "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                "area": round(width * height, 2),
                "iscrowd": 0,
                "attributes": {
                    "component_id": component.id,
                    "label": component.label,
                },
            }
        )
        ann_id += 1

    return {
        "images": [image],
        "annotations": annotations,
        "categories": categories,
    }


def aggregate_coco_annotations(
    label_dir: Path | str,
    output_path: Path | str,
) -> None:
    """
    Aggregate individual COCO label files into a single dataset file.

    Args:
        label_dir: Directory containing individual .coco.json files
        output_path: Path to write the aggregated COCO JSON
    """
    label_dir = Path(label_dir)
    output_path = Path(output_path)

    all_images = []
    all_annotations = []

    # Standard categories
    categories = [
        {"id": 0, "name": "connector", "supercategory": "component"},
        {"id": 1, "name": "fuse", "supercategory": "component"},
        {"id": 2, "name": "relay", "supercategory": "component"},
        {"id": 3, "name": "ground", "supercategory": "component"},
        {"id": 4, "name": "splice", "supercategory": "component"},
        {"id": 5, "name": "ecu", "supercategory": "component"},
        {"id": 6, "name": "node", "supercategory": "component"},
    ]

    image_id = 1
    ann_id = 1

    for coco_file in sorted(label_dir.glob("*.coco.json")):
        with open(coco_file) as f:
            data = json.load(f)

        # Update image ID
        for img in data.get("images", []):
            old_img_id = img["id"]
            img["id"] = image_id
            all_images.append(img)

            # Update annotation image_ids and annotation ids
            for ann in data.get("annotations", []):
                if ann["image_id"] == old_img_id:
                    ann["id"] = ann_id
                    ann["image_id"] = image_id
                    all_annotations.append(ann)
                    ann_id += 1

            image_id += 1

    aggregated = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories,
    }

    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
