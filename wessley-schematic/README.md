# Wessley Schematic

**Automotive Wiring Diagram Synthetic Data & Parsing for Wessley AI**

Generate synthetic automotive wiring diagrams with ML-ready labels for training object detection and graph reconstruction models.

## Overview

This package provides tools for:

1. **Generating synthetic harness graphs** - Random but valid automotive wiring topologies
2. **Rendering 2D diagrams** - SVG (and optionally PNG) visualizations
3. **Exporting ML labels** - YOLO, COCO, and custom JSON formats for training

## Installation

```bash
# Clone and install in development mode
cd wessley-schematic
pip install -e .

# With dev dependencies (for testing)
pip install -e ".[dev]"

# With PNG support (requires Cairo)
pip install -e ".[png]"
```

## Quick Start

### Generate a Synthetic Dataset

```bash
# Generate 10 samples with default settings
wessley-schematic generate --output-dir ./dataset --num-samples 10

# Generate with custom parameters
wessley-schematic generate \
    --output-dir ./dataset \
    --num-samples 100 \
    --min-connectors 3 \
    --max-connectors 8 \
    --min-wires 10 \
    --max-wires 30 \
    --seed 42
```

### Inspect a Single Sample

```bash
# Print JSON to stdout
wessley-schematic inspect-one --seed 42

# Print human-readable summary
wessley-schematic inspect-one --seed 42 --format summary
```

### Output Structure

After generation, you'll have:

```
dataset/
├── svgs/
│   ├── synthetic_0001.svg
│   ├── synthetic_0002.svg
│   └── ...
├── images/          # Only if --png flag used
│   └── ...
├── labels/
│   ├── synthetic_0001.objects.json   # Object detection labels
│   ├── synthetic_0001.wires.json     # Wire/line segment labels
│   ├── synthetic_0001.graph.json     # Full graph ground truth
│   ├── synthetic_0001.txt            # YOLO format
│   ├── synthetic_0001.coco.json      # COCO format
│   └── ...
└── annotations.json  # Aggregated COCO dataset
```

## Python API

```python
from wessley_schematic import HarnessGraph, GraphGeneratorConfig
from wessley_schematic.synthetic import (
    generate_harness_graph,
    layout_harness,
    render_svg,
    export_labels,
)
from wessley_schematic.synthetic.config import LayoutConfig, RenderConfig

# Configure generator
config = GraphGeneratorConfig(
    min_connectors=3,
    max_connectors=6,
    min_wires=10,
    max_wires=25,
    allow_fuses=True,
    allow_relays=True,
    allow_ecus=True,
    seed=42,
)

# Generate a harness graph
harness = generate_harness_graph(config)

# Apply 2D layout
harness, layout_geometry = layout_harness(harness)

# Render to SVG
render_geometry = render_svg(harness, layout_geometry, "output.svg")

# Export ML labels
export_labels(
    harness,
    layout_geometry,
    render_geometry,
    output_dir="./labels",
    sample_name="sample_001",
)
```

## Schema Overview

### Component Types

| Type | Description | Has Pins |
|------|-------------|----------|
| `connector` | Multi-pin connector | Yes |
| `fuse` | Inline fuse | No |
| `relay` | Electromechanical relay | Yes |
| `ground` | Ground point | No |
| `splice` | Wire splice/junction | No |
| `ecu` | Electronic control unit | Yes |
| `node` | Generic junction | No |

### Label Formats

**YOLO format** (`.txt`):
```
<class_id> <x_center> <y_center> <width> <height>
```
Coordinates normalized to [0, 1].

**COCO format** (`.coco.json`):
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

**Objects JSON** (`.objects.json`):
```json
{
  "image": "synthetic_0001.png",
  "objects": [
    {"id": "C001", "class": "connector", "bbox": [x1, y1, x2, y2], ...}
  ]
}
```

**Wires JSON** (`.wires.json`):
```json
{
  "wires": [
    {
      "id": "W001",
      "polyline": [[x1, y1], [x2, y2], ...],
      "from": {"component_id": "C001", "pin_id": "1"},
      "to": {"component_id": "F01", "pin_id": null}
    }
  ]
}
```

**Graph JSON** (`.graph.json`):
Complete ground truth with all component, wire, and net information.

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=wessley_schematic
```

### Code Style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## Configuration Reference

### GraphGeneratorConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_connectors` | 2 | Minimum connectors per diagram |
| `max_connectors` | 6 | Maximum connectors per diagram |
| `min_wires` | 5 | Minimum wires per diagram |
| `max_wires` | 20 | Maximum wires per diagram |
| `allow_fuses` | True | Include fuses |
| `allow_relays` | True | Include relays |
| `allow_splices` | True | Include splices |
| `allow_ecus` | True | Include ECUs |
| `allow_grounds` | True | Include ground points |
| `ensure_connected` | True | Ensure graph connectivity |
| `seed` | None | Random seed |

### RenderConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | 1200 | Output image width (px) |
| `height` | 960 | Output image height (px) |
| `generate_png` | False | Also generate PNG files |
| `show_pin_labels` | True | Show pin numbers |
| `show_wire_labels` | True | Show wire colors/gauges |

## Future Phases

This is Phase 1 of the Wessley Schematic Ingestion pipeline. Future phases will include:

- **Phase 2**: Deep learning models for symbol detection
- **Phase 3**: OCR integration for text labels
- **Phase 4**: Real OEM PDF parsing and domain adaptation
- **Phase 5**: 3D harness reconstruction and Wessley integration

## License

MIT License - See LICENSE file for details.

## Contact

Wessley AI Team - sahar@wessley.ai
