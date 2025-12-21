# Architecture

This document describes the architecture of the Wessley Schematic synthetic data generation system.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (main.py)                           │
│  Commands: generate, inspect-one, version                       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Generation Pipeline                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Graph     │───▶│    Layout    │───▶│   Renderer   │      │
│  │  Generator   │    │    Engine    │    │    (SVG)     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ HarnessGraph │    │LayoutGeometry│    │RenderGeometry│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              │                   │               │
│                              └─────────┬─────────┘               │
│                                        ▼                         │
│                               ┌──────────────┐                   │
│                               │    Labels    │                   │
│                               │   Exporter   │                   │
│                               └──────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
wessley-schematic/
├── src/wessley_schematic/
│   ├── __init__.py          # Package exports
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── graph.py         # Core data models
│   ├── synthetic/
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration classes
│   │   ├── graph_generator.py
│   │   ├── layout.py
│   │   ├── renderer_svg.py
│   │   └── labels.py
│   └── cli/
│       ├── __init__.py
│       └── main.py          # CLI entry point
└── tests/
    ├── test_graph_generator.py
    ├── test_layout.py
    └── test_renderer_and_labels.py
```

## Core Data Models

### HarnessGraph

The central data structure representing an automotive wiring harness.

```python
HarnessGraph
├── id: str
├── components: list[Component]
├── wires: list[Wire]
├── nets: list[Net]  # Optional
└── attributes: dict
```

### Component

Represents a physical component in the harness.

```python
Component
├── id: str                    # e.g., "C101", "F10", "ECU_A"
├── type: ComponentType        # connector, fuse, relay, ground, splice, ecu, node
├── pins: list[Pin]            # Connection points
├── label: str | None
├── attributes: dict
├── position: tuple[float, float] | None  # Set by layout
└── size: tuple[float, float] | None      # Set by layout
```

### Wire

Represents a physical wire connecting two endpoints.

```python
Wire
├── id: str
├── from_endpoint: Endpoint
├── to_endpoint: Endpoint
├── color: str | None          # e.g., "RD", "BL", "RD/BL"
├── gauge_mm2: float | None
├── attributes: dict
└── polyline: list[tuple] | None  # Set by renderer
```

### Endpoint

Specifies where a wire connects.

```python
Endpoint
├── component_id: str
└── pin_id: str | None  # None for pinless components like grounds
```

## Pipeline Stages

### 1. Graph Generation (`graph_generator.py`)

Creates random but valid harness graphs based on configuration.

**Input:** `GraphGeneratorConfig`

**Output:** `HarnessGraph`

**Key constraints:**
- All wires connect valid components/pins
- Graph is connected (when `ensure_connected=True`)
- Degree limits prevent unrealistic fan-in/fan-out
- Component counts respect min/max bounds

**Algorithm:**
1. Generate required connectors
2. Generate optional components (fuses, relays, ECUs, etc.)
3. Create spanning tree for connectivity
4. Add additional random wires up to target count

### 2. Layout Engine (`layout.py`)

Assigns 2D positions to all components and pins.

**Input:** `HarnessGraph`, `LayoutConfig`

**Output:** `(HarnessGraph, LayoutGeometry)`

**Layout strategy:**
- Connectors: Left edge column
- ECUs: Right edge column
- Fuses: Upper-middle row
- Relays: Center area with pins
- Grounds: Bottom row
- Splices: Scattered in center

**LayoutGeometry contains:**
- `component_bboxes`: Bounding boxes for each component
- `component_centers`: Center points
- `pin_positions`: Exact pin coordinates
- `label_positions`: Label anchor points

### 3. SVG Renderer (`renderer_svg.py`)

Produces SVG visualizations with tracked geometry.

**Input:** `HarnessGraph`, `LayoutGeometry`, `RenderConfig`

**Output:** `RenderGeometry`, SVG file

**Visual elements:**
- **Connectors**: Rectangles with pin circles
- **ECUs**: Rounded rectangles with labeled pins
- **Fuses**: Yellow boxes with connection points
- **Relays**: Purple boxes with coil symbol
- **Grounds**: Classic 3-line wedge symbol
- **Splices**: Solid dots
- **Wires**: Orthogonal polylines

**RenderGeometry contains:**
- `wire_polylines`: Routed wire paths
- `wire_label_positions`: Wire label anchors

### 4. Label Exporter (`labels.py`)

Exports annotations in multiple ML-ready formats.

**Input:** `HarnessGraph`, `LayoutGeometry`, `RenderGeometry`

**Output formats:**

| Format | File | Use Case |
|--------|------|----------|
| Objects JSON | `.objects.json` | Custom detection training |
| Wires JSON | `.wires.json` | Wire/line prediction |
| Graph JSON | `.graph.json` | Full ground truth |
| YOLO | `.txt` | YOLO detection training |
| COCO | `.coco.json` | COCO-format training |

## Configuration Classes

### GraphGeneratorConfig

Controls synthetic graph generation:
- Component counts (min/max connectors, wires)
- Optional components (fuses, relays, ECUs, etc.)
- Pin configurations
- Wire attributes (colors, gauges)
- Connectivity constraints

### LayoutConfig

Controls 2D positioning:
- Canvas dimensions
- Margins
- Component sizing
- Spacing

### RenderConfig

Controls SVG output:
- Image dimensions
- Colors and styles
- Label visibility
- PNG generation

## Extension Points

### Adding New Component Types

1. Add to `ComponentType` enum in `schemas/graph.py`
2. Create factory function in `graph_generator.py`
3. Add layout logic in `layout.py`
4. Add render function in `renderer_svg.py`
5. Update `CLASS_ID_MAP` in `labels.py`

### Custom Layout Strategies

Implement a custom layout function with signature:
```python
def custom_layout(harness: HarnessGraph, config: LayoutConfig) -> LayoutGeometry
```

### Custom Rendering Styles

Extend `RenderConfig` and add new render functions in `renderer_svg.py`.

## Data Flow Example

```python
# 1. Configure
config = GraphGeneratorConfig(seed=42)

# 2. Generate graph
harness = generate_harness_graph(config)
# -> HarnessGraph with components and wires

# 3. Apply layout
harness, layout_geo = layout_harness(harness)
# -> HarnessGraph updated with positions
# -> LayoutGeometry with bboxes, centers, pin positions

# 4. Render
render_geo = render_svg(harness, layout_geo, "output.svg")
# -> SVG file written
# -> RenderGeometry with wire polylines

# 5. Export labels
export_labels(harness, layout_geo, render_geo, "./labels", "sample_001")
# -> Multiple label files written
```

## Design Principles

1. **Separation of Concerns**: Each stage is independent and composable
2. **Immutable Data**: Stages produce new data rather than mutating
3. **Configuration over Code**: Behavior controlled through config objects
4. **Type Safety**: Full type hints for IDE support and validation
5. **Testability**: Each module can be tested in isolation

## Future Architecture Considerations

### Phase 2: Detection Models
- Add `models/` directory for PyTorch/TF models
- Detection heads for component classification
- Wire segment prediction networks

### Phase 3: OCR Integration
- Text detection and recognition pipeline
- Label extraction and matching

### Phase 4: Real Diagram Parsing
- PDF/image preprocessing
- Domain adaptation from synthetic to real
- Noise and artifact handling

### Phase 5: 3D Integration
- Graph to 3D harness conversion
- Wessley connector data format export
