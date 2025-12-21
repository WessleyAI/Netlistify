"""
Layout engine for positioning harness components in 2D space.

Uses a simple but deterministic layout strategy:
- Connectors along left and right edges
- ECUs on the right side
- Fuses and relays in the middle/top area
- Grounds at the bottom
- Splices distributed in the center
"""

import math
from dataclasses import dataclass, field

from wessley_schematic.schemas.graph import (
    Component,
    ComponentType,
    HarnessGraph,
    Pin,
)
from wessley_schematic.synthetic.config import LayoutConfig


@dataclass
class LayoutGeometry:
    """
    Geometry information produced by the layout engine.

    Contains all position and bounding box data needed for rendering and labeling.
    """

    # Component bounding boxes: component_id -> (x1, y1, x2, y2)
    component_bboxes: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )

    # Pin positions: (component_id, pin_id) -> (x, y)
    pin_positions: dict[tuple[str, str], tuple[float, float]] = field(
        default_factory=dict
    )

    # Component center positions: component_id -> (x, y)
    component_centers: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Label positions: label_id -> (x, y)
    label_positions: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Canvas dimensions
    width: float = 1000.0
    height: float = 800.0


def layout_harness(
    harness: HarnessGraph, config: LayoutConfig | None = None
) -> tuple[HarnessGraph, LayoutGeometry]:
    """
    Assign 2D positions to all components and pins in the harness.

    Args:
        harness: The harness graph to layout
        config: Layout configuration (uses defaults if None)

    Returns:
        Tuple of (harness with position fields populated, LayoutGeometry)
    """
    if config is None:
        config = LayoutConfig()

    geometry = LayoutGeometry(width=config.width, height=config.height)

    # Group components by type
    connectors = harness.get_components_by_type(ComponentType.CONNECTOR)
    ecus = harness.get_components_by_type(ComponentType.ECU)
    fuses = harness.get_components_by_type(ComponentType.FUSE)
    relays = harness.get_components_by_type(ComponentType.RELAY)
    grounds = harness.get_components_by_type(ComponentType.GROUND)
    splices = harness.get_components_by_type(ComponentType.SPLICE)
    nodes = harness.get_components_by_type(ComponentType.NODE)

    # Calculate usable area
    left = config.margin_left
    right = config.width - config.margin_right
    top = config.margin_top
    bottom = config.height - config.margin_bottom

    usable_width = right - left
    usable_height = bottom - top

    # Layout connectors on left edge
    _layout_column(
        connectors,
        x=left + config.connector_width / 2,
        y_start=top + 50,
        y_end=bottom - 50,
        config=config,
        geometry=geometry,
        component_width=config.connector_width,
        height_per_pin=config.connector_height_per_pin,
    )

    # Layout ECUs on right edge
    _layout_column(
        ecus,
        x=right - config.ecu_width / 2,
        y_start=top + 50,
        y_end=bottom - 100,
        config=config,
        geometry=geometry,
        component_width=config.ecu_width,
        height_per_pin=config.ecu_height_per_pin,
    )

    # Layout fuses in upper-middle area
    _layout_row(
        fuses,
        y=top + 80,
        x_start=left + usable_width * 0.25,
        x_end=left + usable_width * 0.75,
        config=config,
        geometry=geometry,
        component_size=config.fuse_size,
    )

    # Layout relays in middle area
    _layout_row(
        relays,
        y=top + usable_height * 0.4,
        x_start=left + usable_width * 0.3,
        x_end=left + usable_width * 0.7,
        config=config,
        geometry=geometry,
        component_size=config.relay_size,
        has_pins=True,
    )

    # Layout grounds at bottom
    _layout_row(
        grounds,
        y=bottom - 40,
        x_start=left + usable_width * 0.2,
        x_end=left + usable_width * 0.8,
        config=config,
        geometry=geometry,
        component_size=config.ground_size,
    )

    # Layout splices scattered in center
    _layout_scattered(
        splices,
        center_x=left + usable_width / 2,
        center_y=top + usable_height / 2,
        radius=min(usable_width, usable_height) * 0.25,
        config=config,
        geometry=geometry,
        component_size=config.splice_size,
    )

    # Layout any generic nodes
    _layout_scattered(
        nodes,
        center_x=left + usable_width / 2,
        center_y=top + usable_height * 0.6,
        radius=min(usable_width, usable_height) * 0.15,
        config=config,
        geometry=geometry,
        component_size=config.splice_size,
    )

    # Update harness components with position data
    for component in harness.components:
        if component.id in geometry.component_centers:
            component.position = geometry.component_centers[component.id]

        bbox = geometry.component_bboxes.get(component.id)
        if bbox:
            x1, y1, x2, y2 = bbox
            component.size = (x2 - x1, y2 - y1)

        # Update pin positions
        for pin in component.pins:
            key = (component.id, pin.id)
            if key in geometry.pin_positions:
                pin.position = geometry.pin_positions[key]

    return harness, geometry


def _layout_column(
    components: list[Component],
    x: float,
    y_start: float,
    y_end: float,
    config: LayoutConfig,
    geometry: LayoutGeometry,
    component_width: float,
    height_per_pin: float,
) -> None:
    """Layout components in a vertical column with pins."""
    if not components:
        return

    # Calculate total height needed
    total_height = 0
    heights = []
    for comp in components:
        num_pins = max(len(comp.pins), 2)
        height = num_pins * height_per_pin + 20  # 20 for padding
        heights.append(height)
        total_height += height

    # Add spacing
    spacing = config.component_spacing * 0.5
    total_height += spacing * (len(components) - 1)

    # Calculate starting y to center the column
    available = y_end - y_start
    start_y = y_start + max(0, (available - total_height) / 2)

    current_y = start_y
    for comp, height in zip(components, heights):
        cx = x
        cy = current_y + height / 2

        # Store center and bbox
        geometry.component_centers[comp.id] = (cx, cy)
        geometry.component_bboxes[comp.id] = (
            cx - component_width / 2,
            current_y,
            cx + component_width / 2,
            current_y + height,
        )

        # Layout pins along the right edge (for left column) or left edge
        # We'll put pins on the side facing the center
        pin_x = cx + component_width / 2  # Right edge
        if x > geometry.width / 2:
            pin_x = cx - component_width / 2  # Left edge for right-side components

        num_pins = len(comp.pins)
        if num_pins > 0:
            pin_spacing = (height - 10) / (num_pins + 1)
            for i, pin in enumerate(comp.pins):
                pin_y = current_y + 5 + (i + 1) * pin_spacing
                geometry.pin_positions[(comp.id, pin.id)] = (pin_x, pin_y)

        # Store label position (above the component)
        geometry.label_positions[f"{comp.id}_label"] = (cx, current_y - 10)

        current_y += height + spacing


def _layout_row(
    components: list[Component],
    y: float,
    x_start: float,
    x_end: float,
    config: LayoutConfig,
    geometry: LayoutGeometry,
    component_size: float,
    has_pins: bool = False,
) -> None:
    """Layout components in a horizontal row."""
    if not components:
        return

    available_width = x_end - x_start
    spacing = min(
        config.component_spacing,
        available_width / (len(components) + 1),
    )

    # Center components in the row
    total_width = len(components) * component_size + (len(components) - 1) * spacing
    start_x = x_start + (available_width - total_width) / 2

    for i, comp in enumerate(components):
        cx = start_x + component_size / 2 + i * (component_size + spacing)
        cy = y

        half_size = component_size / 2
        geometry.component_centers[comp.id] = (cx, cy)
        geometry.component_bboxes[comp.id] = (
            cx - half_size,
            cy - half_size,
            cx + half_size,
            cy + half_size,
        )

        # Layout pins if component has them (e.g., relays)
        if has_pins and comp.pins:
            num_pins = len(comp.pins)
            # Distribute pins around the component
            for j, pin in enumerate(comp.pins):
                angle = (2 * math.pi * j) / num_pins - math.pi / 2
                pin_x = cx + (half_size + 5) * math.cos(angle)
                pin_y = cy + (half_size + 5) * math.sin(angle)
                geometry.pin_positions[(comp.id, pin.id)] = (pin_x, pin_y)

        # Label position
        geometry.label_positions[f"{comp.id}_label"] = (cx, cy - half_size - 15)


def _layout_scattered(
    components: list[Component],
    center_x: float,
    center_y: float,
    radius: float,
    config: LayoutConfig,
    geometry: LayoutGeometry,
    component_size: float,
) -> None:
    """Layout components scattered around a center point."""
    if not components:
        return

    for i, comp in enumerate(components):
        if len(components) == 1:
            cx, cy = center_x, center_y
        else:
            # Distribute in a circle
            angle = (2 * math.pi * i) / len(components)
            cx = center_x + radius * math.cos(angle)
            cy = center_y + radius * math.sin(angle)

        half_size = component_size / 2
        geometry.component_centers[comp.id] = (cx, cy)
        geometry.component_bboxes[comp.id] = (
            cx - half_size,
            cy - half_size,
            cx + half_size,
            cy + half_size,
        )

        # Label position
        geometry.label_positions[f"{comp.id}_label"] = (cx, cy - half_size - 10)
