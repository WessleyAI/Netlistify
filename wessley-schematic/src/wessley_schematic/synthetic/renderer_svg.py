"""
SVG renderer for harness diagrams.

Produces SVG files with automotive-flavored visual conventions,
tracking exact geometry for ML label export.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from wessley_schematic.schemas.graph import ComponentType, HarnessGraph, Wire
from wessley_schematic.synthetic.config import RenderConfig
from wessley_schematic.synthetic.layout import LayoutGeometry


@dataclass
class RenderGeometry:
    """
    Geometry information produced during rendering.

    Extends LayoutGeometry with wire polyline data.
    """

    # Wire polylines: wire_id -> list of (x, y) points
    wire_polylines: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    # Wire label positions: wire_id -> (x, y)
    wire_label_positions: dict[str, tuple[float, float]] = field(default_factory=dict)


def render_svg(
    harness: HarnessGraph,
    layout_geometry: LayoutGeometry,
    output_path: Path | str,
    config: RenderConfig | None = None,
) -> RenderGeometry:
    """
    Render a harness diagram to SVG.

    Args:
        harness: The harness graph to render
        layout_geometry: Layout geometry from layout_harness()
        output_path: Path to write the SVG file
        config: Render configuration (uses defaults if None)

    Returns:
        RenderGeometry with wire polylines and positions
    """
    if config is None:
        config = RenderConfig()

    output_path = Path(output_path)
    render_geometry = RenderGeometry()

    # Calculate scaling from layout coordinates to render coordinates
    scale_x = config.width / layout_geometry.width
    scale_y = config.height / layout_geometry.height

    def scale_point(x: float, y: float) -> tuple[float, float]:
        return (x * scale_x, y * scale_y)

    def scale_bbox(
        bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        sx1, sy1 = scale_point(x1, y1)
        sx2, sy2 = scale_point(x2, y2)
        return (sx1, sy1, sx2, sy2)

    # Create SVG root element
    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(config.width),
            "height": str(config.height),
            "viewBox": f"0 0 {config.width} {config.height}",
        },
    )

    # Add background
    ET.SubElement(
        svg,
        "rect",
        {
            "width": "100%",
            "height": "100%",
            "fill": config.background_color,
        },
    )

    # Create groups for layering
    wires_group = ET.SubElement(svg, "g", {"id": "wires"})
    components_group = ET.SubElement(svg, "g", {"id": "components"})
    labels_group = ET.SubElement(svg, "g", {"id": "labels"})

    # Render wires first (behind components)
    for wire in harness.wires:
        polyline = _render_wire(
            wire,
            harness,
            layout_geometry,
            scale_point,
            wires_group,
            labels_group,
            config,
        )
        if polyline:
            render_geometry.wire_polylines[wire.id] = polyline

            # Calculate wire label position (midpoint)
            if len(polyline) >= 2:
                mid_idx = len(polyline) // 2
                mx, my = polyline[mid_idx]
                render_geometry.wire_label_positions[wire.id] = (mx, my)

    # Render components
    for component in harness.components:
        bbox = layout_geometry.component_bboxes.get(component.id)
        if not bbox:
            continue

        scaled_bbox = scale_bbox(bbox)

        if component.type == ComponentType.CONNECTOR:
            _render_connector(
                component,
                scaled_bbox,
                layout_geometry,
                scale_point,
                components_group,
                labels_group,
                config,
            )
        elif component.type == ComponentType.ECU:
            _render_ecu(
                component,
                scaled_bbox,
                layout_geometry,
                scale_point,
                components_group,
                labels_group,
                config,
            )
        elif component.type == ComponentType.FUSE:
            _render_fuse(
                component,
                scaled_bbox,
                components_group,
                labels_group,
                config,
            )
        elif component.type == ComponentType.RELAY:
            _render_relay(
                component,
                scaled_bbox,
                layout_geometry,
                scale_point,
                components_group,
                labels_group,
                config,
            )
        elif component.type == ComponentType.GROUND:
            _render_ground(
                component,
                scaled_bbox,
                components_group,
                labels_group,
                config,
            )
        elif component.type == ComponentType.SPLICE:
            _render_splice(
                component,
                scaled_bbox,
                components_group,
                labels_group,
                config,
            )
        else:
            # Generic node
            _render_node(
                component,
                scaled_bbox,
                components_group,
                labels_group,
                config,
            )

    # Write SVG file
    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    # Optionally generate PNG
    if config.generate_png:
        _render_png(output_path, config)

    return render_geometry


def _render_wire(
    wire: Wire,
    harness: HarnessGraph,
    layout_geometry: LayoutGeometry,
    scale_point,
    wires_group: ET.Element,
    labels_group: ET.Element,
    config: RenderConfig,
) -> list[tuple[float, float]] | None:
    """Render a wire as an orthogonal polyline."""
    # Get endpoint positions
    from_pos = _get_endpoint_position(
        wire.from_endpoint.component_id,
        wire.from_endpoint.pin_id,
        layout_geometry,
    )
    to_pos = _get_endpoint_position(
        wire.to_endpoint.component_id,
        wire.to_endpoint.pin_id,
        layout_geometry,
    )

    if not from_pos or not to_pos:
        return None

    # Scale positions
    from_pos = scale_point(*from_pos)
    to_pos = scale_point(*to_pos)

    # Create orthogonal routing (simple L-shape or Z-shape)
    polyline = _route_orthogonal(from_pos, to_pos)

    # Create path data
    path_data = f"M {polyline[0][0]:.1f} {polyline[0][1]:.1f}"
    for x, y in polyline[1:]:
        path_data += f" L {x:.1f} {y:.1f}"

    ET.SubElement(
        wires_group,
        "path",
        {
            "d": path_data,
            "stroke": config.wire_color,
            "stroke-width": str(config.wire_stroke_width),
            "fill": "none",
            "id": f"wire_{wire.id}",
        },
    )

    # Add wire label
    if config.show_wire_labels and wire.color:
        mid_idx = len(polyline) // 2
        mx, my = polyline[mid_idx]
        label_text = wire.color
        if wire.gauge_mm2:
            label_text += f" {wire.gauge_mm2}mm²"

        ET.SubElement(
            labels_group,
            "text",
            {
                "x": str(mx + 5),
                "y": str(my - 5),
                "font-family": config.font_family,
                "font-size": str(config.font_size_wire),
                "fill": config.text_color,
            },
        ).text = label_text

    return polyline


def _get_endpoint_position(
    component_id: str,
    pin_id: str | None,
    layout_geometry: LayoutGeometry,
) -> tuple[float, float] | None:
    """Get the position of a wire endpoint."""
    if pin_id:
        return layout_geometry.pin_positions.get((component_id, pin_id))
    else:
        return layout_geometry.component_centers.get(component_id)


def _route_orthogonal(
    from_pos: tuple[float, float],
    to_pos: tuple[float, float],
) -> list[tuple[float, float]]:
    """Create an orthogonal wire route between two points."""
    x1, y1 = from_pos
    x2, y2 = to_pos

    # Simple L-shape or Z-shape routing
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > dy:
        # Horizontal first, then vertical
        mid_x = (x1 + x2) / 2
        return [
            (x1, y1),
            (mid_x, y1),
            (mid_x, y2),
            (x2, y2),
        ]
    else:
        # Vertical first, then horizontal
        mid_y = (y1 + y2) / 2
        return [
            (x1, y1),
            (x1, mid_y),
            (x2, mid_y),
            (x2, y2),
        ]


def _render_connector(
    component,
    bbox: tuple[float, float, float, float],
    layout_geometry: LayoutGeometry,
    scale_point,
    group: ET.Element,
    labels_group: ET.Element,
    config: RenderConfig,
) -> None:
    """Render a connector as a rectangle with pin markers."""
    x1, y1, x2, y2 = bbox

    # Main rectangle
    ET.SubElement(
        group,
        "rect",
        {
            "x": str(x1),
            "y": str(y1),
            "width": str(x2 - x1),
            "height": str(y2 - y1),
            "fill": config.component_fill,
            "stroke": config.component_stroke,
            "stroke-width": str(config.component_stroke_width),
            "rx": "3",
            "id": f"component_{component.id}",
        },
    )

    # Pin markers
    for pin in component.pins:
        pin_pos = layout_geometry.pin_positions.get((component.id, pin.id))
        if pin_pos:
            px, py = scale_point(*pin_pos)
            ET.SubElement(
                group,
                "circle",
                {
                    "cx": str(px),
                    "cy": str(py),
                    "r": "4",
                    "fill": "#4CAF50" if pin.role and pin.role.value == "power" else "#2196F3",
                    "stroke": config.component_stroke,
                    "stroke-width": "1",
                },
            )

            # Pin label
            if config.show_pin_labels:
                ET.SubElement(
                    labels_group,
                    "text",
                    {
                        "x": str(px - 15),
                        "y": str(py + 3),
                        "font-family": config.font_family,
                        "font-size": str(config.font_size_pin),
                        "fill": config.text_color,
                        "text-anchor": "end",
                    },
                ).text = pin.id

    # Component label
    if config.show_component_labels:
        cx = (x1 + x2) / 2
        ET.SubElement(
            labels_group,
            "text",
            {
                "x": str(cx),
                "y": str(y1 - 5),
                "font-family": config.font_family,
                "font-size": str(config.font_size_component),
                "fill": config.text_color,
                "text-anchor": "middle",
                "font-weight": "bold",
            },
        ).text = component.id


def _render_ecu(
    component,
    bbox: tuple[float, float, float, float],
    layout_geometry: LayoutGeometry,
    scale_point,
    group: ET.Element,
    labels_group: ET.Element,
    config: RenderConfig,
) -> None:
    """Render an ECU as a rounded rectangle with pin markers."""
    x1, y1, x2, y2 = bbox

    # Main rectangle with rounded corners
    ET.SubElement(
        group,
        "rect",
        {
            "x": str(x1),
            "y": str(y1),
            "width": str(x2 - x1),
            "height": str(y2 - y1),
            "fill": "#E3F2FD",
            "stroke": "#1565C0",
            "stroke-width": str(config.component_stroke_width),
            "rx": "8",
            "id": f"component_{component.id}",
        },
    )

    # Pin markers on left side
    for pin in component.pins:
        pin_pos = layout_geometry.pin_positions.get((component.id, pin.id))
        if pin_pos:
            px, py = scale_point(*pin_pos)
            ET.SubElement(
                group,
                "rect",
                {
                    "x": str(px - 3),
                    "y": str(py - 3),
                    "width": "6",
                    "height": "6",
                    "fill": "#1565C0",
                },
            )

            if config.show_pin_labels:
                ET.SubElement(
                    labels_group,
                    "text",
                    {
                        "x": str(px + 10),
                        "y": str(py + 3),
                        "font-family": config.font_family,
                        "font-size": str(config.font_size_pin),
                        "fill": config.text_color,
                    },
                ).text = pin.id

    # Component label
    if config.show_component_labels:
        cx = (x1 + x2) / 2
        cy = y1 + 15
        ET.SubElement(
            labels_group,
            "text",
            {
                "x": str(cx),
                "y": str(cy),
                "font-family": config.font_family,
                "font-size": str(config.font_size_component),
                "fill": "#1565C0",
                "text-anchor": "middle",
                "font-weight": "bold",
            },
        ).text = component.label or component.id


def _render_fuse(
    component,
    bbox: tuple[float, float, float, float],
    group: ET.Element,
    labels_group: ET.Element,
    config: RenderConfig,
) -> None:
    """Render a fuse symbol."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * 0.8
    h = (y2 - y1) * 0.4

    # Fuse body (rectangle with wavy line inside)
    ET.SubElement(
        group,
        "rect",
        {
            "x": str(cx - w / 2),
            "y": str(cy - h / 2),
            "width": str(w),
            "height": str(h),
            "fill": "#FFF9C4",
            "stroke": "#F57F17",
            "stroke-width": str(config.component_stroke_width),
            "rx": "2",
            "id": f"component_{component.id}",
        },
    )

    # Fuse element (line through middle)
    ET.SubElement(
        group,
        "line",
        {
            "x1": str(cx - w / 3),
            "y1": str(cy),
            "x2": str(cx + w / 3),
            "y2": str(cy),
            "stroke": "#F57F17",
            "stroke-width": "2",
        },
    )

    # Connection points
    for offset in [-w / 2 - 5, w / 2 + 5]:
        ET.SubElement(
            group,
            "circle",
            {
                "cx": str(cx + offset),
                "cy": str(cy),
                "r": "3",
                "fill": "#F57F17",
            },
        )

    # Label
    if config.show_component_labels:
        ET.SubElement(
            labels_group,
            "text",
            {
                "x": str(cx),
                "y": str(y1 - 5),
                "font-family": config.font_family,
                "font-size": str(config.font_size_component),
                "fill": config.text_color,
                "text-anchor": "middle",
            },
        ).text = component.label or component.id


def _render_relay(
    component,
    bbox: tuple[float, float, float, float],
    layout_geometry: LayoutGeometry,
    scale_point,
    group: ET.Element,
    labels_group: ET.Element,
    config: RenderConfig,
) -> None:
    """Render a relay symbol."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = min(x2 - x1, y2 - y1) * 0.9

    # Relay body (square with coil symbol)
    ET.SubElement(
        group,
        "rect",
        {
            "x": str(cx - size / 2),
            "y": str(cy - size / 2),
            "width": str(size),
            "height": str(size),
            "fill": "#F3E5F5",
            "stroke": "#7B1FA2",
            "stroke-width": str(config.component_stroke_width),
            "rx": "4",
            "id": f"component_{component.id}",
        },
    )

    # Coil symbol (small rectangle inside)
    coil_w = size * 0.4
    coil_h = size * 0.25
    ET.SubElement(
        group,
        "rect",
        {
            "x": str(cx - coil_w / 2),
            "y": str(cy - coil_h / 2),
            "width": str(coil_w),
            "height": str(coil_h),
            "fill": "none",
            "stroke": "#7B1FA2",
            "stroke-width": "1.5",
        },
    )

    # Pin markers
    for pin in component.pins:
        pin_pos = layout_geometry.pin_positions.get((component.id, pin.id))
        if pin_pos:
            px, py = scale_point(*pin_pos)
            ET.SubElement(
                group,
                "circle",
                {
                    "cx": str(px),
                    "cy": str(py),
                    "r": "4",
                    "fill": "#7B1FA2",
                },
            )

    # Label
    if config.show_component_labels:
        ET.SubElement(
            labels_group,
            "text",
            {
                "x": str(cx),
                "y": str(y1 - 5),
                "font-family": config.font_family,
                "font-size": str(config.font_size_component),
                "fill": config.text_color,
                "text-anchor": "middle",
            },
        ).text = component.id


def _render_ground(
    component,
    bbox: tuple[float, float, float, float],
    group: ET.Element,
    labels_group: ET.Element,
    config: RenderConfig,
) -> None:
    """Render a ground symbol (classic 3-line wedge)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = min(x2 - x1, y2 - y1)

    # Create ground symbol group
    ground_group = ET.SubElement(group, "g", {"id": f"component_{component.id}"})

    # Connection point at top
    ET.SubElement(
        ground_group,
        "circle",
        {
            "cx": str(cx),
            "cy": str(cy - size * 0.3),
            "r": "4",
            "fill": "#333333",
        },
    )

    # Vertical line
    ET.SubElement(
        ground_group,
        "line",
        {
            "x1": str(cx),
            "y1": str(cy - size * 0.3),
            "x2": str(cx),
            "y2": str(cy),
            "stroke": config.component_stroke,
            "stroke-width": str(config.component_stroke_width),
        },
    )

    # Three horizontal lines (wedge pattern)
    for i, (w_factor, y_offset) in enumerate([(0.8, 0), (0.5, 0.15), (0.2, 0.3)]):
        line_w = size * w_factor
        line_y = cy + y_offset * size
        ET.SubElement(
            ground_group,
            "line",
            {
                "x1": str(cx - line_w / 2),
                "y1": str(line_y),
                "x2": str(cx + line_w / 2),
                "y2": str(line_y),
                "stroke": config.component_stroke,
                "stroke-width": str(config.component_stroke_width),
            },
        )

    # Label
    if config.show_component_labels:
        ET.SubElement(
            labels_group,
            "text",
            {
                "x": str(cx),
                "y": str(y1 - 5),
                "font-family": config.font_family,
                "font-size": str(config.font_size_component),
                "fill": config.text_color,
                "text-anchor": "middle",
            },
        ).text = component.id


def _render_splice(
    component,
    bbox: tuple[float, float, float, float],
    group: ET.Element,
    labels_group: ET.Element,
    config: RenderConfig,
) -> None:
    """Render a splice point (solid dot)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    r = min(x2 - x1, y2 - y1) / 2

    ET.SubElement(
        group,
        "circle",
        {
            "cx": str(cx),
            "cy": str(cy),
            "r": str(r),
            "fill": "#333333",
            "id": f"component_{component.id}",
        },
    )

    # Label
    if config.show_component_labels:
        ET.SubElement(
            labels_group,
            "text",
            {
                "x": str(cx),
                "y": str(y1 - 5),
                "font-family": config.font_family,
                "font-size": str(config.font_size_component),
                "fill": config.text_color,
                "text-anchor": "middle",
            },
        ).text = component.id


def _render_node(
    component,
    bbox: tuple[float, float, float, float],
    group: ET.Element,
    labels_group: ET.Element,
    config: RenderConfig,
) -> None:
    """Render a generic node (small circle)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    r = min(x2 - x1, y2 - y1) / 2

    ET.SubElement(
        group,
        "circle",
        {
            "cx": str(cx),
            "cy": str(cy),
            "r": str(r),
            "fill": config.component_fill,
            "stroke": config.component_stroke,
            "stroke-width": str(config.component_stroke_width),
            "id": f"component_{component.id}",
        },
    )


def _render_png(svg_path: Path, config: RenderConfig) -> None:
    """Optionally convert SVG to PNG using cairosvg."""
    try:
        import cairosvg

        png_path = svg_path.with_suffix(".png")
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            scale=config.png_scale,
        )
    except ImportError:
        pass  # cairosvg not installed, skip PNG generation
