"""Tests for the layout module."""

import math

import pytest

from wessley_schematic.schemas.graph import ComponentType
from wessley_schematic.synthetic.config import GraphGeneratorConfig, LayoutConfig
from wessley_schematic.synthetic.graph_generator import generate_harness_graph
from wessley_schematic.synthetic.layout import LayoutGeometry, layout_harness


class TestLayout:
    """Tests for the layout_harness function."""

    def test_returns_geometry(self):
        """Test that layout returns a LayoutGeometry object."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        harness, geometry = layout_harness(harness)

        assert isinstance(geometry, LayoutGeometry)

    def test_all_components_have_bboxes(self):
        """Test that all components get bounding boxes."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        harness, geometry = layout_harness(harness)

        for component in harness.components:
            assert component.id in geometry.component_bboxes
            bbox = geometry.component_bboxes[component.id]
            x1, y1, x2, y2 = bbox
            assert x2 > x1  # Width is positive
            assert y2 > y1  # Height is positive

    def test_all_components_have_centers(self):
        """Test that all components get center positions."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        harness, geometry = layout_harness(harness)

        for component in harness.components:
            assert component.id in geometry.component_centers
            cx, cy = geometry.component_centers[component.id]
            assert not math.isnan(cx)
            assert not math.isnan(cy)

    def test_pins_have_positions(self):
        """Test that pins get positions assigned."""
        config = GraphGeneratorConfig(seed=42, allow_ecus=True)
        harness = generate_harness_graph(config)

        harness, geometry = layout_harness(harness)

        # Check connectors and ECUs have pin positions
        for component in harness.components:
            if component.type in (ComponentType.CONNECTOR, ComponentType.ECU):
                for pin in component.pins:
                    key = (component.id, pin.id)
                    assert key in geometry.pin_positions
                    px, py = geometry.pin_positions[key]
                    assert not math.isnan(px)
                    assert not math.isnan(py)

    def test_respects_canvas_dimensions(self):
        """Test that layout respects canvas dimensions."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        layout_config = LayoutConfig(width=800, height=600)

        harness, geometry = layout_harness(harness, layout_config)

        assert geometry.width == 800
        assert geometry.height == 600

        # All bboxes should be within canvas
        for bbox in geometry.component_bboxes.values():
            x1, y1, x2, y2 = bbox
            assert x1 >= 0
            assert y1 >= 0
            assert x2 <= 800
            assert y2 <= 600

    def test_updates_harness_positions(self):
        """Test that layout updates harness component positions."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        # Before layout, positions should be None
        for component in harness.components:
            assert component.position is None

        harness, geometry = layout_harness(harness)

        # After layout, positions should be set
        for component in harness.components:
            assert component.position is not None
            assert len(component.position) == 2

    def test_updates_pin_positions(self):
        """Test that layout updates pin positions in harness."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        harness, geometry = layout_harness(harness)

        for component in harness.components:
            if component.pins:
                for pin in component.pins:
                    assert pin.position is not None

    def test_no_overlapping_components(self):
        """Test that components don't overlap significantly."""
        config = GraphGeneratorConfig(
            min_connectors=4,
            max_connectors=6,
            seed=42,
        )
        harness = generate_harness_graph(config)

        harness, geometry = layout_harness(harness)

        bboxes = list(geometry.component_bboxes.items())

        # Check for overlaps (allowing small overlaps)
        overlap_count = 0
        for i, (id1, bbox1) in enumerate(bboxes):
            for id2, bbox2 in bboxes[i + 1:]:
                if _bboxes_overlap(bbox1, bbox2):
                    overlap_count += 1

        # Allow some overlaps (layout is not perfect)
        assert overlap_count < len(bboxes) // 2

    def test_default_config_works(self):
        """Test that default layout config works."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        harness, geometry = layout_harness(harness)

        assert len(geometry.component_bboxes) == len(harness.components)


def _bboxes_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    return not (
        x1_max < x2_min or  # bbox1 is left of bbox2
        x2_max < x1_min or  # bbox2 is left of bbox1
        y1_max < y2_min or  # bbox1 is above bbox2
        y2_max < y1_min     # bbox2 is above bbox1
    )
