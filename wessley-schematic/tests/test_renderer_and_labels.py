"""Tests for the renderer and label exporter modules."""

import json
import tempfile
from pathlib import Path

import pytest

from wessley_schematic.schemas.graph import ComponentType
from wessley_schematic.synthetic.config import (
    GraphGeneratorConfig,
    LayoutConfig,
    RenderConfig,
)
from wessley_schematic.synthetic.graph_generator import generate_harness_graph
from wessley_schematic.synthetic.labels import export_labels
from wessley_schematic.synthetic.layout import layout_harness
from wessley_schematic.synthetic.renderer_svg import RenderGeometry, render_svg


class TestRenderer:
    """Tests for the SVG renderer."""

    def test_creates_svg_file(self):
        """Test that renderer creates an SVG file."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, geometry, svg_path)

            assert svg_path.exists()
            assert svg_path.stat().st_size > 0

    def test_returns_render_geometry(self):
        """Test that renderer returns RenderGeometry."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, geometry, svg_path)

            assert isinstance(render_geometry, RenderGeometry)

    def test_svg_contains_components(self):
        """Test that SVG contains component elements."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_svg(harness, geometry, svg_path)

            svg_content = svg_path.read_text()

            # Check for component group
            assert 'id="components"' in svg_content

            # Check for some component IDs
            for component in harness.components[:3]:
                assert f'id="component_{component.id}"' in svg_content

    def test_svg_contains_wires(self):
        """Test that SVG contains wire elements."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_svg(harness, geometry, svg_path)

            svg_content = svg_path.read_text()

            # Check for wires group
            assert 'id="wires"' in svg_content

    def test_wire_polylines_recorded(self):
        """Test that wire polylines are recorded in render geometry."""
        config = GraphGeneratorConfig(seed=42, min_wires=5)
        harness = generate_harness_graph(config)
        harness, geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, geometry, svg_path)

            # At least some wires should have polylines
            assert len(render_geometry.wire_polylines) > 0

            # Polylines should have at least 2 points
            for wire_id, polyline in render_geometry.wire_polylines.items():
                assert len(polyline) >= 2
                for point in polyline:
                    assert len(point) == 2

    def test_respects_render_config(self):
        """Test that renderer respects configuration."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, geometry = layout_harness(harness)

        render_config = RenderConfig(
            width=800,
            height=600,
            show_pin_labels=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_svg(harness, geometry, svg_path, render_config)

            svg_content = svg_path.read_text()

            # Check dimensions
            assert 'width="800"' in svg_content
            assert 'height="600"' in svg_content


class TestLabelExporter:
    """Tests for the label exporter."""

    def test_exports_all_label_files(self):
        """Test that exporter creates all expected label files."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, layout_geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, layout_geometry, svg_path)

            labels_dir = Path(tmpdir) / "labels"
            output_files = export_labels(
                harness,
                layout_geometry,
                render_geometry,
                labels_dir,
                "test_sample",
            )

            assert "objects" in output_files
            assert "wires" in output_files
            assert "graph" in output_files
            assert "yolo" in output_files
            assert "coco" in output_files

            for path in output_files.values():
                assert path.exists()

    def test_objects_json_format(self):
        """Test that objects.json has correct format."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, layout_geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, layout_geometry, svg_path)

            labels_dir = Path(tmpdir) / "labels"
            output_files = export_labels(
                harness,
                layout_geometry,
                render_geometry,
                labels_dir,
                "test_sample",
            )

            with open(output_files["objects"]) as f:
                data = json.load(f)

            assert "image" in data
            assert "objects" in data
            assert "image_width" in data
            assert "image_height" in data

            for obj in data["objects"]:
                assert "id" in obj
                assert "class" in obj
                assert "bbox" in obj
                assert len(obj["bbox"]) == 4

    def test_wires_json_format(self):
        """Test that wires.json has correct format."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, layout_geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, layout_geometry, svg_path)

            labels_dir = Path(tmpdir) / "labels"
            output_files = export_labels(
                harness,
                layout_geometry,
                render_geometry,
                labels_dir,
                "test_sample",
            )

            with open(output_files["wires"]) as f:
                data = json.load(f)

            assert "wires" in data

            for wire in data["wires"]:
                assert "id" in wire
                assert "polyline" in wire
                assert "from" in wire
                assert "to" in wire

    def test_graph_json_format(self):
        """Test that graph.json has correct format."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, layout_geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, layout_geometry, svg_path)

            labels_dir = Path(tmpdir) / "labels"
            output_files = export_labels(
                harness,
                layout_geometry,
                render_geometry,
                labels_dir,
                "test_sample",
            )

            with open(output_files["graph"]) as f:
                data = json.load(f)

            assert "id" in data
            assert "components" in data
            assert "wires" in data

            # Check component structure
            for comp in data["components"]:
                assert "id" in comp
                assert "type" in comp
                assert "pins" in comp

            # Check wire structure
            for wire in data["wires"]:
                assert "id" in wire
                assert "from_endpoint" in wire
                assert "to_endpoint" in wire

    def test_yolo_format(self):
        """Test that YOLO labels have correct format."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, layout_geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, layout_geometry, svg_path)

            labels_dir = Path(tmpdir) / "labels"
            output_files = export_labels(
                harness,
                layout_geometry,
                render_geometry,
                labels_dir,
                "test_sample",
            )

            content = output_files["yolo"].read_text()
            lines = [l for l in content.strip().split("\n") if l]

            assert len(lines) == len(harness.components)

            for line in lines:
                parts = line.split()
                assert len(parts) == 5  # class_id, x_center, y_center, width, height

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # YOLO coordinates are normalized [0, 1]
                assert 0 <= x_center <= 1
                assert 0 <= y_center <= 1
                assert 0 < width <= 1
                assert 0 < height <= 1

    def test_coco_format(self):
        """Test that COCO labels have correct format."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)
        harness, layout_geometry = layout_harness(harness)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "test.svg"
            render_geometry = render_svg(harness, layout_geometry, svg_path)

            labels_dir = Path(tmpdir) / "labels"
            output_files = export_labels(
                harness,
                layout_geometry,
                render_geometry,
                labels_dir,
                "test_sample",
            )

            with open(output_files["coco"]) as f:
                data = json.load(f)

            assert "images" in data
            assert "annotations" in data
            assert "categories" in data

            # Check image info
            assert len(data["images"]) == 1
            assert data["images"][0]["file_name"] == "test_sample.png"

            # Check categories
            category_names = [c["name"] for c in data["categories"]]
            assert "connector" in category_names
            assert "fuse" in category_names

            # Check annotations
            for ann in data["annotations"]:
                assert "id" in ann
                assert "image_id" in ann
                assert "category_id" in ann
                assert "bbox" in ann
                assert len(ann["bbox"]) == 4
                assert "area" in ann


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test the complete generation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            svgs_dir = tmpdir / "svgs"
            labels_dir = tmpdir / "labels"
            svgs_dir.mkdir()
            labels_dir.mkdir()

            # Generate multiple samples
            for i in range(3):
                config = GraphGeneratorConfig(seed=42 + i)
                harness = generate_harness_graph(config)
                harness, layout_geometry = layout_harness(harness)

                sample_name = f"sample_{i:04d}"
                svg_path = svgs_dir / f"{sample_name}.svg"
                render_geometry = render_svg(harness, layout_geometry, svg_path)

                export_labels(
                    harness,
                    layout_geometry,
                    render_geometry,
                    labels_dir,
                    sample_name,
                )

            # Check all files exist
            assert len(list(svgs_dir.glob("*.svg"))) == 3
            assert len(list(labels_dir.glob("*.objects.json"))) == 3
            assert len(list(labels_dir.glob("*.wires.json"))) == 3
            assert len(list(labels_dir.glob("*.graph.json"))) == 3
            assert len(list(labels_dir.glob("*.txt"))) == 3
            assert len(list(labels_dir.glob("*.coco.json"))) == 3
