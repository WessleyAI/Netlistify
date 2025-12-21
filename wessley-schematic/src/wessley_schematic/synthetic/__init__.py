"""Synthetic data generation for automotive wiring diagrams."""

from wessley_schematic.synthetic.config import GraphGeneratorConfig, HarnessPreset
from wessley_schematic.synthetic.graph_generator import generate_harness_graph
from wessley_schematic.synthetic.labels import export_labels
from wessley_schematic.synthetic.layout import layout_harness
from wessley_schematic.synthetic.renderer_svg import render_svg

__all__ = [
    "GraphGeneratorConfig",
    "HarnessPreset",
    "generate_harness_graph",
    "layout_harness",
    "render_svg",
    "export_labels",
]
