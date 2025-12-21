"""
Wessley Schematic - Automotive Wiring Diagram Synthetic Data & Parsing

This package provides tools for generating synthetic automotive wiring diagrams
and exporting them with ML-ready labels for training object detection and
graph reconstruction models.
"""

__version__ = "0.1.0"

from wessley_schematic.schemas.graph import (
    Component,
    ComponentType,
    Endpoint,
    HarnessGraph,
    Net,
    Pin,
    PinRole,
    Wire,
)
from wessley_schematic.synthetic.config import GraphGeneratorConfig

__all__ = [
    "HarnessGraph",
    "Component",
    "ComponentType",
    "Pin",
    "PinRole",
    "Wire",
    "Endpoint",
    "Net",
    "GraphGeneratorConfig",
]
