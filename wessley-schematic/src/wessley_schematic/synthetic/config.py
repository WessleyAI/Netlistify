"""
Configuration for synthetic harness graph generation.
"""

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator


class HarnessPreset(str, Enum):
    """Predefined harness complexity presets."""

    TINY = "tiny"  # 2-3 connectors, 3-5 wires - for quick tests
    BASE = "base"  # 4-6 connectors, fuses, grounds, 10-20 wires - typical diagram
    COMPLEX = "complex"  # 8-12 connectors, splices, relays, 30-50 wires - dense diagram


class GraphGeneratorConfig(BaseModel):
    """
    Configuration for generating synthetic harness graphs.

    Controls the complexity and characteristics of generated diagrams.
    """

    # Component counts
    min_connectors: int = Field(default=2, ge=1, description="Minimum number of connectors")
    max_connectors: int = Field(default=6, ge=1, description="Maximum number of connectors")
    min_wires: int = Field(default=5, ge=1, description="Minimum number of wires")
    max_wires: int = Field(default=20, ge=1, description="Maximum number of wires")

    # Optional component types
    allow_fuses: bool = Field(default=True, description="Include fuses in generated graphs")
    allow_relays: bool = Field(default=True, description="Include relays in generated graphs")
    allow_splices: bool = Field(default=True, description="Include splices in generated graphs")
    allow_ecus: bool = Field(default=True, description="Include ECUs in generated graphs")
    allow_grounds: bool = Field(default=True, description="Include ground points")

    # Component limits
    max_fuses: int = Field(default=4, ge=0, description="Maximum number of fuses")
    max_relays: int = Field(default=3, ge=0, description="Maximum number of relays")
    max_splices: int = Field(default=3, ge=0, description="Maximum number of splices")
    max_ecus: int = Field(default=2, ge=0, description="Maximum number of ECUs")
    max_grounds: int = Field(default=3, ge=0, description="Maximum number of ground points")

    # Pin configuration
    min_pins_per_connector: int = Field(
        default=2, ge=1, description="Minimum pins per connector"
    )
    max_pins_per_connector: int = Field(
        default=8, ge=1, description="Maximum pins per connector"
    )
    min_pins_per_ecu: int = Field(default=4, ge=1, description="Minimum pins per ECU")
    max_pins_per_ecu: int = Field(default=12, ge=1, description="Maximum pins per ECU")

    # Graph properties
    ensure_connected: bool = Field(
        default=True, description="Ensure the generated graph is fully connected"
    )
    max_degree: int = Field(
        default=6, ge=1, description="Maximum number of wires per component/pin"
    )

    # Wire attributes
    wire_colors: list[str] = Field(
        default_factory=lambda: [
            "RD",
            "BL",
            "GN",
            "YL",
            "BK",
            "WH",
            "OR",
            "BR",
            "PK",
            "RD/BL",
            "GN/YL",
            "BK/WH",
        ],
        description="Possible wire colors",
    )
    wire_gauges: list[float] = Field(
        default_factory=lambda: [0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 4.0],
        description="Possible wire gauges in mm²",
    )

    # Randomization
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility (None for random)"
    )

    @field_validator("max_connectors")
    @classmethod
    def max_connectors_gte_min(cls, v: int, info) -> int:
        min_val = info.data.get("min_connectors", 2)
        if v < min_val:
            raise ValueError(f"max_connectors ({v}) must be >= min_connectors ({min_val})")
        return v

    @field_validator("max_wires")
    @classmethod
    def max_wires_gte_min(cls, v: int, info) -> int:
        min_val = info.data.get("min_wires", 5)
        if v < min_val:
            raise ValueError(f"max_wires ({v}) must be >= min_wires ({min_val})")
        return v

    @field_validator("max_pins_per_connector")
    @classmethod
    def max_pins_connector_gte_min(cls, v: int, info) -> int:
        min_val = info.data.get("min_pins_per_connector", 2)
        if v < min_val:
            raise ValueError(
                f"max_pins_per_connector ({v}) must be >= min_pins_per_connector ({min_val})"
            )
        return v

    @field_validator("max_pins_per_ecu")
    @classmethod
    def max_pins_ecu_gte_min(cls, v: int, info) -> int:
        min_val = info.data.get("min_pins_per_ecu", 4)
        if v < min_val:
            raise ValueError(
                f"max_pins_per_ecu ({v}) must be >= min_pins_per_ecu ({min_val})"
            )
        return v

    @classmethod
    def from_preset(cls, preset: HarnessPreset, seed: int | None = None) -> "GraphGeneratorConfig":
        """
        Create a configuration from a predefined preset.

        Args:
            preset: The harness complexity preset
            seed: Optional random seed

        Returns:
            GraphGeneratorConfig configured for the preset
        """
        configs = {
            HarnessPreset.TINY: {
                "min_connectors": 2,
                "max_connectors": 3,
                "min_wires": 3,
                "max_wires": 5,
                "allow_fuses": False,
                "allow_relays": False,
                "allow_splices": False,
                "allow_ecus": False,
                "allow_grounds": True,
                "max_grounds": 1,
                "min_pins_per_connector": 2,
                "max_pins_per_connector": 4,
            },
            HarnessPreset.BASE: {
                "min_connectors": 4,
                "max_connectors": 6,
                "min_wires": 10,
                "max_wires": 20,
                "allow_fuses": True,
                "allow_relays": False,
                "allow_splices": False,
                "allow_ecus": False,
                "allow_grounds": True,
                "max_fuses": 2,
                "max_grounds": 2,
                "min_pins_per_connector": 2,
                "max_pins_per_connector": 6,
            },
            HarnessPreset.COMPLEX: {
                "min_connectors": 8,
                "max_connectors": 12,
                "min_wires": 30,
                "max_wires": 50,
                "allow_fuses": True,
                "allow_relays": True,
                "allow_splices": True,
                "allow_ecus": True,
                "allow_grounds": True,
                "max_fuses": 4,
                "max_relays": 3,
                "max_splices": 4,
                "max_ecus": 2,
                "max_grounds": 4,
                "min_pins_per_connector": 3,
                "max_pins_per_connector": 10,
                "min_pins_per_ecu": 6,
                "max_pins_per_ecu": 16,
            },
        }
        return cls(seed=seed, **configs[preset])


class LayoutConfig(BaseModel):
    """
    Configuration for the layout engine.
    """

    # Canvas dimensions (in abstract units)
    width: float = Field(default=1000.0, gt=0, description="Canvas width")
    height: float = Field(default=800.0, gt=0, description="Canvas height")

    # Margins
    margin_left: float = Field(default=50.0, ge=0, description="Left margin")
    margin_right: float = Field(default=50.0, ge=0, description="Right margin")
    margin_top: float = Field(default=50.0, ge=0, description="Top margin")
    margin_bottom: float = Field(default=50.0, ge=0, description="Bottom margin")

    # Component sizing
    connector_width: float = Field(default=80.0, gt=0, description="Connector box width")
    connector_height_per_pin: float = Field(
        default=20.0, gt=0, description="Height per pin in connector"
    )
    fuse_size: float = Field(default=40.0, gt=0, description="Fuse symbol size")
    relay_size: float = Field(default=50.0, gt=0, description="Relay symbol size")
    ground_size: float = Field(default=30.0, gt=0, description="Ground symbol size")
    splice_size: float = Field(default=20.0, gt=0, description="Splice dot size")
    ecu_width: float = Field(default=100.0, gt=0, description="ECU box width")
    ecu_height_per_pin: float = Field(default=15.0, gt=0, description="Height per pin in ECU")

    # Spacing
    component_spacing: float = Field(
        default=100.0, gt=0, description="Minimum spacing between components"
    )

    # Wire routing
    wire_corner_radius: float = Field(default=5.0, ge=0, description="Corner radius for wires")


class RenderConfig(BaseModel):
    """
    Configuration for SVG rendering.
    """

    # Output dimensions
    width: int = Field(default=1200, gt=0, description="Output image width in pixels")
    height: int = Field(default=960, gt=0, description="Output image height in pixels")

    # Colors
    background_color: str = Field(default="#FFFFFF", description="Background color")
    wire_color: str = Field(default="#333333", description="Default wire color")
    component_fill: str = Field(default="#F5F5F5", description="Component fill color")
    component_stroke: str = Field(default="#333333", description="Component stroke color")
    text_color: str = Field(default="#000000", description="Text label color")

    # Stroke widths
    wire_stroke_width: float = Field(default=1.5, gt=0, description="Wire line width")
    component_stroke_width: float = Field(default=2.0, gt=0, description="Component border width")

    # Font
    font_family: str = Field(default="Arial, sans-serif", description="Font family for labels")
    font_size_component: float = Field(default=12.0, gt=0, description="Component label font size")
    font_size_pin: float = Field(default=9.0, gt=0, description="Pin label font size")
    font_size_wire: float = Field(default=8.0, gt=0, description="Wire label font size")

    # Feature toggles
    show_pin_labels: bool = Field(default=True, description="Show pin number labels")
    show_wire_labels: bool = Field(default=True, description="Show wire color/gauge labels")
    show_component_labels: bool = Field(default=True, description="Show component ID labels")

    # PNG output
    generate_png: bool = Field(default=False, description="Also generate PNG rasterization")
    png_scale: float = Field(default=1.0, gt=0, description="Scale factor for PNG output")
