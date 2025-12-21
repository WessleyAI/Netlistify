"""
Canonical data model for automotive wiring harness graphs.

This module defines the core schema for representing wiring harnesses in Wessley AI:
- Components (connectors, fuses, relays, grounds, splices, ECUs)
- Pins with roles and net assignments
- Wires with endpoints and physical attributes
- Nets representing logical connectivity
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ComponentType(str, Enum):
    """Type of component in the wiring harness."""

    CONNECTOR = "connector"
    FUSE = "fuse"
    RELAY = "relay"
    GROUND = "ground"
    SPLICE = "splice"
    ECU = "ecu"
    NODE = "node"  # Generic junction node


class PinRole(str, Enum):
    """Role/function of a pin in the harness."""

    POWER = "power"
    GROUND = "ground"
    SIGNAL = "signal"
    UNKNOWN = "unknown"


class Pin(BaseModel):
    """
    A pin on a component (connector, ECU, etc.).

    Pins are the connection points where wires attach.
    """

    id: str = Field(..., description="Pin identifier (e.g., '1', '2', 'A1')")
    role: PinRole | None = Field(default=None, description="Electrical role of the pin")
    net_id: str | None = Field(default=None, description="ID of the net this pin belongs to")
    label: str | None = Field(default=None, description="Optional display label")

    # Layout fields (set during layout phase)
    position: tuple[float, float] | None = Field(
        default=None, description="2D position (x, y) in diagram coordinates"
    )


class Component(BaseModel):
    """
    A component in the wiring harness (connector, fuse, relay, etc.).

    Components are the major elements that appear in wiring diagrams.
    """

    id: str = Field(..., description="Unique component ID (e.g., 'C101', 'F10', 'ECU_A')")
    type: ComponentType = Field(..., description="Type of component")
    pins: list[Pin] = Field(default_factory=list, description="List of pins on this component")
    label: str | None = Field(default=None, description="Display label")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., location, part number)",
    )

    # Layout fields (set during layout phase)
    position: tuple[float, float] | None = Field(
        default=None, description="Center position (x, y) in diagram coordinates"
    )
    size: tuple[float, float] | None = Field(
        default=None, description="Size (width, height) in diagram coordinates"
    )

    def get_pin(self, pin_id: str) -> Pin | None:
        """Get a pin by its ID."""
        for pin in self.pins:
            if pin.id == pin_id:
                return pin
        return None

    @property
    def has_pins(self) -> bool:
        """Check if this component type typically has pins."""
        return self.type in (ComponentType.CONNECTOR, ComponentType.ECU, ComponentType.RELAY)


class Endpoint(BaseModel):
    """
    An endpoint of a wire, specifying where it connects.

    For components with pins (connectors, ECUs), pin_id should be specified.
    For components without pins (grounds, fuses), pin_id may be None.
    """

    component_id: str = Field(..., description="ID of the component this endpoint connects to")
    pin_id: str | None = Field(
        default=None, description="ID of the specific pin (None for pinless components)"
    )

    def __hash__(self) -> int:
        return hash((self.component_id, self.pin_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Endpoint):
            return False
        return self.component_id == other.component_id and self.pin_id == other.pin_id


class Wire(BaseModel):
    """
    A wire connecting two endpoints in the harness.

    Wires represent the physical conductors between components.
    """

    id: str = Field(..., description="Unique wire ID (e.g., 'W1', 'W42')")
    from_endpoint: Endpoint = Field(..., description="Starting endpoint of the wire")
    to_endpoint: Endpoint = Field(..., description="Ending endpoint of the wire")
    color: str | None = Field(
        default=None, description="Wire color code (e.g., 'RD', 'BL', 'RD/BL')"
    )
    gauge_mm2: float | None = Field(default=None, description="Wire gauge in mm²")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., length, shielding)",
    )

    # Layout fields (set during rendering phase)
    polyline: list[tuple[float, float]] | None = Field(
        default=None, description="List of (x, y) points forming the wire path"
    )


class Net(BaseModel):
    """
    A net representing a logical electrical connection.

    A net groups all wires and pins that are electrically connected.
    """

    id: str = Field(..., description="Unique net ID (e.g., 'NET_1', 'POWER_12V')")
    wire_ids: list[str] = Field(default_factory=list, description="IDs of wires in this net")
    name: str | None = Field(default=None, description="Human-readable net name")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., voltage, signal type)",
    )


class HarnessGraph(BaseModel):
    """
    Complete representation of an automotive wiring harness graph.

    This is the top-level data structure containing all components,
    wires, and optional net information.
    """

    id: str = Field(default="harness", description="Harness identifier")
    components: list[Component] = Field(
        default_factory=list, description="All components in the harness"
    )
    wires: list[Wire] = Field(default_factory=list, description="All wires connecting components")
    nets: list[Net] = Field(
        default_factory=list, description="Optional net groupings (can be derived from wires)"
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., vehicle model, harness name)",
    )

    def get_component(self, component_id: str) -> Component | None:
        """Get a component by its ID."""
        for component in self.components:
            if component.id == component_id:
                return component
        return None

    def get_wire(self, wire_id: str) -> Wire | None:
        """Get a wire by its ID."""
        for wire in self.wires:
            if wire.id == wire_id:
                return wire
        return None

    def get_components_by_type(self, component_type: ComponentType) -> list[Component]:
        """Get all components of a specific type."""
        return [c for c in self.components if c.type == component_type]

    def get_connected_wires(self, component_id: str, pin_id: str | None = None) -> list[Wire]:
        """Get all wires connected to a component (and optionally a specific pin)."""
        result = []
        for wire in self.wires:
            for endpoint in (wire.from_endpoint, wire.to_endpoint):
                if endpoint.component_id == component_id:
                    if pin_id is None or endpoint.pin_id == pin_id:
                        result.append(wire)
                        break
        return result

    def validate_connectivity(self) -> list[str]:
        """
        Validate that all wire endpoints reference valid components and pins.

        Returns a list of error messages (empty if valid).
        """
        errors = []
        component_ids = {c.id for c in self.components}
        component_pin_map: dict[str, set[str]] = {}

        for component in self.components:
            component_pin_map[component.id] = {p.id for p in component.pins}

        for wire in self.wires:
            for endpoint, label in [
                (wire.from_endpoint, "from"),
                (wire.to_endpoint, "to"),
            ]:
                if endpoint.component_id not in component_ids:
                    errors.append(
                        f"Wire {wire.id}: {label}_endpoint references "
                        f"non-existent component '{endpoint.component_id}'"
                    )
                elif endpoint.pin_id is not None:
                    valid_pins = component_pin_map.get(endpoint.component_id, set())
                    if endpoint.pin_id not in valid_pins:
                        errors.append(
                            f"Wire {wire.id}: {label}_endpoint references "
                            f"non-existent pin '{endpoint.pin_id}' "
                            f"on component '{endpoint.component_id}'"
                        )

        return errors

    @model_validator(mode="after")
    def check_wire_references(self) -> "HarnessGraph":
        """Validate that all wire endpoints reference valid components."""
        errors = self.validate_connectivity()
        if errors:
            raise ValueError(f"Invalid harness graph: {'; '.join(errors)}")
        return self

    def to_adjacency_list(self) -> dict[str, list[str]]:
        """
        Convert the harness to an adjacency list representation.

        Returns a dict mapping component IDs to lists of connected component IDs.
        """
        adjacency: dict[str, list[str]] = {c.id: [] for c in self.components}

        for wire in self.wires:
            from_id = wire.from_endpoint.component_id
            to_id = wire.to_endpoint.component_id
            if to_id not in adjacency[from_id]:
                adjacency[from_id].append(to_id)
            if from_id not in adjacency[to_id]:
                adjacency[to_id].append(from_id)

        return adjacency

    def is_connected(self) -> bool:
        """Check if the harness graph is fully connected."""
        if not self.components:
            return True

        adjacency = self.to_adjacency_list()
        visited: set[str] = set()
        stack = [self.components[0].id]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency[current])

        return len(visited) == len(self.components)
