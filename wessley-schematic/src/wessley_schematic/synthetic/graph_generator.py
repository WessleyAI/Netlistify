"""
Synthetic harness graph generator.

Generates random but valid automotive wiring harness graphs
based on configuration parameters.
"""

import random
from collections import defaultdict

from wessley_schematic.schemas.graph import (
    Component,
    ComponentType,
    Endpoint,
    HarnessGraph,
    Pin,
    PinRole,
    Wire,
)
from wessley_schematic.synthetic.config import GraphGeneratorConfig


def generate_harness_graph(config: GraphGeneratorConfig | None = None) -> HarnessGraph:
    """
    Generate a synthetic harness graph based on configuration.

    Args:
        config: Generation configuration. Uses defaults if None.

    Returns:
        A valid HarnessGraph with randomly generated components and wires.
    """
    if config is None:
        config = GraphGeneratorConfig()

    # Initialize random seed
    if config.seed is not None:
        random.seed(config.seed)

    # Generate components
    components: list[Component] = []
    available_pins: list[tuple[str, str]] = []  # (component_id, pin_id)
    pinless_components: list[str] = []  # component IDs without pins

    # 1. Generate connectors (always present)
    num_connectors = random.randint(config.min_connectors, config.max_connectors)
    for i in range(num_connectors):
        component = _create_connector(i, config)
        components.append(component)
        for pin in component.pins:
            available_pins.append((component.id, pin.id))

    # 2. Generate optional components
    if config.allow_fuses:
        num_fuses = random.randint(0, config.max_fuses)
        for i in range(num_fuses):
            component = _create_fuse(i)
            components.append(component)
            pinless_components.append(component.id)

    if config.allow_relays:
        num_relays = random.randint(0, config.max_relays)
        for i in range(num_relays):
            component = _create_relay(i, config)
            components.append(component)
            # Relays have pins
            for pin in component.pins:
                available_pins.append((component.id, pin.id))

    if config.allow_grounds:
        num_grounds = random.randint(1, config.max_grounds)
        for i in range(num_grounds):
            component = _create_ground(i)
            components.append(component)
            pinless_components.append(component.id)

    if config.allow_splices:
        num_splices = random.randint(0, config.max_splices)
        for i in range(num_splices):
            component = _create_splice(i)
            components.append(component)
            pinless_components.append(component.id)

    if config.allow_ecus:
        num_ecus = random.randint(0, config.max_ecus)
        for i in range(num_ecus):
            component = _create_ecu(i, config)
            components.append(component)
            for pin in component.pins:
                available_pins.append((component.id, pin.id))

    # Generate wires
    num_wires = random.randint(config.min_wires, config.max_wires)

    # Track connection degrees to avoid excessive fan-in/fan-out
    degree_count: dict[tuple[str, str | None], int] = defaultdict(int)

    wires: list[Wire] = []
    wire_id = 0

    # Ensure connectivity: first create a spanning tree
    if config.ensure_connected and len(components) > 1:
        # Shuffle components for random spanning tree
        component_ids = [c.id for c in components]
        random.shuffle(component_ids)

        connected = {component_ids[0]}
        unconnected = set(component_ids[1:])

        while unconnected:
            # Pick a random connected component and an unconnected one
            from_comp_id = random.choice(list(connected))
            to_comp_id = random.choice(list(unconnected))

            # Create wire between them
            wire = _create_wire(
                wire_id,
                from_comp_id,
                to_comp_id,
                components,
                available_pins,
                pinless_components,
                degree_count,
                config,
            )
            if wire:
                wires.append(wire)
                wire_id += 1

            connected.add(to_comp_id)
            unconnected.remove(to_comp_id)

    # Add additional random wires up to the target count
    attempts = 0
    max_attempts = num_wires * 10  # Prevent infinite loops

    while len(wires) < num_wires and attempts < max_attempts:
        attempts += 1

        # Pick two random components
        comp1 = random.choice(components)
        comp2 = random.choice(components)

        if comp1.id == comp2.id:
            continue

        wire = _create_wire(
            wire_id,
            comp1.id,
            comp2.id,
            components,
            available_pins,
            pinless_components,
            degree_count,
            config,
        )
        if wire:
            wires.append(wire)
            wire_id += 1

    return HarnessGraph(
        id=f"harness_{random.randint(1000, 9999)}",
        components=components,
        wires=wires,
    )


def _create_connector(index: int, config: GraphGeneratorConfig) -> Component:
    """Create a connector component with pins."""
    num_pins = random.randint(
        config.min_pins_per_connector, config.max_pins_per_connector
    )
    pins = []
    for i in range(num_pins):
        role = random.choice(list(PinRole))
        pins.append(
            Pin(
                id=str(i + 1),
                role=role,
                label=f"Pin {i + 1}",
            )
        )

    return Component(
        id=f"C{index + 1:03d}",
        type=ComponentType.CONNECTOR,
        pins=pins,
        label=f"Connector {index + 1}",
        attributes={"location": random.choice(["engine bay", "cabin", "trunk", "door"])},
    )


def _create_fuse(index: int) -> Component:
    """Create a fuse component (no pins)."""
    rating = random.choice([5, 10, 15, 20, 25, 30, 40])
    return Component(
        id=f"F{index + 1:02d}",
        type=ComponentType.FUSE,
        pins=[],
        label=f"{rating}A Fuse",
        attributes={"rating_amps": rating},
    )


def _create_relay(index: int, config: GraphGeneratorConfig) -> Component:
    """Create a relay component with standard pins."""
    # Standard relay pins: 85, 86, 30, 87, 87a
    pin_configs = [
        ("85", PinRole.SIGNAL),  # Coil -
        ("86", PinRole.POWER),  # Coil +
        ("30", PinRole.POWER),  # Common
        ("87", PinRole.POWER),  # Normally open
    ]
    if random.random() > 0.5:
        pin_configs.append(("87a", PinRole.POWER))  # Normally closed

    pins = [Pin(id=pid, role=role) for pid, role in pin_configs]

    return Component(
        id=f"R{index + 1:02d}",
        type=ComponentType.RELAY,
        pins=pins,
        label=f"Relay {index + 1}",
    )


def _create_ground(index: int) -> Component:
    """Create a ground point component (no pins)."""
    return Component(
        id=f"G{index + 1:02d}",
        type=ComponentType.GROUND,
        pins=[],
        label=f"Ground {index + 1}",
        attributes={"location": random.choice(["chassis", "engine", "body"])},
    )


def _create_splice(index: int) -> Component:
    """Create a splice point component (no pins)."""
    return Component(
        id=f"S{index + 1:02d}",
        type=ComponentType.SPLICE,
        pins=[],
        label=f"Splice {index + 1}",
    )


def _create_ecu(index: int, config: GraphGeneratorConfig) -> Component:
    """Create an ECU component with pins."""
    num_pins = random.randint(config.min_pins_per_ecu, config.max_pins_per_ecu)
    pins = []

    # ECU pins often use letter-number notation
    for i in range(num_pins):
        letter = chr(ord("A") + (i // 10))
        number = (i % 10) + 1
        role = random.choice(list(PinRole))
        pins.append(
            Pin(
                id=f"{letter}{number}",
                role=role,
            )
        )

    ecu_names = ["ECM", "BCM", "TCM", "ABS", "ADAS", "HVAC"]
    name = random.choice(ecu_names)

    return Component(
        id=f"ECU_{name}_{index + 1}",
        type=ComponentType.ECU,
        pins=pins,
        label=f"{name} Module",
    )


def _create_wire(
    wire_id: int,
    from_comp_id: str,
    to_comp_id: str,
    components: list[Component],
    available_pins: list[tuple[str, str]],
    pinless_components: list[str],
    degree_count: dict[tuple[str, str | None], int],
    config: GraphGeneratorConfig,
) -> Wire | None:
    """
    Create a wire between two components.

    Returns None if the wire cannot be created (e.g., degree limit exceeded).
    """
    # Find the components
    from_comp = next((c for c in components if c.id == from_comp_id), None)
    to_comp = next((c for c in components if c.id == to_comp_id), None)

    if not from_comp or not to_comp:
        return None

    # Determine endpoints
    from_endpoint = _get_endpoint(
        from_comp, available_pins, pinless_components, degree_count, config
    )
    to_endpoint = _get_endpoint(
        to_comp, available_pins, pinless_components, degree_count, config
    )

    if not from_endpoint or not to_endpoint:
        return None

    # Check degree limits
    from_key = (from_endpoint.component_id, from_endpoint.pin_id)
    to_key = (to_endpoint.component_id, to_endpoint.pin_id)

    if degree_count[from_key] >= config.max_degree:
        return None
    if degree_count[to_key] >= config.max_degree:
        return None

    # Update degree counts
    degree_count[from_key] += 1
    degree_count[to_key] += 1

    # Create wire with random attributes
    color = random.choice(config.wire_colors) if config.wire_colors else None
    gauge = random.choice(config.wire_gauges) if config.wire_gauges else None

    return Wire(
        id=f"W{wire_id + 1:03d}",
        from_endpoint=from_endpoint,
        to_endpoint=to_endpoint,
        color=color,
        gauge_mm2=gauge,
    )


def _get_endpoint(
    component: Component,
    available_pins: list[tuple[str, str]],
    pinless_components: list[str],
    degree_count: dict[tuple[str, str | None], int],
    config: GraphGeneratorConfig,
) -> Endpoint | None:
    """Get an endpoint for a wire connection to a component."""
    if component.pins:
        # Component has pins - pick one with available capacity
        comp_pins = [(cid, pid) for cid, pid in available_pins if cid == component.id]
        if not comp_pins:
            return None

        # Sort by degree (prefer less connected pins)
        comp_pins.sort(key=lambda x: degree_count[(x[0], x[1])])

        for comp_id, pin_id in comp_pins:
            if degree_count[(comp_id, pin_id)] < config.max_degree:
                return Endpoint(component_id=comp_id, pin_id=pin_id)

        return None
    else:
        # Pinless component
        if component.id in pinless_components:
            if degree_count[(component.id, None)] < config.max_degree:
                return Endpoint(component_id=component.id, pin_id=None)
        return None
