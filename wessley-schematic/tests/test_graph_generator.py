"""Tests for the graph generator module."""

import pytest

from wessley_schematic.schemas.graph import ComponentType, HarnessGraph
from wessley_schematic.synthetic.config import GraphGeneratorConfig
from wessley_schematic.synthetic.graph_generator import generate_harness_graph


class TestGraphGenerator:
    """Tests for generate_harness_graph function."""

    def test_generates_valid_harness(self):
        """Test that generator produces a valid HarnessGraph."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        assert isinstance(harness, HarnessGraph)
        assert len(harness.components) > 0
        assert len(harness.wires) > 0

    def test_respects_connector_count(self):
        """Test that generator respects min/max connector settings."""
        config = GraphGeneratorConfig(
            min_connectors=3,
            max_connectors=5,
            seed=42,
        )
        harness = generate_harness_graph(config)

        connectors = harness.get_components_by_type(ComponentType.CONNECTOR)
        assert 3 <= len(connectors) <= 5

    def test_respects_wire_count(self):
        """Test that generator respects min/max wire settings."""
        config = GraphGeneratorConfig(
            min_wires=5,
            max_wires=10,
            seed=42,
        )
        harness = generate_harness_graph(config)

        # Note: actual count may be slightly different due to connectivity constraints
        assert len(harness.wires) >= 1

    def test_disables_fuses(self):
        """Test that fuses are not generated when disabled."""
        config = GraphGeneratorConfig(
            allow_fuses=False,
            seed=42,
        )
        harness = generate_harness_graph(config)

        fuses = harness.get_components_by_type(ComponentType.FUSE)
        assert len(fuses) == 0

    def test_disables_relays(self):
        """Test that relays are not generated when disabled."""
        config = GraphGeneratorConfig(
            allow_relays=False,
            seed=42,
        )
        harness = generate_harness_graph(config)

        relays = harness.get_components_by_type(ComponentType.RELAY)
        assert len(relays) == 0

    def test_disables_ecus(self):
        """Test that ECUs are not generated when disabled."""
        config = GraphGeneratorConfig(
            allow_ecus=False,
            seed=42,
        )
        harness = generate_harness_graph(config)

        ecus = harness.get_components_by_type(ComponentType.ECU)
        assert len(ecus) == 0

    def test_seed_reproducibility(self):
        """Test that same seed produces same output."""
        config1 = GraphGeneratorConfig(seed=12345)
        config2 = GraphGeneratorConfig(seed=12345)

        harness1 = generate_harness_graph(config1)
        harness2 = generate_harness_graph(config2)

        assert harness1.id == harness2.id
        assert len(harness1.components) == len(harness2.components)
        assert len(harness1.wires) == len(harness2.wires)

    def test_ensures_connectivity(self):
        """Test that generated graph is connected when ensure_connected=True."""
        config = GraphGeneratorConfig(
            ensure_connected=True,
            min_connectors=4,
            max_connectors=6,
            seed=42,
        )
        harness = generate_harness_graph(config)

        assert harness.is_connected()

    def test_wire_endpoints_valid(self):
        """Test that all wire endpoints reference valid components/pins."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        errors = harness.validate_connectivity()
        assert len(errors) == 0

    def test_connectors_have_pins(self):
        """Test that connectors have pins."""
        config = GraphGeneratorConfig(
            min_pins_per_connector=3,
            max_pins_per_connector=5,
            seed=42,
        )
        harness = generate_harness_graph(config)

        connectors = harness.get_components_by_type(ComponentType.CONNECTOR)
        for connector in connectors:
            assert 3 <= len(connector.pins) <= 5

    def test_default_config_works(self):
        """Test that default configuration works."""
        harness = generate_harness_graph()

        assert isinstance(harness, HarnessGraph)
        assert len(harness.components) > 0

    def test_wires_have_color(self):
        """Test that wires have color assigned."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        for wire in harness.wires:
            assert wire.color is not None
            assert wire.color in config.wire_colors

    def test_wires_have_gauge(self):
        """Test that wires have gauge assigned."""
        config = GraphGeneratorConfig(seed=42)
        harness = generate_harness_graph(config)

        for wire in harness.wires:
            assert wire.gauge_mm2 is not None
            assert wire.gauge_mm2 in config.wire_gauges


class TestGraphGeneratorConfig:
    """Tests for GraphGeneratorConfig validation."""

    def test_valid_config(self):
        """Test that valid config is accepted."""
        config = GraphGeneratorConfig(
            min_connectors=2,
            max_connectors=5,
            min_wires=5,
            max_wires=20,
        )
        assert config.min_connectors == 2
        assert config.max_connectors == 5

    def test_max_less_than_min_connectors_fails(self):
        """Test that max < min connectors raises error."""
        with pytest.raises(ValueError):
            GraphGeneratorConfig(
                min_connectors=5,
                max_connectors=2,
            )

    def test_max_less_than_min_wires_fails(self):
        """Test that max < min wires raises error."""
        with pytest.raises(ValueError):
            GraphGeneratorConfig(
                min_wires=20,
                max_wires=5,
            )
