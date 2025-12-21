"""
CLI for wessley-schematic.

Provides commands for generating synthetic automotive wiring diagrams
and exporting ML-ready labels.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from wessley_schematic.synthetic.config import (
    GraphGeneratorConfig,
    LayoutConfig,
    RenderConfig,
)
from wessley_schematic.synthetic.graph_generator import generate_harness_graph
from wessley_schematic.synthetic.labels import aggregate_coco_annotations, export_labels
from wessley_schematic.synthetic.layout import layout_harness
from wessley_schematic.synthetic.renderer_svg import render_svg

app = typer.Typer(
    name="wessley-schematic",
    help="Automotive Wiring Diagram Synthetic Data Generator",
    add_completion=False,
)

console = Console()


@app.command()
def generate(
    output_dir: Path = typer.Option(
        "./dataset",
        "--output-dir",
        "-o",
        help="Output directory for generated dataset",
    ),
    num_samples: int = typer.Option(
        10,
        "--num-samples",
        "-n",
        help="Number of synthetic samples to generate",
    ),
    min_connectors: int = typer.Option(
        2,
        "--min-connectors",
        help="Minimum number of connectors per diagram",
    ),
    max_connectors: int = typer.Option(
        6,
        "--max-connectors",
        help="Maximum number of connectors per diagram",
    ),
    min_wires: int = typer.Option(
        5,
        "--min-wires",
        help="Minimum number of wires per diagram",
    ),
    max_wires: int = typer.Option(
        20,
        "--max-wires",
        help="Maximum number of wires per diagram",
    ),
    allow_fuses: bool = typer.Option(True, help="Include fuses in diagrams"),
    allow_relays: bool = typer.Option(True, help="Include relays in diagrams"),
    allow_splices: bool = typer.Option(True, help="Include splices in diagrams"),
    allow_ecus: bool = typer.Option(True, help="Include ECUs in diagrams"),
    allow_grounds: bool = typer.Option(True, help="Include ground points in diagrams"),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    generate_png: bool = typer.Option(
        False,
        "--png",
        help="Also generate PNG rasterizations (requires cairosvg)",
    ),
    width: int = typer.Option(
        1200,
        "--width",
        help="Output image width in pixels",
    ),
    height: int = typer.Option(
        960,
        "--height",
        help="Output image height in pixels",
    ),
    aggregate_coco: bool = typer.Option(
        True,
        "--aggregate-coco/--no-aggregate-coco",
        help="Create aggregated COCO annotations file",
    ),
) -> None:
    """
    Generate synthetic automotive wiring diagrams with ML labels.

    Creates a dataset with SVG/PNG images and corresponding label files
    in multiple formats (YOLO, COCO, custom JSON).
    """
    output_dir = Path(output_dir)

    # Create directory structure
    images_dir = output_dir / "images"
    svgs_dir = output_dir / "svgs"
    labels_dir = output_dir / "labels"

    for d in [images_dir, svgs_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create configs
    gen_config = GraphGeneratorConfig(
        min_connectors=min_connectors,
        max_connectors=max_connectors,
        min_wires=min_wires,
        max_wires=max_wires,
        allow_fuses=allow_fuses,
        allow_relays=allow_relays,
        allow_splices=allow_splices,
        allow_ecus=allow_ecus,
        allow_grounds=allow_grounds,
        seed=seed,
    )

    layout_config = LayoutConfig()

    render_config = RenderConfig(
        width=width,
        height=height,
        generate_png=generate_png,
    )

    console.print(f"\n[bold blue]Generating {num_samples} synthetic samples...[/bold blue]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=num_samples)

        for i in range(num_samples):
            sample_name = f"synthetic_{i + 1:04d}"
            progress.update(task, description=f"Generating {sample_name}...")

            # Update seed for each sample if a base seed was provided
            if seed is not None:
                gen_config.seed = seed + i
            else:
                gen_config.seed = None

            # Generate graph
            harness = generate_harness_graph(gen_config)

            # Layout
            harness, layout_geometry = layout_harness(harness, layout_config)

            # Render SVG
            svg_path = svgs_dir / f"{sample_name}.svg"
            render_geometry = render_svg(harness, layout_geometry, svg_path, render_config)

            # If PNG generation is enabled, copy to images dir
            if generate_png:
                png_src = svg_path.with_suffix(".png")
                if png_src.exists():
                    import shutil
                    shutil.copy(png_src, images_dir / f"{sample_name}.png")

            # Export labels
            export_labels(
                harness,
                layout_geometry,
                render_geometry,
                labels_dir,
                sample_name,
                render_config,
            )

            progress.advance(task)

    # Aggregate COCO annotations
    if aggregate_coco:
        console.print("\n[bold]Aggregating COCO annotations...[/bold]")
        aggregate_coco_annotations(
            labels_dir,
            output_dir / "annotations.json",
        )

    # Print summary
    console.print("\n[bold green]Generation complete![/bold green]\n")

    table = Table(title="Dataset Summary")
    table.add_column("Directory", style="cyan")
    table.add_column("Contents", style="green")

    table.add_row("svgs/", f"{num_samples} SVG files")
    if generate_png:
        table.add_row("images/", f"{num_samples} PNG files")
    table.add_row("labels/", f"{num_samples * 5} label files")
    if aggregate_coco:
        table.add_row("annotations.json", "Aggregated COCO dataset")

    console.print(table)
    console.print(f"\n[dim]Output directory: {output_dir.absolute()}[/dim]\n")


@app.command("inspect-one")
def inspect_one(
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    min_connectors: int = typer.Option(2, "--min-connectors"),
    max_connectors: int = typer.Option(4, "--max-connectors"),
    min_wires: int = typer.Option(3, "--min-wires"),
    max_wires: int = typer.Option(10, "--max-wires"),
    output_format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json or summary",
    ),
) -> None:
    """
    Generate a single sample and print its graph to stdout for debugging.
    """
    config = GraphGeneratorConfig(
        min_connectors=min_connectors,
        max_connectors=max_connectors,
        min_wires=min_wires,
        max_wires=max_wires,
        seed=seed,
    )

    harness = generate_harness_graph(config)

    if output_format == "json":
        # Output full JSON
        output = {
            "id": harness.id,
            "components": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "label": c.label,
                    "pins": [{"id": p.id, "role": p.role.value if p.role else None} for p in c.pins],
                    "attributes": c.attributes,
                }
                for c in harness.components
            ],
            "wires": [
                {
                    "id": w.id,
                    "from": {
                        "component_id": w.from_endpoint.component_id,
                        "pin_id": w.from_endpoint.pin_id,
                    },
                    "to": {
                        "component_id": w.to_endpoint.component_id,
                        "pin_id": w.to_endpoint.pin_id,
                    },
                    "color": w.color,
                    "gauge_mm2": w.gauge_mm2,
                }
                for w in harness.wires
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        # Print summary
        console.print(f"\n[bold]Harness: {harness.id}[/bold]\n")

        # Components table
        comp_table = Table(title="Components")
        comp_table.add_column("ID", style="cyan")
        comp_table.add_column("Type", style="green")
        comp_table.add_column("Label")
        comp_table.add_column("Pins")

        for comp in harness.components:
            pin_count = len(comp.pins)
            pin_str = f"{pin_count} pins" if pin_count > 0 else "-"
            comp_table.add_row(comp.id, comp.type.value, comp.label or "-", pin_str)

        console.print(comp_table)

        # Wires table
        wire_table = Table(title="\nWires")
        wire_table.add_column("ID", style="cyan")
        wire_table.add_column("From", style="yellow")
        wire_table.add_column("To", style="yellow")
        wire_table.add_column("Color")
        wire_table.add_column("Gauge")

        for wire in harness.wires:
            from_str = wire.from_endpoint.component_id
            if wire.from_endpoint.pin_id:
                from_str += f":{wire.from_endpoint.pin_id}"

            to_str = wire.to_endpoint.component_id
            if wire.to_endpoint.pin_id:
                to_str += f":{wire.to_endpoint.pin_id}"

            wire_table.add_row(
                wire.id,
                from_str,
                to_str,
                wire.color or "-",
                f"{wire.gauge_mm2}mm²" if wire.gauge_mm2 else "-",
            )

        console.print(wire_table)

        # Connectivity check
        console.print(
            f"\n[bold]Connected:[/bold] {'Yes' if harness.is_connected() else 'No'}"
        )
        console.print()


@app.command()
def version() -> None:
    """Show version information."""
    from wessley_schematic import __version__

    console.print(f"wessley-schematic version {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
