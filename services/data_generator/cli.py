#!/usr/bin/env python
"""
Command-line entry-point for the synthetic data generator.
"""

from pathlib import Path

import click
from app.generator import Generator


@click.command()
@click.option(
    "--n-per-reason",
    "-n",
    default=1,
    show_default=True,
    help="Number of examples to create for each abandonment reason.",
)
@click.option(
    "--outdir",
    "-o",
    default="/opt/output",  # works both local & inside container
    show_default=True,
    help="Directory where JSONL files will be written.",
)
def main(n_per_reason: int, outdir: str):
    out_dir = Path(outdir)
    gen = Generator(n_per_reason=n_per_reason, out_dir=out_dir)
    gen.run()  # per-reason JSONL files
    gen.merge(out_dir / "all_examples.jsonl")
    click.echo(f"✅ Dataset ready under: {out_dir}")


if __name__ == "__main__":
    main()
