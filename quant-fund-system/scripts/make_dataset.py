import os
import sys
from typing import Optional, cast

import click

from quantfund.features.core import build_features_dataset, Interval


@click.command()
@click.option("--universe", "universe_path", default="configs/universe.yaml", show_default=True)
@click.option("--data", "data_path", default="configs/data.yaml", show_default=True)
@click.option("--parquet", "parquet_dir", default="data/parquet", show_default=True)
@click.option("--out", "out_dir", default="data/datasets", show_default=True)
@click.option("--intervals", "intervals_override", default=None, help="Override, comma-separated, e.g., 60m,120m")
def main(universe_path: str, data_path: str, parquet_dir: str, out_dir: str, intervals_override: Optional[str]):
    try:
        import yaml  # type: ignore
    except Exception:
        click.echo("pyyaml is required. Please add it to dependencies and install.", err=True)
        sys.exit(2)

    with open(universe_path, "r", encoding="utf-8") as f:
        uni = yaml.safe_load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    symbols = list(uni.get("symbols", []))
    raw_intervals = intervals_override.split(",") if intervals_override else list(data_cfg.get("intervals", []))
    normalized = [s.strip() for s in raw_intervals]
    allowed = {"1d", "60m", "120m"}
    filtered = [s for s in normalized if s in allowed]
    intervals_typed = cast(list[Interval], filtered)

    os.makedirs(out_dir, exist_ok=True)
    written = build_features_dataset(symbols=symbols, intervals=intervals_typed, parquet_dir=parquet_dir, out_dir=out_dir)
    click.echo(f"Wrote features: {written}")


if __name__ == "__main__":
    main()