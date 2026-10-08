import os
import sys
from typing import Optional, cast

import click

from quantfund.data.dates import resolve_eod_end
from quantfund.data.ingest import ingest_symbols, Interval


@click.command()
@click.option("--universe", "universe_path", default="configs/universe.yaml", show_default=True, help="Universe YAML path")
@click.option("--data", "data_path", default="configs/data.yaml", show_default=True, help="Data YAML path")
@click.option("--out", "out_dir", default="data/parquet", show_default=True, help="Output parquet base directory")
@click.option("--intervals", "intervals_override", default=None, help="Comma-separated intervals to override config (e.g., 1d,60m,120m)")
@click.option("--start", "start_override", default=None, help="Override start date (YYYY-MM-DD)")
@click.option("--end", "end_override", default=None, help="Override end date (YYYY-MM-DD)")
def main(universe_path: str, data_path: str, out_dir: str, intervals_override: Optional[str], start_override: Optional[str], end_override: Optional[str]):
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
    start = start_override or data_cfg.get("eod_start")
    end = end_override or resolve_eod_end(data_cfg.get("eod_end"))

    raw_intervals = intervals_override.split(",") if intervals_override else list(data_cfg.get("intervals", []))
    normalized = [s.strip() for s in raw_intervals]
    allowed = {"1d", "5d", "60m", "120m"}
    filtered = [s for s in normalized if s in allowed]
    intervals_typed = cast(list[Interval], filtered)

    os.makedirs(out_dir, exist_ok=True)
    written = ingest_symbols(symbols=symbols, intervals=intervals_typed, start=start, end=end, out_dir=out_dir)

    click.echo(f"Wrote {len(written)} parquet files under {out_dir} (zstd, partitioned by symbol)")


if __name__ == "__main__":
    main()