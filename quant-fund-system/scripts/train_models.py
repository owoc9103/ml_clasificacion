"""Entrena y compara todos los modelos supervisados configurados."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import yaml

from quantfund.models.multi_train import train_all_models
from quantfund.models.registry import MODEL_NAMES


@click.command()
@click.option("--start", required=True, type=str, help="YYYY-MM-DD")
@click.option("--end", required=True, type=str, help="YYYY-MM-DD")
@click.option("--interval", default="1d", type=click.Choice(["1d", "5d", "60m", "120m"]), show_default=True)
@click.option(
    "--models",
    default=None,
    help="Lista separada por comas. Por defecto usa configs/train.yaml",
)
def main(start: str, end: str, interval: str, models: str | None) -> None:
    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    else:
        cfg_path = Path("configs/train.yaml")
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
            model_list = list(cfg.get("models") or MODEL_NAMES)

    res = train_all_models(
        start=datetime.fromisoformat(start).date(),
        end=datetime.fromisoformat(end).date(),
        interval=interval,
        models=model_list,
    )
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
