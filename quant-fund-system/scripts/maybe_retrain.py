"""Reentrena modelos si pasó el intervalo configurado (mismo flujo que train_models.py)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import click
import yaml

from quantfund.data.dates import resolve_train_end
from quantfund.models.multi_train import train_all_models
from quantfund.models.registry import MODEL_NAMES

STAMP_PATH = Path("reports/last_retrain.json")


def _load_train_cfg() -> dict:
    path = Path("configs/train.yaml")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _last_retrain_date() -> date | None:
    if STAMP_PATH.exists():
        try:
            data = json.loads(STAMP_PATH.read_text(encoding="utf-8"))
            return date.fromisoformat(str(data["trained_on"])[:10])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    live = Path("artifacts/1d/models/xgboost/model_all.joblib")
    if live.exists():
        ts = datetime.fromtimestamp(live.stat().st_mtime, tz=timezone.utc).date()
        return ts
    legacy = Path("artifacts/1d/model_all.joblib")
    if legacy.exists():
        ts = datetime.fromtimestamp(legacy.stat().st_mtime, tz=timezone.utc).date()
        return ts
    return None


def _write_stamp(end: str) -> None:
    STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAMP_PATH.write_text(
        json.dumps({"trained_on": end, "utc": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


@click.command()
@click.option("--force", is_flag=True, help="Entrenar aunque no haya pasado el intervalo.")
@click.option("--interval", default="1d", type=click.Choice(["1d", "5d", "60m", "120m"]), show_default=True)
def main(force: bool, interval: str) -> None:
    cfg = _load_train_cfg()
    every = int(cfg.get("retrain_every_days") or 7)
    start_s = str(cfg.get("train_start") or "2018-01-01")
    end_s = resolve_train_end(cfg.get("train_end") or "today")
    model_list = list(cfg.get("models") or MODEL_NAMES)

    today = datetime.now(timezone.utc).date()
    last = _last_retrain_date()
    if not force and last is not None:
        age = (today - last).days
        if age < every:
            click.echo(
                f"Reentrenamiento omitido: último={last.isoformat()}, "
                f"antigüedad={age}d < retrain_every_days={every}. Use --force."
            )
            return

    click.echo(f"Entrenando {len(model_list)} modelos: {start_s} → {end_s} ({interval})")
    res = train_all_models(
        start=date.fromisoformat(start_s[:10]),
        end=date.fromisoformat(end_s[:10]),
        interval=interval,
        models=model_list,
    )
    _write_stamp(end_s)
    click.echo(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
