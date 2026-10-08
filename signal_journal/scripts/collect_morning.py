"""CLI matutino: genera señales y append al CSV (sin Streamlit)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals import build_daily_report, load_symbols
from src.storage import append_snapshot, has_run_for_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardar señales del día en CSV")
    parser.add_argument("--replace", action="store_true", help="Reemplazar la sesión de hoy si ya existe")
    parser.add_argument("--notional", type=float, default=100_000.0)
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    if has_run_for_date(today) and not args.replace:
        print(f"Ya existe sesión {today}. Usa --replace para sobrescribir.")
        return

    df = build_daily_report(load_symbols(), session_date=today, notional_account=args.notional)
    path = append_snapshot(df, replace_session=args.replace or has_run_for_date(today))
    print(f"Guardado: {path} ({len(df)} filas)")


if __name__ == "__main__":
    main()
