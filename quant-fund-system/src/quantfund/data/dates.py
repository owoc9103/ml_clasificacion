"""Resolución de fechas para ingest (hasta hoy)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


def resolve_eod_end(value: str | None, *, as_of: date | None = None) -> str:
    """
    Convierte eod_end del YAML a YYYY-MM-DD.

    Valores especiales: today, auto, latest, null → fecha de hoy (UTC).
    yfinance usa `end` exclusivo: devolvemos mañana para incluir el cierre de hoy.
    """
    today = as_of or datetime.now(timezone.utc).date()
    if value is None:
        inclusive_end = today + timedelta(days=1)
        return inclusive_end.isoformat()

    token = str(value).strip().lower()
    if token in {"", "today", "auto", "latest", "now"}:
        inclusive_end = today + timedelta(days=1)
        return inclusive_end.isoformat()

    # Fecha fija YYYY-MM-DD: también +1 día para incluir ese día en yfinance
    try:
        fixed = date.fromisoformat(token[:10])
        return (fixed + timedelta(days=1)).isoformat()
    except ValueError:
        return (today + timedelta(days=1)).isoformat()


def resolve_train_end(value: str | None, *, as_of: date | None = None) -> str:
    """
    Fecha final inclusive para entrenamiento (YYYY-MM-DD).

    Valores especiales: today, auto, latest → hoy (UTC).
    """
    today = as_of or datetime.now(timezone.utc).date()
    if value is None:
        return today.isoformat()

    token = str(value).strip().lower()
    if token in {"", "today", "auto", "latest", "now"}:
        return today.isoformat()

    try:
        return date.fromisoformat(token[:10]).isoformat()
    except ValueError:
        return today.isoformat()


__all__ = ["resolve_eod_end", "resolve_train_end"]
