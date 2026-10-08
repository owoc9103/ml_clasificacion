from datetime import date

from quantfund.data.dates import resolve_train_end


def test_resolve_train_end_today():
    assert resolve_train_end("today", as_of=date(2026, 10, 8)) == "2026-10-08"


def test_resolve_train_end_fixed():
    assert resolve_train_end("2020-06-15", as_of=date(2026, 1, 1)) == "2020-06-15"
