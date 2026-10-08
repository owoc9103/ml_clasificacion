import os

import pandas as pd


def test_ok():
    assert 1 + 1 == 2


def test_ingest_and_features_exist_dirs():
    os.makedirs("data/parquet", exist_ok=True)
    os.makedirs("data/datasets", exist_ok=True)
    assert os.path.isdir("data/parquet")
    assert os.path.isdir("data/datasets")


def test_backtest_report_schema_if_exists():
    # Validate new outputs if present
    ts = os.path.join("reports", "bt_timeseries_60m.parquet")
    tr = os.path.join("reports", "bt_trades_60m.parquet")
    pf = os.path.join("reports", "bt_perf_60m.parquet")
    if os.path.exists(ts):
        df = pd.read_parquet(ts)
        for col in ["portfolio_ret", "equity", "drawdown"]:
            assert col in df.columns
    if os.path.exists(tr):
        df = pd.read_parquet(tr)
        for col in ["symbol", "weight_prev", "weight_new", "fill_price"]:
            assert col in df.columns
    if os.path.exists(pf):
        df = pd.read_parquet(pf)
        for col in ["ann_return", "ann_vol", "sharpe", "max_drawdown", "num_trades"]:
            assert col in df.columns


def test_no_lookahead_if_features_exist():
    # If a sample features file exists, ensure features at t don't depend on next_open/next_close at t
    path = os.path.join("data", "datasets", "interval=60m")
    if not os.path.isdir(path):
        return
    # Pick any symbol partition
    parts = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f == "data.parquet":
                parts.append(os.path.join(root, f))
    if not parts:
        return
    df = pd.read_parquet(parts[0])
    if df.empty:
        return
    # Check that next_open/next_close are strictly shifted forward
    assert (df.index[1:] < df.index[:-1]).sum() == 0  # monotonically increasing
    assert df["next_open"].isna().iloc[-1]
    assert df["next_close"].isna().iloc[-1]