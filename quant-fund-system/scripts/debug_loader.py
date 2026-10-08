import json
from datetime import date

from quantfund.models.train import (
    load_processed_design_matrix,
    load_train_config,
    timeseries_cv_with_purge_embargo,
)


def main() -> None:
    X, y, meta = load_processed_design_matrix(date(2018, 1, 1), date(2024, 12, 31), "1d")
    print("meta:", json.dumps(meta, default=str))
    if X is None or y is None:
        print("X or y is None")
        return

    feat_cols = [c for c in X.columns if c != "_timestamp"]
    print("rows:", len(X), "nfeat:", len(feat_cols))
    print("first_features:", feat_cols[:10])

    cfg = load_train_config()
    rep, oof = timeseries_cv_with_purge_embargo(X, y, cfg)
    print("cv_error:", rep.get("error"))
    print("fold_count:", len(rep.get("folds", [])))
    if "oof" in rep:
        print("oof_metrics:", json.dumps(rep["oof"], default=str))


if __name__ == "__main__":
    main()


