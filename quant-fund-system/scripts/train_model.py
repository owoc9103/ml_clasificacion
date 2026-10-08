# scripts/train_model.py
import json
import click
from datetime import datetime
from quantfund.models.train import train_main

@click.command()
@click.option("--start", required=True, type=str, help="YYYY-MM-DD")
@click.option("--end", required=True, type=str, help="YYYY-MM-DD")
@click.option("--wf-quarters", default=4, type=int, show_default=True)
@click.option("--interval", default="1d", type=click.Choice(["1d","5d","60m","120m"]), show_default=True)
@click.option("--label-type", default="binary", type=click.Choice(["binary", "vol_regime", "trend_strength", "sharpe_regime"]), show_default=True, help="Type of prediction target")
@click.option("--lookforward", default=10, type=int, show_default=True, help="Days to look forward for regime labels")
@click.option(
    "--model",
    "model_name",
    default="xgboost",
    type=click.Choice(
        [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "svm",
            "knn",
            "naive_bayes",
        ]
    ),
    show_default=True,
)
def main(start, end, wf_quarters, interval, label_type, lookforward, model_name):
    res = train_main(
        start=datetime.fromisoformat(start).date(),
        end=datetime.fromisoformat(end).date(),
        wf_quarters=wf_quarters,
        interval=interval,
        label_type=label_type,
        lookforward=lookforward,
        model_name=model_name,
    )
    if not res:
        print(json.dumps({
            "error": "No training result",
            "hint": "Likely no rows after date filter/labeling/CV.",
            "checks": [
                f"Verify data/datasets/interval={interval}/symbol=*/data.parquet exists",
                "Open 1 parquet and confirm next_open/next_close not all NaN",
                "Start with interval=1d; intraday has ~730d limit"
            ]
        }, indent=2)); return
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
