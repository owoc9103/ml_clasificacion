"""Buy notional cap when DTBP is zero."""

from unittest.mock import patch

from quantfund.execution.alpaca_broker import AlpacaBroker


def test_buy_cap_falls_back_when_dtbp_zero():
    broker = AlpacaBroker(api_key="pk-test", secret_key="sk-test", paper=True)
    acct = {
        "cash": "100000",
        "buying_power": "100000",
        "regt_buying_power": "100000",
        "daytrading_buying_power": "0",
    }
    with patch.object(broker, "get_account", return_value=acct):
        assert broker._get_buy_notional_cap() == 100_000.0
