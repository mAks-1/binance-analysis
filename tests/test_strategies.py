import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from strategies.sma_cross import SMACrossover
from strategies.ma_crossover import MACrossover
from strategies.rsi_bb import RSIWithBB


@pytest.fixture
def sample_price_data():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = np.linspace(100, 200, len(dates))
    return pd.DataFrame(
        {
            "close": prices,
            "open": prices - 5,
            "high": prices + 5,
            "low": prices - 10,
            "volume": np.random.randint(1000, 5000, len(dates)),
        },
        index=dates,
    )


def test_sma_crossover_generate_signals(sample_price_data):
    strategy = SMACrossover(sample_price_data, "TESTBTC", 10, 20)
    signals = strategy.generate_signals()

    assert isinstance(signals, tuple)
    assert len(signals) == 2  # entries, exits
    assert len(signals[0]) == len(sample_price_data)


def test_ma_crossover_backtest(sample_price_data):
    strategy = MACrossover(sample_price_data, "TESTBTC", 10, 20)
    result = strategy.run_backtest()

    assert isinstance(result, dict)
    assert "portfolio" in result
    assert "stats" in result
    assert isinstance(result["stats"], pd.Series)


def test_rsi_with_bb_metrics(sample_price_data):
    strategy = RSIWithBB(sample_price_data, "TESTBTC")
    metrics = strategy.get_metrics()

    expected_keys = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
    assert all(key in metrics for key in expected_keys)
    assert metrics["strategy"] == "RSI with BB"


@pytest.mark.parametrize("window,expected", [(14, 14), (20, 20)])
def test_strategy_parameters(sample_price_data, window, expected):
    # Тестування різних параметрів для стратегій
    strategy = RSIWithBB(sample_price_data, "TESTBTC", rsi_window=window)
    assert strategy.rsi_window == expected


def test_sma_crossover_edge_cases():
    # Тестування з пустими даними
    empty_data = pd.DataFrame(columns=["close", "open", "high", "low", "volume"])
    strategy = SMACrossover(empty_data, "TESTBTC")

    with pytest.raises(Exception):
        strategy.run_backtest()
