import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
import asyncio
from core.backtester import Backtester
from strategies.base import StrategyBase


@pytest.fixture
def mock_strategy():
    strategy = MagicMock(spec=StrategyBase)
    strategy.get_metrics.return_value = {
        "total_return": 10.0,
        "sharpe_ratio": 1.5,
        "max_drawdown": 15.0,
        "win_rate": 60.0,
        "trades": 100,
    }
    strategy.__class__.__name__ = "TestStrategy"
    strategy.pair = "TESTBTC"
    return strategy


@pytest.fixture
def backtester(mock_strategy):
    return Backtester([mock_strategy])


@pytest.mark.asyncio
async def test_run_strategy(backtester, mock_strategy):
    result = await backtester._run_strategy(mock_strategy)

    assert isinstance(result, dict)
    assert result["total_return"] == 10.0
    assert result["strategy_name"] == "TestStrategy"
    mock_strategy.get_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_run_all(backtester, mock_strategy):
    with patch.object(
        backtester, "_save_equity_curve", new_callable=AsyncMock
    ) as mock_save:
        results = await backtester.run_all()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        mock_save.assert_called_once()


def test_create_heatmap(backtester):
    test_data = pd.DataFrame(
        {
            "pair": ["TESTBTC", "TESTBTC"],
            "strategy_name": ["Strategy1", "Strategy2"],
            "total_return": [10.0, -5.0],
        }
    )

    with patch("matplotlib.pyplot.show"):
        backtester._create_heatmap(test_data)

    # Якщо не виникло виключень - тест пройдений


@pytest.mark.asyncio
async def test_save_results(backtester):
    test_df = pd.DataFrame({"test": [1, 2, 3]})
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        await backtester.save_results(test_df, "test.csv")
        mock_to_csv.assert_called_once_with("test.csv", index=False)
