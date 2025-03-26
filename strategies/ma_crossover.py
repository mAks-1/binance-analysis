import pandas as pd
import vectorbt as vbt
from strategies.base import StrategyBase


class MACrossover(StrategyBase):
    """Стратегія на основі перетину двох ковзних середніх (MA Crossover)."""

    def __init__(
        self,
        price_data: pd.DataFrame,
        pair: str = None,
        short_window: int = 25,
        long_window: int = 100,
    ):
        """Ініціалізує стратегію з параметрами."""
        super().__init__(price_data)
        self.pair = pair
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> pd.DataFrame:
        """Генерує вхідні та вихідні сигнали на основі MA."""
        close = self.data["close"]
        short_ma = vbt.MA.run(close, self.short_window).ma
        long_ma = vbt.MA.run(close, self.long_window).ma

        entries = short_ma.vbt.crossed_above(long_ma)
        exits = short_ma.vbt.crossed_below(long_ma)

        return pd.DataFrame(
            {
                "entries": entries,
                "exits": exits,
                "close": close,
                "short_ma": short_ma,
                "long_ma": long_ma,
            },
            index=self.data.index,
        )

    def run_backtest(self) -> dict:
        """Виконує бектест стратегії."""
        signals = self.generate_signals()

        pf = vbt.Portfolio.from_signals(
            signals["close"],
            entries=signals["entries"],
            exits=signals["exits"],
            fees=0.001,
            slippage=0.005,
            freq="1min",
        )

        return {
            "portfolio": pf,
            "equity_curve": pf.cumulative_returns(),
            "signals": signals,
            "stats": pf.stats(),
        }

    def get_metrics(self) -> dict:
        """Розраховує основні метрики стратегії."""
        backtest_result = self.run_backtest()
        stats = backtest_result["stats"]

        return {
            "strategy": "MA Crossover",
            "pair": self.pair,
            "total_return": stats.loc["Total Return [%]"],
            "sharpe_ratio": stats.loc["Sharpe Ratio"],
            "max_drawdown": stats.loc["Max Drawdown [%]"],
            "win_rate": stats.loc["Win Rate [%]"],
            "trades": stats.loc["Total Trades"],
            "short_window": self.short_window,
            "long_window": self.long_window,
        }
