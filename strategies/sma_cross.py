import pandas as pd
import vectorbt as vbt
from strategies.base import StrategyBase


class SMACrossover(StrategyBase):
    """Стратегія на основі перетину двох простих ковзних середніх (SMA)."""

    def __init__(
        self,
        price_data: pd.DataFrame,
        pair: str = None,
        fast_window: int = 15,
        slow_window: int = 40,
    ):
        """Ініціалізує стратегію з параметрами."""
        super().__init__(price_data)
        self.pair = pair
        self.fast_window = fast_window
        self.slow_window = slow_window
        self._pf = None

    def generate_signals(self) -> tuple:
        """Генерує сигнали входу/виходу на основі перетину SMA."""
        close = self.data["close"]
        fast_ma = vbt.MA.run(close, self.fast_window).ma
        slow_ma = vbt.MA.run(close, self.slow_window).ma

        # Vectorized signal generation
        entries = fast_ma.vbt.crossed_above(slow_ma)
        exits = fast_ma.vbt.crossed_below(slow_ma)

        return entries, exits

    def run_backtest(self) -> vbt.Portfolio:
        """Виконує бектест стратегії."""
        entries, exits = self.generate_signals()

        if self.data["close"].empty:
            return None

        try:
            self._pf = vbt.Portfolio.from_signals(
                self.data["close"],
                entries=entries,
                exits=exits,
                fees=0.001,
                slippage=0.005,
                freq="1min",
            )
            return self._pf
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def get_metrics(self) -> dict:
        """Розраховує метрики продуктивності стратегії."""
        if not hasattr(self, "_pf") or self._pf is None:
            self._pf = self.run_backtest()

        if self._pf is None:
            return {
                "strategy": "SMA Crossover",
                "pair": self.pair,
                "error": "Failed to create portfolio",
            }

        stats = self._pf.stats()

        return {
            "strategy": "SMA Crossover",
            "pair": self.pair,
            "total_return": stats.get("Total Return [%]"),
            "sharpe_ratio": stats.get("Sharpe Ratio"),
            "max_drawdown": stats.get("Max Drawdown [%]"),
            "win_rate": stats.get("Win Rate [%]"),
            "trades": stats.get("Total Trades"),
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
        }
