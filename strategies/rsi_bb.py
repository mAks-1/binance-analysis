import pandas as pd
import vectorbt as vbt
from strategies.base import StrategyBase


class RSIWithBB(StrategyBase):
    """Комбінована стратегія на основі RSI та Болінджерівських смуг."""

    def __init__(
        self,
        price_data: pd.DataFrame,
        pair: str = None,
        rsi_window: int = 14,
        rsi_overbought: int = 60,
        rsi_oversold: int = 40,
        bb_window: int = 20,
        bb_std: float = 1.5,
    ):
        """Ініціалізує стратегію з параметрами."""
        super().__init__(price_data)
        self.pair = pair
        self.rsi_window = rsi_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_window = bb_window
        self.bb_std = bb_std
        self._pf = None

    def generate_signals(self):
        """Генерує сигнали на основі RSI та Bollinger Bands."""
        close = self.data["close"]
        rsi = vbt.RSI.run(close, self.rsi_window).rsi
        bb = vbt.BBANDS.run(close, self.bb_window, self.bb_std)

        # М'якші умови з додатковими фільтрами
        atr = vbt.ATR.run(
            self.data["high"], self.data["low"], self.data["close"], 14
        ).atr
        atr_filter = atr > atr.rolling(50).mean() * 0.5  # Фільтр низької волатильності

        long_entry = (rsi < 45) & (close <= bb.lower * 1.02) & atr_filter
        short_entry = (rsi > 55) & (close >= bb.upper * 0.98) & atr_filter

        # Вихід при середньому рівні RSI
        long_exit = rsi >= 50
        short_exit = rsi <= 50

        return long_entry, long_exit, short_entry, short_exit

    def run_backtest(self) -> vbt.Portfolio:
        """Виконує бектест стратегії з обмеженням ризиків."""
        long_entry, long_exit, short_entry, short_exit = self.generate_signals()

        self._pf = vbt.Portfolio.from_signals(
            self.data["close"],
            entries=long_entry | short_entry,
            exits=long_exit | short_exit,
            fees=0.0005,
            slippage=0.001,  # Сліппейдж 0.1%
            sl_stop=0.016,  # Stop-Loss 0.8%
            tp_stop=0.016,  # Take-Profit 4%
            freq="1min",  # Таймфрейм
            direction="both",  # Дозволити лонги та шорти
        )
        return self._pf

    def get_metrics(self) -> dict:
        """Розраховує метрики продуктивності стратегії."""
        if not hasattr(self, "_pf") or self._pf is None:
            self._pf = self.run_backtest()

        stats = self._pf.stats()

        return {
            "strategy": "RSI with BB",
            "pair": self.pair,
            "total_return": stats.get("Total Return [%]"),
            "sharpe_ratio": stats.get("Sharpe Ratio"),
            "max_drawdown": stats.get("Max Drawdown [%]"),
            "win_rate": stats.get("Win Rate [%]"),
            "trades": stats.get("Total Trades"),
            "rsi_window": self.rsi_window,
            "bb_window": self.bb_window,
        }
