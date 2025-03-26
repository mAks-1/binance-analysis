from abc import ABC, abstractmethod
import pandas as pd

class StrategyBase(ABC):
    def __init__(self, price_data: pd.DataFrame):
        self.data = price_data

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Генерація торгових сигналів."""
        pass

    @abstractmethod
    def run_backtest(self) -> pd.DataFrame:
        """Запуск бектесту."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """Розрахунок метрик."""
        pass