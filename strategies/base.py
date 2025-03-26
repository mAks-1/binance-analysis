from abc import ABC, abstractmethod
import pandas as pd

class StrategyBase(ABC):
    def __init__(self, price_data: pd.DataFrame):
        self.data = price_data

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Генерация торговых сигналов."""
        pass

    @abstractmethod
    def run_backtest(self) -> pd.DataFrame:
        """Запуск бектеста."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """Расчет метрик."""
        pass