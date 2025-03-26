import asyncio
import pandas as pd
from typing import List, Dict, Any
from strategies.base import StrategyBase


class Backtester:
    def __init__(self, strategies: List[StrategyBase]):
        self.strategies = strategies

    async def _run_strategy(self, strategy: StrategyBase) -> Dict[str, Any]:
        """Асинхронно виконує одну стратегію та повертає її метрики"""
        try:
            metrics = strategy.get_metrics()

            # Конвертація у словник, якщо необхідно
            if isinstance(metrics, pd.Series):
                metrics = metrics.to_dict()
            elif not isinstance(metrics, dict):
                metrics = {"value": metrics}

            # Додаємо ідентифікаційні поля
            metrics.update({
                "strategy_name": strategy.__class__.__name__,
                "pair": getattr(strategy, "pair", "N/A")
            })

            return metrics

        except Exception as e:
            print(f"❌ Помилка в стратегії {strategy.__class__.__name__}: {str(e)}")
            return {
                "strategy_name": strategy.__class__.__name__,
                "error": str(e)
            }