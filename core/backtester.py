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

    async def run_all(self) -> pd.DataFrame:
        """Асинхронно виконує всі стратегії паралельно"""
        tasks = [self._run_strategy(strategy) for strategy in self.strategies]
        results = await asyncio.gather(*tasks)

        # Фільтруємо пусті результати
        valid_results = [r for r in results if r is not None]
        return pd.DataFrame(valid_results)

    async def save_results(self, df: pd.DataFrame, path: str = "results/metrics.csv"):
        """Асинхронне збереження результатів"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: df.to_csv(path, index=False)
        )
        print(f"✅ Результати збережено у {path}")