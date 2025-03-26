import asyncio
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from typing import Any, List
from strategies.base import StrategyBase


class Backtester:
    """Клас для проведення бектесту торгових стратегій."""

    def __init__(self, strategies: List[StrategyBase]):
        """Клас для проведення бектесту торгових стратегій."""
        self.strategies = strategies
        self.results_dir = Path("results")
        self.results_screens = self.results_dir / "screenshots"
        self.results_dir.mkdir(exist_ok=True)
        self.results_screens.mkdir(exist_ok=True)

    async def _run_strategy(self, strategy: StrategyBase) -> dict[str, Any]:
        """Виконує стратегію та повертає метрики"""
        try:
            metrics = strategy.get_metrics()
            return {
                **metrics,
                "strategy_name": strategy.__class__.__name__,
                "pair": getattr(strategy, "pair", "N/A"),
            }
        except Exception as e:
            print(f"❌ Помилка в стратегії {strategy.__class__.__name__}: {e}")
            return {"strategy_name": strategy.__class__.__name__, "error": str(e)}

    async def _save_equity_curve(self, result: Any, strategy_name: str, pair: str):
        """Універсальне збереження графіків"""
        try:
            if hasattr(result, "cumulative_returns"):
                equity = result.cumulative_returns()
            elif isinstance(result, dict) and "portfolio" in result:
                equity = result["portfolio"].cumulative_returns()
            else:
                return

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=equity.index, y=equity, mode="lines", name="Equity Curve")
            )

            plot_path = self.results_dir / f"equity_{strategy_name}_{pair}"
            fig.write_html(f"{plot_path}.html")
            fig.write_image(f"{plot_path}.png")

        except Exception as e:
            print(f"❌ Помилка при збереженні графіку: {e}")

    def _create_heatmap(self, metrics_df: pd.DataFrame):
        """Створення теплокарти продуктивності"""
        try:
            pivot_df = metrics_df.pivot_table(
                index="pair",
                columns="strategy_name",
                values="total_return",
                aggfunc="mean",
            )

            plt.figure(figsize=(20, 15))
            sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=0.5)
            plt.title("Performance Heatmap")

            heatmap_path = self.results_screens / "heatmap.png"
            plt.savefig(heatmap_path)
            plt.close()
        except Exception as e:
            print(f"❌ Помилка при створенні теплокарти: {e}")

    async def run_all(self) -> pd.DataFrame:
        """Паралельний бектест всіх стратегій"""
        tasks = [self._run_strategy(strategy) for strategy in self.strategies]
        results = await asyncio.gather(*tasks)

        # Обробка результатів
        valid_results = [r for r in results if isinstance(r, dict)]
        metrics_df = pd.DataFrame(valid_results)

        # Паралельне збереження графіків
        save_tasks = [
            self._save_equity_curve(
                strategy.run_backtest(),
                strategy.__class__.__name__,
                getattr(strategy, "pair", "N/A"),
            )
            for strategy in self.strategies
        ]
        await asyncio.gather(*save_tasks)

        # Створення теплокарти
        if not metrics_df.empty:
            self._create_heatmap(metrics_df)

        return metrics_df

    async def save_results(self, df: pd.DataFrame, path: str):
        """Асинхронне збереження результатів"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: df.to_csv(path, index=False))
