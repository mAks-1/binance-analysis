import asyncio
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
import vectorbt as vbt

from strategies.base import StrategyBase


class Backtester:
    def __init__(self, strategies: list[StrategyBase]):
        self.strategies = strategies
        self.results_dir = Path("results")
        self.results_screens = Path("results/screenshots")
        self.results_dir.mkdir(exist_ok=True)

    async def _run_strategy(self, strategy: StrategyBase) -> dict[str, Any]:
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

    async def _save_equity_curve(self, backtest_result, strategy_name: str, pair: str):
        """Універсальний метод для збереження графіків"""
        try:
            if isinstance(backtest_result, dict) and 'portfolio' in backtest_result:
                # Обробка результатів MACrossover
                pf = backtest_result['portfolio']
                equity = pf.cumulative_returns()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity.index,
                    y=equity,
                    mode='lines',
                    name='Equity Curve'
                ))

                # Збереження
                plot_path = self.results_dir / f"equity_{strategy_name}_{pair}"
                fig.write_html(f"{plot_path}.html")
                fig.write_image(f"{plot_path}.png")

            elif isinstance(backtest_result, vbt.Portfolio):
                # Обробка інших стратегій
                equity = backtest_result.cumulative_returns()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity.index,
                    y=equity,
                    mode='lines',
                    name='Equity Curve'
                ))

                plot_path = self.results_dir / f"equity_{strategy_name}_{pair}"
                fig.write_html(f"{plot_path}.html")
                fig.write_image(f"{plot_path}.png")

        except Exception as e:
            print(f"❌ Помилка при збереженні графіку для {strategy_name}/{pair}: {str(e)}")

    def _create_heatmap(self, metrics_df: pd.DataFrame, strategy_name: str):
        """Створити теплокарту продуктивності"""
        pivot_df = metrics_df.pivot_table(
            index='pair',
            columns='strategy_name',
            values='total_return'
        )

        plt.figure(figsize=(20, 15))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            linewidths=.5
        )
        plt.title(f"Performance Heatmap - {strategy_name}")

        heatmap_path = self.results_screens / f"heatmap_{strategy_name}.png"
        plt.savefig(heatmap_path)
        plt.close()

    async def run_all(self) -> pd.DataFrame:
        """Асинхронно виконує всі стратегії паралельно"""

        tasks = [self._run_strategy(strategy) for strategy in self.strategies]
        results = await asyncio.gather(*tasks)

        # Після отримання результатів:
        valid_results = [r for r in results if r is not None]
        metrics_df = pd.DataFrame(valid_results)

        # Збереження графіків
        for strategy in self.strategies:
            pf = strategy.run_backtest()
            await self._save_equity_curve(pf, strategy.__class__.__name__, strategy.pair)
            self._create_heatmap(metrics_df, strategy.__class__.__name__)

        # Генерація звіту
        # self._generate_html_report(metrics_df)

        return metrics_df

    async def save_results(self, df: pd.DataFrame, path: str = "results/metrics.csv"):
        """Асинхронне збереження результатів"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: df.to_csv(path, index=False)
        )
        print(f"✅ Результати збережено у {path}")