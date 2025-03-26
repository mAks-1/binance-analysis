import os

from core.data_loader import DataLoader
from strategies.ma_crossover import MACrossover
from strategies.rsi_bb import RSIWithBB
from strategies.sma_cross import SMACrossover
from core.backtester import Backtester

import asyncio

PAIRS_TO_GET = 100


async def main():
    # Ініціалізація
    loader = DataLoader()
    backtester = Backtester([])

    # Завантаження даних
    print("🔄 Завантаження даних...")
    all_data = await loader.load_month(2025, 2, PAIRS_TO_GET)

    if all_data.empty:
        print("❌ Не вдалося завантажити дані")
        return

    # Створення стратегій
    strategies = []
    for pair, data in all_data.groupby("pair"):
        strategies.extend(
            [
                SMACrossover(data, pair=pair),
                RSIWithBB(data, pair=pair),
                MACrossover(data, pair=pair),
            ]
        )

    # Паралельний бектест
    print("🚀 Запуск бектестів...")
    backtester.strategies = strategies
    results = await backtester.run_all()

    # Збереження результатів
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/metrics.csv", index=False)
    print("✅ Результати збережено у results/metrics.csv")


if __name__ == "__main__":
    asyncio.run(main())
