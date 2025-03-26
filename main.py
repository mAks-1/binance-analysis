import os

from core.data_loader import DataLoader


import asyncio


async def main():
    # Ініціалізація
    loader = DataLoader()

    # Завантаження даних
    print("🔄 Завантаження даних...")
    all_data = await loader.load_top_pairs("2025-02-01", top_n=10)

    if all_data.empty:
        print("❌ Не вдалося завантажити дані")
        return


if __name__ == "__main__":
    asyncio.run(main())