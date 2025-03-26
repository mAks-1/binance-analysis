import os
import pandas as pd
from zipfile import ZipFile
import asyncio
import aiohttp


class DataLoader:
    BASE_URL = "https://data.binance.vision/data/spot/daily/klines"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    async def download_data(self, session: aiohttp.ClientSession, pair: str, date: str) -> str:
        pass

    async def load_top_pairs(self, date: str, top_n: int = 10) -> pd.DataFrame:
        """Асинхронне завантаження топ-пар"""
        top_pairs = await self.get_top_pairs(top_n)

    async def get_top_pairs(self, top_n: int) -> list[str]:
        """Асинхронне отримання списку топ-пар"""
        url = "https://api.binance.com/api/v3/ticker/24hr"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                btc_pairs = [t['symbol'] for t in data if t['symbol'].endswith('BTC')]
                return sorted(btc_pairs, key=lambda x: float(next(
                    t['quoteVolume'] for t in data if t['symbol'] == x)), reverse=True)[:top_n]


# Приклад використання
if __name__ == "__main__":
    loader = DataLoader()
    loader.load_top_pairs("2025-02-01")