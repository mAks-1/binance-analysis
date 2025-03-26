import os
import pandas as pd
from zipfile import ZipFile
import asyncio
import aiohttp

PAIRS_TO_GET = 100

class DataLoader:
    BASE_URL = "https://data.binance.vision/data/spot/daily/klines"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    async def download_data(self, session: aiohttp.ClientSession, pair: str, date: str) -> str:
        """Асинхронне завантаження даних для однієї пари"""
        url = f"{self.BASE_URL}/{pair}/1m/{pair}-1m-{date}.zip"
        local_path = os.path.join(self.data_dir, f"{pair}_{date}.parquet")

        if os.path.exists(local_path):
            return local_path

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(local_path.replace('.parquet', '.zip'), 'wb') as f:
                        f.write(content)
                    print(f"✅ {url} завантажено.")  # for debugging
                    print(f"✅ Збережено до {local_path}.")  # for debugging
                    return await self.extract_and_save(pair, date)
                print(f"❌ Помилка завантаження {pair}: {response.status}")
        except Exception as e:
            print(f"❌ Помилка {pair}: {str(e)}")
        return None

    async def extract_and_save(self, pair: str, date: str) -> str:
        """Розпакування та збереження даних"""
        zip_path = os.path.join(self.data_dir, f"{pair}_{date}.zip")
        parquet_path = os.path.join(self.data_dir, f"{pair}_{date}.parquet")

        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                csv_file = zip_ref.namelist()[0]
                zip_ref.extract(csv_file, self.data_dir)

            df = pd.read_csv(os.path.join(self.data_dir, csv_file),
                             names=["timestamp", "open", "high", "low", "close", "volume"],
                             usecols=[0, 1, 2, 3, 4, 5])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            df.to_parquet(parquet_path, compression='snappy')
            return parquet_path
        except Exception as e:
            print(f"❌ Помилка обробки {pair}: {str(e)}")
            return None

    async def load_top_pairs(self, date: str, top_n: int = 10) -> pd.DataFrame:
        """Асинхронне завантаження топ-пар"""
        top_pairs = await self.get_top_pairs(top_n)

        async with aiohttp.ClientSession() as session:
            tasks = [self.download_data(session, pair, date) for pair in top_pairs]
            results = await asyncio.gather(*tasks)

        # Об'єднання даних
        all_dfs = []
        for path in results:
            if path and os.path.exists(path):
                df = pd.read_parquet(path)
                df['pair'] = os.path.basename(path).split('_')[0]
                all_dfs.append(df)

        return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

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
    loader.load_top_pairs("2025-02-01", PAIRS_TO_GET)