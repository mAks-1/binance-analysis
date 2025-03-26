import os
import pandas as pd
from zipfile import ZipFile
import asyncio
import aiohttp


class DataLoader:
    """Клас для завантаження та обробки історичних даних з Binance."""

    BASE_URL = "https://data.binance.vision/data/spot/daily/klines"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    async def _download_month(
        self, session: aiohttp.ClientSession, pair: str, year: int, month: int
    ) -> list[str]:
        """Завантаження даних за весь місяць для однієї пари"""
        dates = pd.date_range(
            start=f"{year}-{month:02d}-01",
            end=f"{year}-{month:02d}-28",  # Лютий має 28 днів
            freq="D",
        )

        saved_paths = []
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            url = f"{self.BASE_URL}/{pair}/1m/{pair}-1m-{date_str}.zip"
            local_path = os.path.join(self.data_dir, f"{pair}_{date_str}.parquet")

            if os.path.exists(local_path):
                saved_paths.append(local_path)
                continue

            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        zip_path = local_path.replace(".parquet", ".zip")

                        with open(zip_path, "wb") as f:
                            f.write(content)

                        parquet_path = await self._extract_and_save(
                            zip_path, pair, date_str
                        )
                        saved_paths.append(parquet_path)

            except Exception as e:
                print(f"❌ Помилка для {pair} {date_str}: {str(e)}")

        return saved_paths

    async def download_data(
        self, session: aiohttp.ClientSession, pair: str, date: str
    ) -> str:
        """Асинхронне завантаження даних для однієї пари"""
        url = f"{self.BASE_URL}/{pair}/1m/{pair}-1m-{date}.zip"
        local_path = os.path.join(self.data_dir, f"{pair}_{date}.parquet")

        if os.path.exists(local_path):
            return local_path

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(local_path.replace(".parquet", ".zip"), "wb") as f:
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
            with ZipFile(zip_path, "r") as zip_ref:
                csv_file = zip_ref.namelist()[0]
                zip_ref.extract(csv_file, self.data_dir)

            df = pd.read_csv(
                os.path.join(self.data_dir, csv_file),
                names=["timestamp", "open", "high", "low", "close", "volume"],
                usecols=[0, 1, 2, 3, 4, 5],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
            df.to_parquet(parquet_path, compression="snappy")
            return parquet_path
        except Exception as e:
            print(f"❌ Помилка обробки {pair}: {str(e)}")
            return None

    async def load_month(self, year: int, month: int, top_n: int = 10) -> pd.DataFrame:
        """Завантаження даних за весь місяць для топ-пар"""
        # Генеруємо всі дати місяця
        dates = (
            pd.date_range(
                start=f"{year}-{month:02d}-01",
                end=f"{year}-{month:02d}-28",
                freq="D",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )

        # Отримуємо топ-пари
        top_pairs = await self.get_top_pairs(top_n)

        async with aiohttp.ClientSession() as session:
            # Створюємо завдання для кожної пари та кожного дня
            tasks = [
                self.download_data(session, pair, date)
                for pair in top_pairs
                for date in dates
            ]
            results = await asyncio.gather(*tasks)

        # Об'єднуємо всі дані
        all_dfs = []
        for path in results:
            if path and os.path.exists(path):
                df = pd.read_parquet(path)
                df["pair"] = os.path.basename(path).split("_")[0]

                all_dfs.append(df)

        # Додаткова перевірка після об'єднання
        if all_dfs:
            combined_df = pd.concat(all_dfs).sort_values("timestamp")

            return combined_df

        return pd.DataFrame()

    @staticmethod
    async def get_top_pairs(top_n: int) -> list[str]:
        """Асинхронне отримання списку топ-пар"""
        url = "https://api.binance.com/api/v3/ticker/24hr"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                btc_pairs = [t["symbol"] for t in data if t["symbol"].endswith("BTC")]
                return sorted(
                    btc_pairs,
                    key=lambda x: float(
                        next(t["quoteVolume"] for t in data if t["symbol"] == x)
                    ),
                    reverse=True,
                )[:top_n]
