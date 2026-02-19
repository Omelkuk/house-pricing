import pandas as pd
import os


class DataLoader:
    @staticmethod
    def load_csv(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Файл не найден по пути: {os.path.abspath(path)}")

        print(f" Загрузка данных из: {path}")
        return pd.read_csv(path)

    @staticmethod
    def save_results(df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f" Файл сохранен: {path}")