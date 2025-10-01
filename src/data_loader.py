import logging
from pathlib import Path
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_URL = "https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv"
DEFAULT_LOCAL_PATH = Path("data/fake_news.csv")

def load_data(
        path_or_url: Optional[str] = None, 
        save_to: Optional[str] = "data/fake_news.csv"
) -> pd.DataFrame:
    """
    Загружает CSV с локального пути или по URL.
    - Если path_or_url не указан → используется DEFAULT_URL.
    - Файл сохраняется локально (если задан save_to).
    - Проверяет обязательные колонки ['text', 'label'].
    """
    try:
        src = path_or_url or DEFAULT_URL
        log.info("Loading dataset from %s", src)

        df = pd.read_csv(src)

        # проверка обязательных колонок
        expected = {"text", "label"}
        if not expected.issubset(set(df.columns)):
            raise ValueError(
                f"Dataset must contain columns: {expected}. Found: {df.columns.tolist()}"
            )
        # dropna по тексту или метке
        df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
        df = df[df["text"].astype(str).str.strip() != ""]
        df = df[df["label"].astype(str).str.strip() != ""]

        if save_to:
            save_to.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_to, index=False)
            log.info("Saved dataset to %s", save_to)

        return df
    
    except Exception as e:
        log.exception("Failed to load dataset: %s", e)
        raise