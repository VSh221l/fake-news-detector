import logging
from pathlib import Path
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import Optional, Tuple


log = logging.getLogger(__name__)

DEFAULT_URL = "https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv"
DEFAULT_LOCAL_PATH = Path("data/fake_news.csv")

# === Константы ===
TFIDF_MAX_DF = 0.7
TFIDF_LANGUAGE = "english"
TFIDF_NGRAM_RANGE = (1, 2)

def load_data(
        path_or_url: Optional[str] = DEFAULT_URL, 
        save_to: Optional[str] = DEFAULT_LOCAL_PATH
) -> pd.DataFrame:
    """
    Загружает CSV с локального пути или по URL.
    - Если path_or_url не указан → используется DEFAULT_URL.
    - Файл сохраняется локально (если задан save_to).
    - Проверяет обязательные колонки ['text', 'label'].
    """
    try:
        log.info("Loading dataset from %s", path_or_url)

        df = pd.read_csv(path_or_url)

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

def clean_text_preprocessing(text):
    """
    Универсальная очистка текста:
    - приведение к нижнему регистру
    - удаление пунктуации
    - сохранение букв и цифр (убираем спецсимволы, эмодзи и т.п.)
    """
    try:
        # приводим к нижнему регистру
        text = text.lower()

        # оставляем только буквы и цифры
        tokens = re.findall(r"[a-z0-9]+", text)

        return " ".join(tokens)
    except Exception as e:
        raise RuntimeError(f"Text preprocessing failed: {e}")

def fit_vectorizer(
        texts, 
        max_df: float = TFIDF_MAX_DF,
        stop_words: str = TFIDF_LANGUAGE
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """
    Обучает TfidfVectorizer на texts и возвращает (vectorizer, X_train)
    """
    try:
        vect = TfidfVectorizer(
            max_df=max_df, stop_words=stop_words
        )
        X = vect.fit_transform(texts)
        log.info("Fitted TfidfVectorizer: vocab_size=%d", len(vect.vocabulary_))
        return vect, X
    except Exception as e:
        log.exception("Error in fit_vectorizer: %s", e)
        raise

def transform_vectorizer(vectorizer: TfidfVectorizer, texts) -> csr_matrix:
    """
    Применяет уже обученный vectorizer к новым текстам.
    """
    try:
        X = vectorizer.transform(texts)
        return X
    except Exception as e:
        log.exception("Error in transform_vectorizer: %s", e)
        raise