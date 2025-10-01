import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import Tuple

log = logging.getLogger(__name__)

# === Константы ===
TFIDF_MAX_DF = 0.7
TFIDF_LANGUAGE = "english"
TFIDF_NGRAM_RANGE = (1, 2)

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