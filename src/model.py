import logging
from pathlib import Path
from typing import Any, Tuple
import joblib
from sklearn.linear_model import PassiveAggressiveClassifier

log = logging.getLogger(__name__)

# === Пути ===
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "pac_model.joblib"
VECT_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

RANDOM_STATE = 42

def train_model(
        X, y, max_iter: int = 1000, tol: float = 1e-3
) -> PassiveAggressiveClassifier:
    """
    Обучает PassiveAggressiveClassifier и возвращает обученный классификатор.
    """
    try:
        clf = PassiveAggressiveClassifier(
            max_iter=max_iter, tol=tol, random_state=RANDOM_STATE
        )
        clf.fit(X, y)
        log.info("Model trained: classes=%s", clf.classes_)
        return clf
    except Exception as e:
        log.exception("Error training model: %s", e)
        raise

def save_model(
        clf: Any, 
        vect: Any, 
        model_path: str = MODEL_PATH, 
        vect_path: str = VECT_PATH
) -> None:
    """
    Сохраняет модель и vectorizer в папку models/.
    """
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_path)
        joblib.dump(vect, vect_path)
        log.info("Saved model → %s", model_path)
        log.info("Saved vectorizer → %s", vect_path)
    except Exception as e:
        log.exception("Error saving model/vectorizer: %s", e)
        raise

def load_model(
        model_path: str = MODEL_PATH, vect_path: str = VECT_PATH
) -> Tuple[Any, Any]:
    """
    Загружает модель и vectorizer из файлов.
    """
    try:
        clf = joblib.load(model_path)
        vect = joblib.load(vect_path)
        log.info("Loaded model from %s", model_path)
        return clf, vect
    except Exception as e:
        log.exception("Error loading model/vectorizer: %s", e)
        raise