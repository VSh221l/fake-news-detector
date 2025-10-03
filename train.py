import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data import load_data, clean_text_preprocessing, fit_vectorizer, transform_vectorizer
from src.model import train_model, save_model
from src.evaluate import evaluate_and_report

# === Константы проекта ===
DATA_LOCAL_PATH = Path("data/fake_news.csv")
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

def main(data_path: str = None):
    """
    Основной pipeline обучения модели:
      1. Загрузка данных
      2. Предобработка текста
      3. Разделение на train/test
      4. Векторизация
      5. Обучение модели
      6. Сохранение модели и vectorizer
      7. Оценка качества и сохранение confusion matrix
      """
    try:
        # 1) Загрузка данных
        df = load_data(path_or_url=data_path, save_to=DATA_LOCAL_PATH)

        # 2) Предобработка текста
        df["text"] = df["text"].astype(str).apply(clean_text_preprocessing)

        # 3) Разделение
        x_raw = df["text"].astype(str).values
        y = df["label"].astype(str).values
        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            x_raw, 
            y, 
            test_size=DEFAULT_TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=y
        )
        log.info("Train/test split: %d / %d", len(x_train_raw), len(x_test_raw))

        # 4) Векторизация
        vect, X_train = fit_vectorizer(x_train_raw)
        X_test = transform_vectorizer(vect, x_test_raw)

        # 5) Обучение модели
        clf = train_model(X_train, y_train)

        # 6) Сохранение модели и vectorizer
        save_model(clf, vect)

        # 7) Оценка и сохранение confusion matrix
        labels = sorted(set(y))  # порядок меток
        metrics = evaluate_and_report(clf, X_test, y_test, labels=labels)

        log.info("Training completed. Accuracy: %.4f", metrics["accuracy"])

    except Exception as e:
        log.exception("Training pipeline failed: %s", e)

if __name__ == "__main__":
    main()