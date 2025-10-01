import logging
from pathlib import Path
from typing import Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

log = logging.getLogger(__name__)

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CONF_MATRIX_PATH = DOCS_DIR / "confusion_matrix.png"

def evaluate_and_report(
        clf: Any, 
        X_test, y_test, 
        labels: List[str] = None, 
        save_conf_path: Path = CONF_MATRIX_PATH
) -> dict:
    """
    Считает метрики, печатает отчет и сохраняет confusion matrix.
    Возвращает dict с метриками.
    """
    try:
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=labels, zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=labels)

        log.info("Accuracy: %.4f", acc)
        log.info("Classification report:\n%s", report)

        # Визуализация матрицы ошибок
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            xticklabels=labels, 
            yticklabels=labels
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_conf_path, dpi=200)
        plt.close()
        log.info("Saved confusion matrix to %s", save_conf_path)

        return {"accuracy": acc, "report": report, "confusion_matrix": cm}
    except Exception as e:
        log.exception("Error in evaluation: %s", e)
        raise