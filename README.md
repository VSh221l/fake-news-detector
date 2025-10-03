# Fake News Detector

Классический ML-проект для обнаружения фальшивых новостей.  
Модель обучена на датасете [fake_news.csv](https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv),  
использует **TF-IDF** для извлечения признаков и **PassiveAggressiveClassifier** для классификации.  
Точность достигает **90%+**.

## 📂 Структура проекта
```
fake-news-detector/
│── data/ # Датасеты
│ └── fake_news.csv
│── notebooks/ # Jupyter-ноутбуки с исследованиями
│── src/ # Исходный код проекта
│ ├── data.py # Загрузка и подготовка данных
│ ├── evaluate.py # Метрики и визуализации
│ └── model.py   # Загрузка/сохранение модели
├── train.py                 # точка входа для обучения
│── requirements.txt # Зависимости проекта
└──  README.md # Описание проекта
```
## 🚀 Установка и запуск

1. Клонировать репозиторий:
```bash
git clone https://github.com/username/fake-news-detector.git
cd fake-news-detector
```
2. Установить зависимости:

```bash
pip install -r requirements.txt
```
3. Запустить обучение модели:

```bash
python src/train.py
```
4. Оценить результаты:

```bash
python src/evaluate.py
```

## 📊 Методы
* TfidfVectorizer — преобразование текста в вектор признаков

* PassiveAggressiveClassifier — алгоритм классификации

* Confusion Matrix, Accuracy — метрики качества

## 📈 Визуализации
В проекте реализованы:
- Матрица ошибок (confusion matrix)

- Диаграммы распределения классов

- Графики точности и полноты

# 🛠 Используемые технологии
* Python 3.11
* scikit-learn
* nlkt
* matplotlib
* pandas
* numpy
* jupyter

## 🎯 Результаты
* Классификация новостей на REAL / FAKE
* Точность: >90%
* Готовое решение для проверки достоверности новостей
---
📌 Проект разработан как часть учебной практики и может использоваться в портфолио.
