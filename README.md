# Multimodal Calorizer

Мультимодальная модель для предсказания калорийности блюд по:
- изображению еды,
- списку ингредиентов,
- массе блюда.

Проект построен на PyTorch с использованием:
- EfficientNet для изображений,
- DistilBERT для текста,
- regression head для предсказания калорий.

## Используемые технологии
- Python
- PyTorch
- Transformers
- timm
- Pandas
- NumPy

## Файлы проекта
- `dataset_calorizer.py` — подготовка данных
- `utils_calorizer.py` — модель и train pipeline
- `Sprint_4_notebook.ipynb` — обучение и эксперименты
