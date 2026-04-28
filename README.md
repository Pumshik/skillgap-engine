# SkillGap Engine

## О проекте

**SkillGap Engine** - это полнофункциональный ML-пайплайн, разработанный для автоматизированного анализа разрыва между текущими навыками соискателя и требованиями рынка труда. Проект решает задачу ранжирования рекомендаций по обучению, оценивает уверенность предсказаний с помощью статистических методов и предоставляет готовый к продакшену REST API для интеграции в HR-платформы, системы карьерного консультирования или Learning Management Systems.

### Ключевые особенности:
-  **Две модели**: кастомная логистическая регрессия (SGD) + PyTorch MLP
-  **Production-ready**: FastAPI API, Docker, TensorBoard, валидация входа
-  **Инженерная культура**: конфиги, логирование, тесты, воспроизводимость
-  **Статистическая валидация**: бутстрап 95% CI, сравнение моделей

---

### Структура проекта:
```
skillgap-engine/
├── configs/
│   └── default.yaml          # Конфигурация эксперимента
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── loader.py         # Загрузка и препроцессинг данных
│   ├── features/
│   │   └── preprocessing.py  # Пайплайны обработки признаков
│   ├── models/
│   │   ├── classical.py      # Кастомная логистическая регрессия
│   │   └── pytorch_net.py    # MLP на PyTorch
│   ├── train.py              # Скрипт обучения классической модели
│   ├── train_torch.py        # Скрипт обучения PyTorch модели
│   └── utils/
│       └── logger.py         # Настройка логирования
├── tests/
│   ├── test_loader.py        # Тесты для загрузчика данных
│   └── test_model.py         # Тесты для моделей
├── artifacts/                # Сохранённые модели и метрики
├── logs/                     # Логи обучения
├── runs/                     # Логи TensorBoard
├── notebooks/                # Jupyter ноутбуки для экспериментов
├── app.py                    # FastAPI inference сервер
├── Dockerfile                # Docker образ для деплоя
├── docker-compose.yml        # Оркестрация контейнеров
├── requirements.txt          # Зависимости для обучения
├── requirements_cpu.txt      # Зависимости для CPU-деплоя
├── .dockerignore             # Исключения для Docker
├── .gitignore                # Исключения для Git
└── README.md                 # Этот файл
```

---

## Быстрый старт

### Локальный запуск (без Docker)

```bash
# 1. Клонировать репозиторий
git clone <repository-url>
cd skillgap-engine

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Обучить классическую модель
python src/train.py

# 5. Обучить PyTorch модель
python src/train_torch.py

# 6. Запустить API сервер
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 7. Проверить поднятие сервиса
curl http://localhost:8000/health

# 8. Сделать прогноз
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, 0.8, ...]}'
```

### Запуск через Docker (рекомендуется)

```bash
# 1. Собрать и запустить контейнер
docker-compose up --build -d

# 2. Проверить статус
docker-compose ps

# 3. Проверить здоровье
curl http://localhost:8000/health

# 4. Открыть Swagger UI для тестирования API
# Перейдите в браузере: http://localhost:8000/docs

# 5. Остановить сервис
docker-compose down
```

---

## Модели и метрики

### Классическая модель: `MyLogisticRegression`
- **Реализация**: с нуля на NumPy, наследует `sklearn.base.BaseEstimator`
- **Оптимизация**: SGD с мини-батчами, early stopping, clipping градиентов
- **Функция потерь**: Binary Cross-Entropy с numerical stability

### Нейросетевая модель: `MLPClassifier`
- **Архитектура**: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Sigmoid
- **Фреймворк**: PyTorch с поддержкой GPU/CPU auto-detect
- **Оптимизация**: Adam + ReduceLROnPlateau scheduler + gradient clipping

### Сравнение на демо-датасете (breast_cancer):

| Модель | Test Accuracy | Test F1 | Val Loss | Params |
|--------|--------------|---------|----------|--------|
| Custom Logistic Regression (SGD) | 0.965 | 0.972 | 0.094 | ~200 |
| PyTorch MLP (3-layer) | 0.971 | 0.978 | 0.021 | ~13K |

---

## Конфигурация

Все параметры вынесены в `configs/default.yaml` для воспроизводимости:

```yaml
data:
  seed: 42
  test_size: 0.15
  val_size: 0.15

model:
  lr: 0.01
  max_iter: 1000
  tol: 1e-4
  batch_size: 64
  method: "sgd"

paths:
  model: "artifacts/model.joblib"
  scaler: "artifacts/scaler.joblib"
  metrics: "artifacts/metrics.json"

training:
  device: "cpu"
  epochs: 50
  early_stopping_patience: 10
  log_dir: "artifacts/logs"
  artifact_dir: "artifacts/models"
```

---

## Визуализация и мониторинг

### TensorBoard
```bash
tensorboard --logdir=runs --host 0.0.0.0 --port 6006
# Открыть: http://localhost:6006
```

**Отслеживаемые метрики:**
- `Loss/train`, `Loss/val` — динамика функции потерь
- `Metrics/val_accuracy`, `Metrics/val_f1` — качество на валидации

### Логирование
- Консоль (INFO level)
- Файл `logs/train.log` / `logs/torch_train.log` (DEBUG level)

---

## Тестирование

```bash
# Запустить все тесты
pytest tests/ -v

# С отчётом о покрытии
pytest tests/ -v --cov=src --cov-report=html
```

---

## API Endpoints

### `GET /health`
```json
{"status": "healthy", "models_loaded": true}
```

### `POST /predict`
**Запрос:**
```json
{"features": [0.5, 0.3, 0.8, 0.2, ...]}
```

**Ответ:**
```json
{"prediction": 1, "probability": 0.8734, "status": "success"}
```

**Swagger UI:** http://localhost:8000/docs

---

## Docker

### Сборка образа вручную:
```bash
docker build -t skillgap-engine:latest .
```

### Запуск контейнера:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/configs:/app/configs \
  skillgap-engine:latest
```

---

## Воспроизводимость

1. **Фиксированные seed**: `random_state=42` везде
2. **Конфиг-управление**: все гиперпараметры в `configs/default.yaml`
3. **Версионирование**: `requirements.txt` с точными версиями
4. **Артефакты**: модели, скалеры, метрики сохраняются с уникальными путями
5. **Логирование**: полный трейн каждого эксперимента в `logs/`

---
