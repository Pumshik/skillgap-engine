
# SkillGap Engine

## О проекте

**SkillGap Engine** - production-ready ML-пайплайн для автоматизированного анализа соответствия резюме соискателя требованиям вакансии. Проект решает задачу бинарной классификации "подходит/не подходит", оценивает уверенность предсказаний с помощью калибровки вероятностей и предоставляет REST API с поддержкой ансамблевого голосования моделей.

### Ключевые особенности:
- **Датасет**: модели обучались на датасете https://huggingface.co/datasets/batuhanmtl/job_resume_fit/blob/main/job_resume_fit.csv, содержащем описания вакансий, резюме, требуемые навыки и имеющиеся навыки, а также оценки их сходство, на основе которого я и определял целевую переменную.
- **Ансамблевое предсказание**: Комбинация кастомной логистической регрессии и PyTorch MLP для максимальной стабильности и устойчивости к аномалиям.
- **Бизнес-ориентированные признаки**: TF-IDF для текстов + 4 строгих числовых метрики пересечения навыков.
- **Калиброванные вероятности**: Использование Platt Scaling для приведения сырых логитов нейросети к интерпретируемым вероятностям.
- **Инженерная культура**: Единый `requirements.txt`, конфигурация через YAML, логирование, Docker-контейнеризация и защита от аномальных входных данных.

---

## Архитектура

┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────┐
│   Data Loader   │────▶│  Feature Engineering     │────▶│   Model Train   │
│  (CSV / Pandas) │     │ (TF-IDF + Skill Metrics) │     │ (LR + PyTorch)  │
└─────────────────┘     └──────────────────────────┘     └────────┬────────┘
                                                                  │
┌─────────────────┐     ┌──────────────────────────┐              │
│   FastAPI App   │◀────│  Ensemble Inference      │◀─────────────┘
│  /predict + /   │     │  (Linear + MLP Average)  │
│  /health        │     │  + Platt Calibration     │
└────────┬────────┘     └──────────────────────────┘
         │
┌─────────────────┐
│   Docker Image  │
│  (uvicorn/FastAPI)
└─────────────────┘

### Структура проекта:

skillgap-engine/
 ├── configs/
 │   └── default.yaml          # Конфигурация гиперпараметров и путей
 ├── src/
 │   ├── data/
 │   │   └── loader.py         # Загрузка, парсинг навыков и препроцессинг
 │   ├── models/
 │   │   ├── classical.py      # Кастомная логистическая регрессия
 │   │   └── pytorch_net.py    # MLP на PyTorch (Linear -> BatchNorm -> ReLU -> Dropout)
 │   ├── train.py              # Скрипт обучения классической модели
 │   └── train_torch.py        # Скрипт обучения PyTorch модели + калибратор
 ├── tests/                    # Юнит-тесты пайплайна
 ├── artifacts/                # Сохранённые модели (.joblib, .pth, .json)
 ├── logs/                     # Логи обучения
 ├── runs/                     # Логи TensorBoard
 ├── app.py                    # FastAPI inference сервер
 ├── Dockerfile                # Docker образ для деплоя
 ├── docker-compose.yml        # Оркестрация контейнеров
 ├── requirements.txt          # Зависимости
 ├── .dockerignore             # Исключения для Docker
 ├── .gitignore                # Исключения для Git
 └── README.md                 # Этот файл

---

## Быстрый старт

### Локальный запуск (Разработка)

# 1. Клонировать репозиторий
git clone <https://github.com/Pumshik/skillgap-engine>
cd skillgap-engine

# 2. Создать и активировать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate (Windows)

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Обучить обе модели (создаст артефакты в папке artifacts/)
python src/train.py
python src/train_torch.py

# 5. Запустить API сервер
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 6. Проверить здоровье сервиса
curl http://localhost:8000/health

### Запуск через Docker

> **Важно:** Убедитесь, что папка `artifacts/` уже содержит обученные модели перед запуском контейнера, так как они монтируются как volume.

# 1. Собрать и запустить контейнер
docker-compose up --build -d

# 2. Проверить статус и логи
docker-compose ps
docker-compose logs -f skillgap-api

# 3. Остановить сервис
docker-compose down

---

## Модели и данные

### Инженерия признаков
1. **Текстовые**: Независимый `TfidfVectorizer` для текста резюме и текста вакансии.
2. **Числовые**: 
   - `resume_skill_count`, `required_skill_count`
   - `matched_skill_count` (пересечение множеств навыков)
   - `skill_match_ratio` (доля совпавших навыков от требуемых)


### Метрики (на валидационной выборке)

| Модель                           | Val Accuracy | Val F1-Score | Особенности                                                        |
|----------------------------------|--------------|--------------|--------------------------------------------------------------------|
| Custom Logistic Regression       | ~0.85 - 0.87 | ~0.86 - 0.88 | Быстрая, интерпретируемая, устойчивая к линейным границам          |
| PyTorch MLP (Calibrated)         | ~0.86 - 0.88 | ~0.87 - 0.89 | Ловит нелинейные паттерны, вероятность откалибрована               |
| **Ensemble (Default)**           | **~0.87 - 0.89** | **~0.88 - 0.90** | Усреднение вероятностей. Сглаживает излишнюю уверенность моделей   |

---

## API Endpoints

Сервер предоставляет REST API. По умолчанию используется **ансамбль** моделей, но вы можете принудительно выбрать одну из них через параметр `model_choice`.

### `GET /health`
Проверка готовности сервиса и загрузки артефактов.

{"status": "healthy", "models_loaded": true}

### `POST /predict`
Оценка соответствия резюме и вакансии.

**Пример запроса 1:**

curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Senior Java developer with 5 years of experience in Spring Boot, Hibernate, and microservices architecture. Proficient in AWS and Docker.",
    "job_text": "We are looking for a Senior Java Developer. Required: Spring Boot, Hibernate, AWS, Docker, microservices.",
    "resume_skill_list": "java, spring boot, hibernate, aws, docker, microservices, git, sql",
    "job_required_skills": "java, spring boot, hibernate, aws, docker, microservices",
    "model_choice": "ensemble"
  }'

**Ожидаемый ответ:**

{
  "prediction": 1,
  "probability": 0.8605,
  "model_used": "Ensemble (Linear + MLP)",
  "status": "success"
}

**Пример запроса 2:**

curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Chef with experience in Italian cuisine and restaurant management.",
    "job_text": "Full stack developer with JavaScript, React, and Node.js.",
    "resume_skill_list": "cooking, pizza, pasta, restaurant, management",
    "job_required_skills": "javascript, react, node.js, html, css",
    "model_choice": "ensemble"
  }'

**Ожидаемый ответ:**

{
  "prediction": 0,
  "probability": 0.1365,
  "model_used": "Ensemble (Linear + MLP)",
  "status": "success"
}

> **Параметр `model_choice`**: Принимает значения `"ensemble"` по умолчанию, `"linear"` или `"mlp"`.

**Swagger UI:** Полная документация доступна по адресу: http://localhost:8000/docs

---

## Визуализация и мониторинг

### TensorBoard
Для отслеживания процесса обучения нейросети:

tensorboard --logdir=runs --host 0.0.0.0 --port 6006
# Открыть в браузере: http://localhost:6006

**Отслеживаемые метрики:** `Loss/train`, `Loss/val`, `Metrics/val_accuracy`, `Metrics/val_f1`.

### Логирование
- Консольный вывод.
- Файлы `logs/train.log` и `logs/torch_train.log` для детального анализа и отладки.

---

## Docker

### Сборка образа вручную:

docker build -t skillgap-engine:latest .

### Запуск контейнера с монтированием томов:

docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/configs:/app/configs \
  skillgap-engine:latest
---

## Воспроизводимость

1. **Детерминизм**: Фиксированный `seed=42` во всех операциях `numpy`, `torch` и `sklearn`.
2. **Конфигурация**: Все гиперпараметры вынесены в `configs/default.yaml`.
3. **Артефакты**: Модели, `StandardScaler`, конфиг архитектуры MLP `model_config.json` и калибратор сохраняются атомарно в папке `artifacts/`.
4. **Безопасность инференса**: Все числовые признаки перед подачей в модели проходят через `np.clip(features[:, -4:], -3.0, 3.0)`, что физически предотвращает взрыв предсказаний на аномальных входных данных.
