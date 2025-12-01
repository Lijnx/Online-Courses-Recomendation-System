# Intelligent Course Recommendation System

Этот проект — прототип интеллектуальной системы рекомендаций онлайн‑курсов, которая оценивает курсы не по звёздочкам и маркетинговым описаниям, а по **реальным карьерным результатам выпускников**.

Проект строится вокруг полного ML‑пайплайна:
1. Сбор и очистка данных
2. NLP‑обработка: извлечение навыков, успешностей, карьерных переходов
3. Построение метрик эффективности курсов
4. Агрегация признаков на уровне курса
5. Построение пользовательского профиля
6. Ранжирование курсов (Learning‑to‑Rank, XGBoost)

---

# Структура проекта

```
project_root/
│
├── data/                      # синтетические датасеты (CSV)
│   ├── courses.csv
│   ├── career_history.csv
│   ├── students.csv
│   ├── enrollments.csv
│   ├── social_signals.csv
│   ├── vacancies.csv
│   └── ...
│
├── data_handler.py            # шаг 1: загрузка и очистка
├── nlp_step.py                # шаг 2: обработка текстов, навыков, соц. сигналов
├── compute_metrics.py         # шаги 3–4: карьерные исходы и эффективность курсов
├── ranking.py                 # шаги 5–6: профиль пользователя + ML‑ранжирование
├── datasets_synthesizer.py    # генерация синтетических данных
├── main.py                    # основной pipeline запуска
│
├── requirements.txt           # зависимости проекта
└── README.md                  # документация
```

---

# Пайплайн обработки (по шагам)

## **Шаг 1 — Загрузка и очистка (`data_handler.py`)**
- Чтение всех CSV из `data/`
- Приведение дат → `datetime`
- Нормализация уровней ролей → `junior / middle / senior / other`
- Нормализация зарплат → категории (`low/medium/high`) и числовые уровни

## **Шаг 2 — NLP (`nlp_step.py`)**
- Извлечение навыков из описаний курсов и вакансий — spaCy + SkillNER
- Классификация постов (`success / neutral / negative`)
- Извлечение компаний и позиций (regex)
- Определение домена роли (`data_analytics`, `software_engineering`, ...)
- Построение переходов студентов в целевой домен

## **Шаг 3 — Карьерные исходы (`compute_metrics.py`)**
Для каждой пары *(student, course)* считается:
- time_to_target_job — время до устройства
- salary_change — изменение зарплатного грейда
- career_switch — вход в новую сферу
- success_label — бинарная успешность (≤ 6 месяцев + правильный домен)
- social_success_score — позитивные упоминания курса у студента

## **Шаг 4 — Метрики курса (`compute_metrics.py`)**
Агрегируются объективные показатели:
- success_rate
- median_time_to_job
- median_salary_change
- career_switch_rate
- avg_social_success_score

Эти признаки отражают *эффективность курса в реальном трудоустройстве*.

## **Шаг 5 — Профиль пользователя (`ranking.py`)**
Путём интерактивных вопросов собирается:
```python
user_profile = {
    "current_level": "beginner",
    "hours_per_week": 10,
    "budget": 400,
    "need_certificate": True,
    "time_horizon_months": 6,
    "goal_domain": "data_analytics"
}
```
Строятся признаки совместимости курса с этим профилем.

## **Шаг 6 — Ранжирование (Learning‑to‑Rank, XGBoost)**
Реализовано в `ranking.py`:
- генерируются «исторические запросы» из реальных студентов,
- формируется выборка с релевантностью:
  - 2 — курс, приведший к успеху,
  - 1 — курс, который студент проходил без успеха,
  - 0 — курсы, которые не проходил.
- обучается `XGBRanker` (pairwise ranking)
- на инференсе оцениваются все курсы под профиль пользователя.

---

# Как установить

## 1. Создать виртуальное окружение
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## 2. Установить зависимости
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -m spacy download ru_core_news_md
```

## 3. Запустить проект

# Запуск проекта

Предварительно запустить `datasets_synthesizer.py` для генерации датасета, если потребуется.

Минимальный пример из `main.py`:
```python
from data_handler import load_and_clean_data
from nlp_step import run_nlp_step
from compute_metrics import compute_student_course_outcomes, compute_course_effectiveness_metrics
from ranking import ask_user_profile_cli, train_xgboost_ranker, rank_courses_for_user

# 1. Загрузка данных + NLP + метрики
dfs = load_and_clean_data("data")
dfs = run_nlp_step(dfs)
dfs = compute_student_course_outcomes(dfs, target_domain="data_analytics")
dfs = compute_course_effectiveness_metrics(dfs)

# 2. Обучаем ранкер
model, feat_cols = train_xgboost_ranker(dfs)

# 3. Получаем профиль пользователя
user_profile = ask_user_profile_cli()

# 4. Ранжируем курсы
df_ranked = rank_courses_for_user(dfs, model, user_profile, feature_columns=feat_cols, top_n=10)
print(df_ranked)
```

---

# Дальнейшие цели

## Шаг 7 — Интерфейс и объяснимость
- Выводить обоснования рекомендаций: почему курс попал в топ?
- Генерировать текстовые объяснения (LLM)
- Опционально: добавить веб-интерфейс (Streamlit / Gradio)

На данном этапе:
* Шаги 1–6 полностью реализованы  
* Данные генерируются синтетически  
* Реализован ML‑ранкер XGBoost  
* Построена архитектура для масштабирования
