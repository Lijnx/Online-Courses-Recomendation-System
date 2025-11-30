import pandas as pd
from pathlib import Path

def load_and_clean_data(base_path: str = "."):
    """
    Загружает все таблицы из CSV и делает базовую очистку:
    - даты -> datetime
    - должности -> уровни (junior/middle/senior/other)
    - зарплаты -> упорядоченные категории + числовые коды
    """
    base_path = Path(base_path)

    # 1. Чтение всех CSV
    files = {
        "courses": "courses.csv",
        "course_skills": "course_skills.csv",
        "skills": "skills.csv",
        "students": "students.csv",
        "enrollments": "enrollments.csv",
        "career_history": "career_history.csv",
        "social_signals": "social_signals.csv",
        "repos": "repos.csv",
        "vacancies": "vacancies.csv",
        "vacancy_skills": "vacancy_skills.csv",
        "student_skills_current": "student_skills_current.csv",
    }

    dfs = {}
    for name, fname in files.items():
        dfs[name] = pd.read_csv(base_path / fname)

    # 2. Нормализация дат -> datetime

    # enrollments: enroll_date, completion_date
    dfs["enrollments"]["enroll_date"] = pd.to_datetime(
        dfs["enrollments"]["enroll_date"]
    )
    dfs["enrollments"]["completion_date"] = pd.to_datetime(
        dfs["enrollments"]["completion_date"], errors="coerce"
    )

    # career_history: event_date
    dfs["career_history"]["event_date"] = pd.to_datetime(
        dfs["career_history"]["event_date"]
    )

    # social_signals: date
    dfs["social_signals"]["date"] = pd.to_datetime(
        dfs["social_signals"]["date"]
    )

    # 3. Должности -> унифицированные уровни

    def role_to_level(role: str) -> str:
        """Грубая эвристика: из текстового названия должности получить уровень."""
        if not isinstance(role, str):
            return "other"
        r = role.lower()

        # стажеры считаем ближе к junior
        if "intern" in r or "стажер" in r or "trainee" in r:
            return "junior"
        if "junior" in r or "младший" in r:
            return "junior"
        if "middle" in r or "mid " in r or "mid-" in r:
            return "middle"
        if "senior" in r or "lead" in r or "principal" in r or "head" in r:
            return "senior"
        # все остальные вроде Waiter, Economist, etc.
        return "other"

    dfs["career_history"]["role_level"] = dfs["career_history"]["role_title"].apply(
        role_to_level
    )

    # (опционально) то же самое можно сделать для вакансий:
    # dfs["vacancies"]["role_level"] = dfs["vacancies"]["title"].apply(role_to_level)

    # 4. Зарплаты -> категории (low/medium/high) + числовые коды

    salary_order = ["low", "medium", "high"]

    # career_history.salary_band
    dfs["career_history"]["salary_band"] = (
        dfs["career_history"]["salary_band"]
        .astype(str)
        .str.lower()
    )
    dfs["career_history"]["salary_band"] = pd.Categorical(
        dfs["career_history"]["salary_band"],
        categories=salary_order,
        ordered=True,
    )
    dfs["career_history"]["salary_level_num"] = (
        dfs["career_history"]["salary_band"].cat.codes
    )
    # low -> 0, medium -> 1, high -> 2, NaN -> -1

    # students.prev_salary_band
    dfs["students"]["prev_salary_band"] = (
        dfs["students"]["prev_salary_band"]
        .astype(str)
        .str.lower()
    )
    dfs["students"]["prev_salary_band"] = pd.Categorical(
        dfs["students"]["prev_salary_band"],
        categories=salary_order,
        ordered=True,
    )
    dfs["students"]["prev_salary_level_num"] = (
        dfs["students"]["prev_salary_band"].cat.codes
    )

    return dfs


# # пример использования:
# dfs = load_and_clean_data(".")
# print(dfs["enrollments"].head())
# print(dfs["career_history"].head())
# print(dfs["students"].head())
