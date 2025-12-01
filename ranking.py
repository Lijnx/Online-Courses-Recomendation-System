from typing import Dict, Any
import numpy as np
import pandas as pd
from xgboost import XGBRanker


# Уровни курса/юзера для сопоставления
LEVEL_ORDER = {
    "beginner": 0,
    "intermediate": 1,
    "advanced": 2,
}


# ---------- 1. Сбор профиля пользователя (CLI) ----------

def _ask_with_default(prompt: str, default: str) -> str:
    """Вспомогательная: спрашивает значение, если пусто — берёт default."""
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def ask_user_profile_cli(default_goal_domain: str = "data_analytics") -> Dict[str, Any]:
    """
    Интерфейс вопросов в консоли, чтобы заполнить профиль пользователя.

    current_level:  beginner / intermediate / advanced
    hours_per_week: целое число
    budget:         число (максимальный бюджет на курс)
    need_certificate: True/False (нужен ли обязательно сертификат)
    time_horizon_months: сколько месяцев пользователь готов учиться до выхода на работу
    goal_domain:   целевая область (пока используем 'data_analytics')
    """

    print("=== Настройка профиля пользователя ===")

    # 1. Уровень
    print("Выберите текущий уровень:")
    print("  1 - новичок (beginner)")
    print("  2 - уже немного опыта (intermediate)")
    print("  3 - уверенный уровень (advanced)")
    lvl_raw = _ask_with_default("Введите 1 / 2 / 3", "1")
    lvl_map = {"1": "beginner", "2": "intermediate", "3": "advanced"}
    current_level = lvl_map.get(lvl_raw.strip(), "beginner")

    # 2. Часы в неделю
    hours_raw = _ask_with_default("Сколько часов в неделю вы готовы учиться?", "10")
    try:
        hours_per_week = int(hours_raw)
    except ValueError:
        hours_per_week = 10

    # 3. Бюджет
    budget_raw = _ask_with_default("Какой максимальный бюджет на курс (в у.е.)?", "400")
    try:
        budget = float(budget_raw)
    except ValueError:
        budget = 400.0

    # 4. Сертификат
    cert_raw = _ask_with_default(
        "Насколько важен сертификат? (y/n — y = нужен обязательно)", "y"
    ).lower()
    need_certificate = cert_raw.startswith("y")

    # 5. Горизонт по времени
    horizon_raw = _ask_with_default(
        "Через сколько месяцев вы хотите выйти на новую работу?", "6"
    )
    try:
        time_horizon_months = float(horizon_raw)
    except ValueError:
        time_horizon_months = 6.0

    # 6. Целевая область
    goal_domain = _ask_with_default(
        "Целевая область (пока используем 'data_analytics')",
        default_goal_domain,
    )

    user_profile = {
        "current_level": current_level,
        "hours_per_week": hours_per_week,
        "budget": budget,
        "need_certificate": need_certificate,
        "time_horizon_months": time_horizon_months,
        "goal_domain": goal_domain,
    }

    print("\nПрофиль пользователя:")
    for k, v in user_profile.items():
        print(f"  {k}: {v}")

    return user_profile


# ---------- 2. Построение признаков для курсов ----------

def _level_to_num(level: str) -> int:
    if not isinstance(level, str):
        return 1  # default intermediate
    return LEVEL_ORDER.get(level.lower(), 1)


def _compute_level_match(course_level: str, user_level: str) -> float:
    """
    Чем ближе уровни курса и пользователя, тем выше скор.
    0 — совсем не подходит, 1 — идеально.
    """
    c = _level_to_num(course_level)
    u = _level_to_num(user_level)
    dist = abs(c - u)
    # max dist = 2 -> оставим линейную шкалу
    score = 1.0 - dist / 2.0
    return float(max(0.0, min(1.0, score)))


def _weeks_to_months(weeks: float) -> float:
    return float(weeks) / 4.345 if pd.notna(weeks) else np.nan


def _compute_budget_features(price: float, budget: float) -> Dict[str, float]:
    """
    price <= budget -> fit = 1, gap <= 0
    иначе fit убывает, gap > 0
    """
    if budget <= 0:
        # если бюджет 0, просто считаем, что всё не влезает
        return {"budget_fit": 0.0, "budget_gap": float(price)}

    gap = price - budget
    if price <= budget:
        return {"budget_fit": 1.0, "budget_gap": gap}
    else:
        # чем сильнее превышение, тем хуже (но не менее 0)
        over_ratio = gap / budget
        fit = max(0.0, 1.0 - over_ratio)
        return {"budget_fit": fit, "budget_gap": gap}


def _compute_time_features(duration_weeks: float, user_horizon_months: float) -> Dict[str, float]:
    """
    Оцениваем, укладывается ли длительность курса в горизонт пользователя.
    """
    course_months = _weeks_to_months(duration_weeks)
    if pd.isna(course_months):
        return {
            "time_feasible": 0.0,
            "time_over_horizon_months": np.nan,
        }

    over = course_months - user_horizon_months
    time_feasible = 1.0 if over <= 0 else max(0.0, 1.0 - over / max(user_horizon_months, 1.0))

    return {
        "time_feasible": time_feasible,
        "time_over_horizon_months": over,
    }


def _compute_skill_overlap_for_courses(dfs, goal_domain: str) -> pd.Series:
    """
    Jaccard-похожесть между набором скиллов курса и целевым набором скиллов
    (объединённые skills всех вакансий target_domain).
    Требует, чтобы nlp_step уже проставил:
      - dfs["courses"]["skill_ids_extracted"]
      - dfs["vacancies"]["skill_ids_extracted"]
    :contentReference[oaicite:1]{index=1}
    """

    courses = dfs["courses"]
    vacancies = dfs["vacancies"]

    if "skill_ids_extracted" not in courses.columns:
        # NLP-скиллы не посчитаны — вернём нули
        return pd.Series(0.0, index=courses["course_id"])

    # собираем целевой набор навыков
    vac_target = vacancies[vacancies.get("target_domain", "") == goal_domain]
    if "skill_ids_extracted" not in vac_target.columns or vac_target.empty:
        # целевых вакансий нет — тоже нули
        return pd.Series(0.0, index=courses["course_id"])

    target_skill_sets = []
    for lst in vac_target["skill_ids_extracted"]:
        if isinstance(lst, (list, tuple, set)):
            target_skill_sets.append(set(lst))

    if not target_skill_sets:
        return pd.Series(0.0, index=courses["course_id"])

    target_skills_union = set().union(*target_skill_sets)
    if not target_skills_union:
        return pd.Series(0.0, index=courses["course_id"])

    overlaps = {}

    for _, row in courses.iterrows():
        cid = row["course_id"]
        skills = row.get("skill_ids_extracted", [])
        if isinstance(skills, str):
            # если вдруг сохранили как строку, попробовать распарсить через запятую
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        if not isinstance(skills, (list, tuple, set)):
            overlaps[cid] = 0.0
            continue

        s_course = set(skills)
        inter = len(s_course & target_skills_union)
        union = len(s_course | target_skills_union)
        overlaps[cid] = inter / union if union > 0 else 0.0

    return pd.Series(overlaps, name="skill_overlap_with_goal")


def build_course_feature_matrix(dfs, user_profile: Dict[str, Any]) -> pd.DataFrame:
    """
    Строит матрицу признаков для ранжирования курсов под заданный user_profile.

    На входе ожидается, что уже выполнены:
      - load_and_clean_data("data")
      - run_nlp_step(dfs)
      - compute_student_course_outcomes(dfs, ...)
      - compute_course_effectiveness_metrics(dfs)
    т.е. в dfs есть:
      - dfs["course_effectiveness"]
      - dfs["courses"] (с skill_ids_extracted)
      - dfs["vacancies"] (с skill_ids_extracted)
    
    """

    eff = dfs["course_effectiveness"].copy()
    courses = dfs["courses"].copy()

    # Присоединим skill_overlap
    skill_overlap = _compute_skill_overlap_for_courses(dfs, user_profile["goal_domain"])
    eff = eff.merge(
        skill_overlap.rename("skill_overlap_with_goal"),
        left_on="course_id",
        right_index=True,
        how="left",
    )

    # Уровень пользователя
    user_level = user_profile["current_level"]
    user_hours = user_profile["hours_per_week"]
    user_budget = user_profile["budget"]
    user_need_cert = user_profile["need_certificate"]
    user_horizon = user_profile["time_horizon_months"]

    # Вычисляем признаки для каждого курса
    feature_rows = []

    for _, row in eff.iterrows():
        cid = row["course_id"]
        level = row.get("level", "")
        price = float(row.get("price", 0.0))
        duration_weeks = float(row.get("duration_weeks", np.nan))
        hours_per_week = float(row.get("hours_per_week", np.nan))
        has_certificate = bool(row.get("has_certificate", False)) if "has_certificate" in row.index else False

        # 1. Совместимость по уровню
        level_match = _compute_level_match(level, user_level)

        # 2. Нагрузка vs доступное время
        # (hours_fit: если курс требует <= user_hours, то 1, иначе падает до 0)
        if np.isnan(hours_per_week) or user_hours <= 0:
            hours_fit = 0.0
        else:
            if hours_per_week <= user_hours:
                hours_fit = 1.0
            else:
                over_ratio = (hours_per_week - user_hours) / user_hours
                hours_fit = max(0.0, 1.0 - over_ratio)

        # 3. Бюджет
        budget_feats = _compute_budget_features(price, user_budget)

        # 4. Вписывается ли по времени в горизонт
        time_feats = _compute_time_features(duration_weeks, user_horizon)

        # 5. Совпадение по сертификату
        if user_need_cert:
            certificate_match = 1.0 if has_certificate else 0.0
        else:
            certificate_match = 1.0  # если сертификат не обязателен, все ок

        # 6. Эффективность курса (уже посчитанная)
        success_rate = float(row.get("success_rate", np.nan))
        median_time_to_job = float(row.get("median_time_to_job_months", np.nan))
        median_salary_change = float(row.get("median_salary_change", np.nan))
        career_switch_rate = float(row.get("career_switch_rate", np.nan))
        avg_social_success_score = float(row.get("avg_social_success_score", 0.0))
        skill_overlap_score = float(row.get("skill_overlap_with_goal", 0.0))

        feature_rows.append({
            "course_id": cid,
            "title": row.get("title", ""),
            "level": level,
            "price": price,
            "duration_weeks": duration_weeks,
            "hours_per_week": hours_per_week,
            # признаки совместимости с пользователем:
            "user_level": user_level,
            "level_match_score": level_match,
            "hours_fit_score": hours_fit,
            "budget_fit": budget_feats["budget_fit"],
            "budget_gap": budget_feats["budget_gap"],
            "time_feasible": time_feats["time_feasible"],
            "time_over_horizon_months": time_feats["time_over_horizon_months"],
            "certificate_match": certificate_match,
            "skill_overlap_with_goal": skill_overlap_score,
            # объективные метрики эффективности:
            "success_rate": success_rate,
            "median_time_to_job_months": median_time_to_job,
            "median_salary_change": median_salary_change,
            "career_switch_rate": career_switch_rate,
            "avg_social_success_score": avg_social_success_score,
        })

    df_features = pd.DataFrame(feature_rows)
    return df_features

def derive_user_profile_from_student(row: pd.Series, goal_domain: str = "data_analytics") -> Dict[str, Any]:
    """
    Строим user_profile для обучения ранжирования из строки студента.

    current_level:
        - из prev_domain: out_of_IT -> beginner, IT_related -> intermediate, business -> intermediate
    остальные поля - разумные default'ы.

    Это нужно только для генерации обучающей выборки (исторических запросов).
    """
    prev_domain = str(row.get("prev_domain", "")).lower()

    if prev_domain == "out_of_it":
        current_level = "beginner"
    elif prev_domain == "it_related":
        current_level = "intermediate"
    else:
        current_level = "beginner"

    # Можно усложнить, но для синтетики достаточно:
    user_profile = {
        "current_level": current_level,
        "hours_per_week": 10,          # допустим, все готовы учиться 10 ч/нед
        "budget": 300.0,               # усреднённый бюджет
        "need_certificate": False,     # для обучения не будем требовать сертификат
        "time_horizon_months": 6.0,    # хотим выйти на работу за 6 месяцев
        "goal_domain": goal_domain,
    }
    return user_profile

# Те же числовые признаки, что мы строим в build_course_feature_matrix
RANKING_FEATURE_COLUMNS: list[str] = [
    "price",
    "duration_weeks",
    "hours_per_week",
    "level_match_score",
    "hours_fit_score",
    "budget_fit",
    "budget_gap",
    "time_feasible",
    "time_over_horizon_months",
    "certificate_match",
    "skill_overlap_with_goal",
    "success_rate",
    "median_time_to_job_months",
    "median_salary_change",
    "career_switch_rate",
    "avg_social_success_score",
]


def build_ranking_training_dataset(
    dfs,
    goal_domain: str = "data_analytics") -> tuple[pd.DataFrame, pd.Series, list[int], list[str]]:
    """
    Строит обучающую выборку для XGBRanker:

    Возвращает:
      - X: DataFrame с признаками (строки — (student, course))
      - y: Series с релевантностями (0/1/2)
      - group: список длиной = числу "запросов" (студентов),
               где каждый элемент = кол-во строк (курсов) для данного студента.
      - query_ids: список student_id в том же порядке, что и group.

    Логика разметки:
      - курс, который привёл к успеху (success_label=1) -> relevance = 2
      - курс, который студент прошёл, но успеха не достиг -> relevance = 1
      - курс, который студент не проходил -> relevance = 0
    """
    if "student_course_outcomes" not in dfs:
        raise ValueError("В dfs нет 'student_course_outcomes'. Сначала вызови compute_student_course_outcomes().")
    if "course_effectiveness" not in dfs:
        raise ValueError("В dfs нет 'course_effectiveness'. Сначала вызови compute_course_effectiveness_metrics().")

    outcomes = dfs["student_course_outcomes"].copy()
    students = dfs["students"].copy()

    all_courses = dfs["course_effectiveness"]["course_id"].tolist()

    X_rows = []
    y_rows = []
    group_sizes = []
    query_ids = []

    # Для удобства - сгруппируем исходы по студентам
    outcomes_by_student = {
        sid: grp for sid, grp in outcomes.groupby("student_id")
    }

    for _, stud in students.iterrows():
        sid = stud["student_id"]
        # Профиль пользователя для этого "исторического запроса"
        user_profile = derive_user_profile_from_student(stud, goal_domain=goal_domain)

        # Признаки для всех курсов под этот профиль
        df_features = build_course_feature_matrix(dfs, user_profile)

        # Исходы по курсам для этого студента (только завершённые)
        stud_outcomes = outcomes_by_student.get(sid)
        if stud_outcomes is None or stud_outcomes.empty:
            # нет информации — можно пропустить запрос
            continue

        # Словарь: course_id -> success_label (0/1)
        success_map = stud_outcomes.set_index("course_id")["success_label"].to_dict()

        # Релевантность по правилам задачи
        relevance = []
        for _, row in df_features.iterrows():
            cid = row["course_id"]
            if cid in success_map:
                if success_map[cid] == 1:
                    rel = 2  # прошёл и успешно вышел в целевой домен вовремя
                else:
                    rel = 1  # проходил, но не success по нашим критериям
            else:
                rel = 0      # не проходил
            relevance.append(rel)

        # Если у студента все релевантности 0 — смысла в таком запросе мало
        if max(relevance) == 0:
            continue

        # Оставляем только числовые фичи для XGBoost
        X_student = df_features[RANKING_FEATURE_COLUMNS].copy()
        y_student = pd.Series(relevance)

        X_rows.append(X_student)
        y_rows.append(y_student)
        group_sizes.append(len(X_student))
        query_ids.append(sid)

    if not X_rows:
        raise ValueError("Не удалось собрать ни одного 'запроса' для обучения ранжирования.")

    X = pd.concat(X_rows, axis=0, ignore_index=True)
    y = pd.concat(y_rows, axis=0, ignore_index=True)

    return X, y, group_sizes, query_ids

def train_xgboost_ranker(
    dfs,
    goal_domain: str = "data_analytics",
    params: Dict[str, Any] | None = None,
) -> tuple[XGBRanker, list[str]]:
    """
    Обучает XGBRanker на исторических "запросах" (студентах).

    Возвращает:
      - обученную модель XGBRanker
      - список имён признаков (RANKING_FEATURE_COLUMNS)
    """
    X, y, group, query_ids = build_ranking_training_dataset(dfs, goal_domain=goal_domain)

    if params is None:
        params = {
            "objective": "rank:pairwise",
            "learning_rate": 0.1,
            "n_estimators": 200,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

    model = XGBRanker(
        objective=params.get("objective", "rank:pairwise"),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 4),
        min_child_weight=params.get("min_child_weight", 1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        random_state=params.get("random_state", 42),
    )

    model.fit(
        X,
        y,
        group=group,
        verbose=False,
    )

    return model, RANKING_FEATURE_COLUMNS

def rank_courses_for_user(
    dfs,
    model: XGBRanker,
    user_profile: Dict[str, Any],
    feature_columns: list[str] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Ранжирует курсы под конкретный user_profile с помощью обученного XGBRanker.

    Возвращает DataFrame с колонками:
      - course_id
      - title
      - level
      - price
      - ... + все признаки
      - score  (предсказанный моделью ранг-скор)
    """
    if feature_columns is None:
        feature_columns = RANKING_FEATURE_COLUMNS

    df_features = build_course_feature_matrix(dfs, user_profile)

    X = df_features[feature_columns].copy()
    scores = model.predict(X)

    df_features = df_features.copy()
    df_features["score"] = scores

    df_ranked = df_features.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return df_ranked
