import numpy as np
import pandas as pd

def compute_student_course_outcomes(
    dfs,
    target_domain: str = "data_analytics",
    success_horizon_months: float = 6.0,
):
    """
    Шаг 3.
    На основе:
      - dfs["enrollments"]          (completion_date, completion_status)
      - dfs["career_history"]       (event_date, role_domain_nlp, role_level, salary_level_num)
      - dfs["students"]             (prev_salary_level_num)
      - dfs["social_signals"]       (sentiment_label)
    строит таблицу student_course_outcomes:

      - student_id
      - course_id
      - success_label (0/1)
      - time_to_target_job_months (float или NaN)
      - salary_change (int или NaN)
      - career_switch (bool)
      - social_success_score (int)
      - social_negative_score (int)
      - baseline_salary_level (int или NaN)
      - target_salary_level   (int или NaN)
      - target_event_date     (datetime или NaT)
    """

    enroll = dfs["enrollments"].copy()
    career = dfs["career_history"].copy()
    students = dfs["students"].copy()
    social = dfs["social_signals"].copy() if "social_signals" in dfs else pd.DataFrame()

    # --- подготовка: группы по студентам / студенту+курсу ---

    # career_history по студентам, отсортировано по дате
    career = career.sort_values(["student_id", "event_date"])
    career_by_student = {
        sid: grp for sid, grp in career.groupby("student_id")
    }

    # словарь базовой зарплаты из students (на случай, если в истории нет события до курса)
    prev_salary_map = students.set_index("student_id")["prev_salary_level_num"].to_dict()

    # social_signals: считаем success/negative по паре (student_id, course_id)
    if not social.empty and "sentiment_label" in social.columns:
        social_success = (
            social[social["sentiment_label"] == "success"]
            .groupby(["student_id", "course_id"])
            .size()
            .to_dict()
        )
        social_negative = (
            social[social["sentiment_label"] == "negative"]
            .groupby(["student_id", "course_id"])
            .size()
            .to_dict()
        )
    else:
        social_success = {}
        social_negative = {}

    # --- берём только завершённые прохождения курсов ---

    completed = enroll[enroll["completion_status"] == "completed"].copy()
    completed = completed[completed["completion_date"].notna()]

    rows = []

    for _, row in completed.iterrows():
        sid = row["student_id"]
        cid = row["course_id"]
        comp_date = row["completion_date"]

        # --- карьерная история этого студента ---
        ch = career_by_student.get(sid)
        if ch is None or ch.empty:
            # нет никакой истории — считаем, что успеха нет
            rows.append({
                "student_id": sid,
                "course_id": cid,
                "success_label": 0,
                "time_to_target_job_months": pd.NA,
                "salary_change": pd.NA,
                "career_switch": False,
                "social_success_score": social_success.get((sid, cid), 0),
                "social_negative_score": social_negative.get((sid, cid), 0),
                "baseline_salary_level": pd.NA,
                "target_salary_level": pd.NA,
                "target_event_date": pd.NaT,
            })
            continue

        # --- базовая зарплата до курса: последнее событие до completion_date ---
        before_course = ch[ch["event_date"] <= comp_date]

        if not before_course.empty:
            base_row = before_course.iloc[-1]
            baseline_salary_level = base_row.get("salary_level_num", pd.NA)
            baseline_domain = base_row.get("role_domain_nlp", base_row.get("domain", None))
        else:
            # fallback: salary из students
            baseline_salary_level = prev_salary_map.get(sid, pd.NA)
            baseline_domain = None

        # --- ищем целевое событие: после completion_date, в нужном домене/уровне ---

        after_course = ch[ch["event_date"] > comp_date]

        # условие: domain == target_domain ИЛИ уровень junior/middle
        cond_domain = after_course.get("role_domain_nlp", after_course.get("domain"))
        cond_level = after_course.get("role_level")

        mask = pd.Series([True] * len(after_course), index=after_course.index)

        if cond_domain is not None:
            mask = mask & (cond_domain == target_domain)

        if cond_level is not None:
            mask = mask | cond_level.isin(["junior", "middle"])

        target_events = after_course[mask]

        if not target_events.empty:
            # ближайшее событие по дате
            target_row = target_events.iloc[0]
            target_event_date = target_row["event_date"]
            target_domain_val = target_row.get("role_domain_nlp", target_row.get("domain", None))
            target_salary_level = target_row.get("salary_level_num", pd.NA)

            # время до целевой работы в месяцах
            delta_days = (target_event_date - comp_date).days
            time_months = delta_days / 30.4375  # средняя длина месяца

            # изменение зарплаты
            if pd.isna(baseline_salary_level) or baseline_salary_level < 0:
                salary_change = pd.NA
            else:
                salary_change = (
                    target_salary_level - baseline_salary_level
                    if pd.notna(target_salary_level) and target_salary_level >= 0
                    else pd.NA
                )

            # смена сферы: был не в target_domain → стал в target_domain
            if baseline_domain is None:
                career_switch = False
            else:
                career_switch = (baseline_domain != target_domain) and (target_domain_val == target_domain)

            # успех: уложился в горизонт и в правильном домене
            success_label = int(
                (target_domain_val == target_domain)
                and (time_months <= success_horizon_months)
            )

        else:
            # нет ни одного подходящего события после курса
            target_event_date = pd.NaT
            target_salary_level = pd.NA
            time_months = pd.NA
            salary_change = pd.NA
            career_switch = False
            success_label = 0

        # --- социальные сигналы ---
        social_success_score = social_success.get((sid, cid), 0)
        social_negative_score = social_negative.get((sid, cid), 0)

        rows.append({
            "student_id": sid,
            "course_id": cid,
            "success_label": success_label,
            "time_to_target_job_months": time_months,
            "salary_change": salary_change,
            "career_switch": career_switch,
            "social_success_score": social_success_score,
            "social_negative_score": social_negative_score,
            "baseline_salary_level": baseline_salary_level,
            "target_salary_level": target_salary_level,
            "target_event_date": target_event_date,
        })

    df_outcomes = pd.DataFrame(rows)
    dfs["student_course_outcomes"] = df_outcomes
    return dfs


def assign_student_segment(row) -> str:
    """
    Грубая сегментация студентов по исходным данным.
    Можно усложнить, но для синтетики достаточно:

    - 'beginner'    — пришёл из не-IT (prev_domain == 'out_of_IT')
    - 'mid_career'  — все остальные (business, IT_related и т.д.)
    """
    prev_domain = row.get("prev_domain", None)
    if isinstance(prev_domain, str):
        prev_domain = prev_domain.lower()
    else:
        return "unknown"

    if prev_domain == "out_of_it":
        return "beginner"
    return "mid_career"


def compute_course_effectiveness_metrics(dfs):
    """
    Шаг 4. Аггрегация на уровне курса.

    Вход:
      dfs["student_course_outcomes"] — результат compute_student_course_outcomes
      dfs["students"]                — информация о студентах

    Выход:
      dfs["course_effectiveness"] — DataFrame с метриками по курсам:

        - course_id
        - n_enrollments_completed
        - success_rate
        - median_time_to_job_months
        - median_salary_change
        - career_switch_rate
        - avg_social_success_score
        - avg_social_negative_score

        - success_rate_for_beginners
        - success_rate_for_mid_career
    """
    if "student_course_outcomes" not in dfs:
        raise ValueError("В dfs нет 'student_course_outcomes'. Сначала вызови compute_student_course_outcomes().")

    outcomes = dfs["student_course_outcomes"].copy()
    students = dfs["students"].copy()

    # Подтянем к исходам информацию о студенте
    # важно, чтобы в students были prev_domain и prev_salary_* (из data_handler)
    merge_cols = [
        "student_id",
        "prev_domain",
        "prev_salary_band",
        "prev_salary_level_num",
    ]
    # если каких-то колонок нет, просто отфильтруем их
    merge_cols = [c for c in merge_cols if c in students.columns]

    students_short = students[merge_cols].copy()

    df = outcomes.merge(students_short, on="student_id", how="left")

    # Сегментируем студентов
    df["student_segment"] = df.apply(assign_student_segment, axis=1)

    rows = []

    # Группируем по course_id
    for course_id, g in df.groupby("course_id"):
        # количество завершённых прохождений
        n = len(g)

        if n == 0:
            # на всякий случай, но группировка по определению пустой группы не даст
            continue

        # success_rate: среднее по success_label (0/1)
        success_rate = g["success_label"].mean() if "success_label" in g.columns else np.nan

        # медианное время до устройства
        median_time_to_job = (
            g["time_to_target_job_months"].median()
            if "time_to_target_job_months" in g.columns
            else np.nan
        )

        # медианный рост зарплаты
        median_salary_change = (
            g["salary_change"].median()
            if "salary_change" in g.columns
            else np.nan
        )

        # career_switch_rate: среднее по bool
        career_switch_rate = (
            g["career_switch"].mean()
            if "career_switch" in g.columns
            else np.nan
        )

        # social scores
        avg_social_success_score = (
            g["social_success_score"].mean()
            if "social_success_score" in g.columns
            else 0.0
        )
        avg_social_negative_score = (
            g["social_negative_score"].mean()
            if "social_negative_score" in g.columns
            else 0.0
        )

        # --- стратифицированные метрики ---

        # начинающие (beginner)
        g_beg = g[g["student_segment"] == "beginner"]
        if len(g_beg) > 0:
            success_rate_beginners = g_beg["success_label"].mean()
        else:
            success_rate_beginners = np.nan

        # mid_career — все остальные, кроме beginner/unknown
        g_mid = g[g["student_segment"] == "mid_career"]
        if len(g_mid) > 0:
            success_rate_mid_career = g_mid["success_label"].mean()
        else:
            success_rate_mid_career = np.nan

        rows.append({
            "course_id": course_id,
            "n_enrollments_completed": n,
            "success_rate": success_rate,
            "median_time_to_job_months": median_time_to_job,
            "median_salary_change": median_salary_change,
            "career_switch_rate": career_switch_rate,
            "avg_social_success_score": avg_social_success_score,
            "avg_social_negative_score": avg_social_negative_score,
            "success_rate_for_beginners": success_rate_beginners,
            "success_rate_for_mid_career": success_rate_mid_career,
        })

    df_courses_eff = pd.DataFrame(rows)

    # можно ещё присоединить сюда базовую инфу о курсах (название, уровень, цена)
    if "courses" in dfs:
        df_courses_eff = df_courses_eff.merge(
            dfs["courses"][["course_id", "title", "level", "price", "duration_weeks", "hours_per_week"]],
            on="course_id",
            how="left",
        )

    dfs["course_effectiveness"] = df_courses_eff
    return dfs
