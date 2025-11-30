import re
import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
from data_handler import load_and_clean_data


def init_nlp():
    """
    Загружаем английскую и русскую модели + инициализируем SkillExtractor.
    Для наших текстов курсов/вакансий хватит en_core_web_md.
    """
    nlp_en = spacy.load("en_core_web_md")
    nlp_ru = spacy.load("ru_core_news_md")  # для русских постов в соцсетях
    skill_extractor = SkillExtractor(nlp_en, SKILL_DB, PhraseMatcher)
    return nlp_en, nlp_ru, skill_extractor


def build_skill_mapping(df_skills: pd.DataFrame):
    """
    Маппинг: имя навыка (lower) -> skill_id
    """
    return {
        name.lower(): sid
        for sid, name in zip(df_skills["skill_id"], df_skills["name"])
    }


def extract_skills_from_text(text: str, nlp, skill_extractor, skill_name_to_id):
    """
    Прогоняем текст через SkillNER и возвращаем список skill_id,
    которые нашлись в нашем словаре skills.csv.

    annotations = skill_extractor.annotate(text) даёт структуру:
    {
        "text": "...",
        "results": {
            "full_matches": [ {...}, {...}, ... ],
            "ngram_scored": [ {...}, {...}, ... ]
        }
    }

    Внутри каждого элемента есть, в частности:
      - doc_node_value — текст навыка ("python", "web development", ...)
      - skill_id       — внутренний id из базы SkillNER (EMSI и т.п.)
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # SkillNER сам внутри прогоняет через nlp, так что doc можно вообще не считать,
    # но если хочешь — можешь оставить:
    # doc = nlp(text)
    # annotations = skill_extractor.annotate(doc.text)
    annotations = skill_extractor.annotate(text)

    results = annotations.get("results", {})

    extracted_names = set()

    # Точные совпадения
    for item in results.get("full_matches", []):
        name = item.get("doc_node_value")
        if isinstance(name, str):
            extracted_names.add(name.lower())

    # n-gram совпадения
    for item in results.get("ngram_scored", []):
        name = item.get("doc_node_value")
        if isinstance(name, str):
            extracted_names.add(name.lower())

    # Переводим имена в наши skill_id по таблице skills.csv
    ids = [
        skill_name_to_id[name]
        for name in extracted_names
        if name in skill_name_to_id
    ]

    return ids



def nlp_extract_course_and_vacancy_skills(dfs, nlp_en, skill_extractor):
    """
    Для:
      - dfs["courses"] (колонку 'description' или title+skills_taught)
      - dfs["vacancies"] ('description_text')
    извлекаем skills с помощью skillNER.

    Добавляем:
      - courses: колонку 'skill_ids_extracted'
      - vacancies: 'skill_ids_extracted'
      - новые таблицы df_course_skills_nlp, df_vacancy_skills_nlp (если нужно).
    """
    df_skills = dfs["skills"]
    skill_name_to_id = build_skill_mapping(df_skills)

    courses = dfs["courses"].copy()
    vacancies = dfs["vacancies"].copy()

    # Если нет отдельного описания — используем title + skills_taught
    if "description" in courses.columns:
        course_texts = courses["description"]
    else:
        course_texts = courses["title"] + ". Skills: " + courses["skills_taught"].fillna("")

    vacancies_texts = vacancies["description_text"]

    # Извлекаем скиллы
    courses["skill_ids_extracted"] = course_texts.apply(
        lambda t: extract_skills_from_text(t, nlp_en, skill_extractor, skill_name_to_id)
    )
    vacancies["skill_ids_extracted"] = vacancies_texts.apply(
        lambda t: extract_skills_from_text(t, nlp_en, skill_extractor, skill_name_to_id)
    )

    # Можно развернуть в отдельные таблицы «course_id – skill_id»
    course_skill_rows = []
    for cid, skill_ids in zip(courses["course_id"], courses["skill_ids_extracted"]):
        for sid in skill_ids:
            course_skill_rows.append({"course_id": cid, "skill_id": sid})
    df_course_skills_nlp = pd.DataFrame(course_skill_rows).drop_duplicates()

    vacancy_skill_rows = []
    for vid, skill_ids in zip(vacancies["vacancy_id"], vacancies["skill_ids_extracted"]):
        for sid in skill_ids:
            vacancy_skill_rows.append({"vacancy_id": vid, "skill_id": sid})
    df_vacancy_skills_nlp = pd.DataFrame(vacancy_skill_rows).drop_duplicates()

    dfs["courses"] = courses
    dfs["vacancies"] = vacancies
    dfs["course_skills_nlp"] = df_course_skills_nlp
    dfs["vacancy_skills_nlp"] = df_vacancy_skills_nlp

    return dfs


SUCCESS_PATTERNS = [
    "получил оффер",
    "получила оффер",
    "устроился",
    "устроилась",
    "нашел работу",
    "нашла работу",
    "вышел на новую работу",
    "вышла на новую работу",
]
NEGATIVE_PATTERNS = [
    "не помог",
    "бесполезный курс",
    "ничего не дал",
    "разочарован",
    "разочарована",
]


def classify_post_sentiment(text: str) -> str:
    """
    Очень простой rule-based классификатор:
    success / negative / neutral.
    """
    if not isinstance(text, str):
        return "neutral"
    t = text.lower()
    if any(p in t for p in SUCCESS_PATTERNS):
        return "success"
    if any(p in t for p in NEGATIVE_PATTERNS):
        return "negative"
    return "neutral"


# Примеры шаблонов: 
# "получил оффер на позицию Junior Data Analyst в компании Y"
ROLE_PATTERN = re.compile(r"позици[юи]\s+([^.,]+)", re.IGNORECASE)
COMPANY_PATTERN = re.compile(r"в\s+компан[иияе]\s+([^.,]+)", re.IGNORECASE)


def extract_role_and_company(text: str):
    """
    Извлекаем название позиции и компании из текста (если удаётся).
    Возвращаем (role_title, company_name).
    """
    if not isinstance(text, str):
        return None, None

    role_match = ROLE_PATTERN.search(text)
    comp_match = COMPANY_PATTERN.search(text)

    role = role_match.group(1).strip() if role_match else None
    company = comp_match.group(1).strip() if comp_match else None

    return role, company


def nlp_process_social_signals(dfs, nlp_ru):
    """
    Обрабатываем dfs["social_signals"]:
      - sentiment_label: success/negative/neutral
      - extracted_role_title
      - extracted_company
    (nlp_ru сюда можно подключить для более умного парсинга,
     но в базовой версии достаточно regex + rule-based.)
    """
    social = dfs["social_signals"].copy()

    social["sentiment_label"] = social["text"].apply(classify_post_sentiment)

    roles = []
    companies = []
    for txt in social["text"]:
        role, comp = extract_role_and_company(txt)
        roles.append(role)
        companies.append(comp)

    social["extracted_role_title"] = roles
    social["extracted_company"] = companies

    dfs["social_signals"] = social
    return dfs


def role_to_domain(role: str) -> str:
    """
    Грубое определение домена по названию должности.
    Для задачи достаточно выделить data_analytics и всё остальное.
    """
    if not isinstance(role, str):
        return "other"
    r = role.lower()
    # любые аналитикующие роли
    if "data analyst" in r or "аналитик данных" in r or "bi analyst" in r:
        return "data_analytics"
    if "analyst" in r or "аналитик" in r:
        # можно тоже считать data_analytics, если хочешь быть щедрым
        return "data_analytics"
    if "developer" in r or "разработчик" in r or "engineer" in r:
        return "software_engineering"
    return "other"


def enrich_career_history_with_nlp(dfs, target_domain: str = "data_analytics"):
    """
    Добавляем:
      - role_domain_nlp (на основе role_title)
      - entered_target_domain: bool на уровне студента
      - first_target_domain_date: дата первого входа в target_domain
    """
    ch = dfs["career_history"].copy()

    ch["role_domain_nlp"] = ch["role_title"].apply(role_to_domain)

    # Считаем переходы по студентам
    transitions = []

    for sid, group in ch.sort_values("event_date").groupby("student_id"):
        # исходный домен по первой записи
        first_row = group.iloc[0]
        initial_domain = first_row["role_domain_nlp"]

        # все события в целевом домене после начала
        in_target = group[group["role_domain_nlp"] == target_domain]

        if len(in_target) > 0 and initial_domain != target_domain:
            entered = True
            first_date = in_target["event_date"].min()
        else:
            entered = False
            first_date = pd.NaT

        transitions.append({
            "student_id": sid,
            "initial_domain_nlp": initial_domain,
            "entered_target_domain": entered,
            "first_target_domain_date": first_date
        })

    df_transitions = pd.DataFrame(transitions)

    dfs["career_history"] = ch
    dfs["student_domain_transitions"] = df_transitions
    return dfs


def run_nlp_step2(dfs):
    """
    Выполняет Шаг 2 (NLP) над уже очищенными dfs:
    - извлекает skills для курсов и вакансий (skillNER)
    - обогащает social_signals: sentiment + extracted_role_title/company
    - обогащает career_history: role_domain_nlp + transitions в data_analytics
    """
    nlp_en, nlp_ru, skill_extractor = init_nlp()

    dfs = nlp_extract_course_and_vacancy_skills(dfs, nlp_en, skill_extractor)
    dfs = nlp_process_social_signals(dfs, nlp_ru)
    dfs = enrich_career_history_with_nlp(dfs, target_domain="data_analytics")

    return dfs


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


# # пример использования:
# dfs = load_and_clean_data(".")
# dfs = run_nlp_step2(dfs)
# print(dfs["courses"].head())
# print(dfs["vacancies"].head())
# print(dfs["social_signals"].head())
# print(dfs["career_history"].head())
# print(dfs["student_domain_transitions"].head())
