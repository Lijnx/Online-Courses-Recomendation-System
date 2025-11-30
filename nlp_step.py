import re
import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB


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


def run_nlp_step(dfs):
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
