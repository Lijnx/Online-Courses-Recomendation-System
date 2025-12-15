from typing import Dict, Any, List
import json
import pandas as pd

from ollama_client import OllamaClient


def build_alumni_examples(dfs, course_id: str, target_domain: str = "data_analytics", n: int = 3) -> List[Dict[str, Any]]:
    """
    Достаём 2–3 синтетических примера выпускников для конкретного курса:
    prev_role -> new_role, time_to_job, salary_change, ссылка на проект (final_project_url).
    """
    enroll = dfs["enrollments"].copy()
    students = dfs["students"].copy()
    career = dfs["career_history"].copy()
    outcomes = dfs.get("student_course_outcomes", pd.DataFrame()).copy()

    # только завершившие курс
    enr = enroll[(enroll["course_id"] == course_id) & (enroll["completion_status"] == "completed")].copy()
    if enr.empty:
        return []

    # если есть outcomes — предпочтём успешных, иначе просто первых
    if not outcomes.empty:
        oc = outcomes[outcomes["course_id"] == course_id].copy()
        oc = oc.sort_values(["success_label", "time_to_target_job_months"], ascending=[False, True])
        student_order = oc["student_id"].dropna().unique().tolist()
    else:
        student_order = enr["student_id"].dropna().unique().tolist()

    examples: List[Dict[str, Any]] = []
    for sid in student_order:
        if len(examples) >= n:
            break

        stud = students[students["student_id"] == sid]
        if stud.empty:
            continue
        prev_role = stud.iloc[0].get("prev_role_title")

        # дата окончания курса
        row_enr = enr[enr["student_id"] == sid]
        if row_enr.empty:
            continue
        comp_date = row_enr.iloc[0].get("completion_date")
        proj_url = row_enr.iloc[0].get("final_project_url")

        # ближайшая карьерная запись в target_domain после completion_date
        ch = career[career["student_id"] == sid].copy()
        ch = ch.sort_values("event_date")
        if pd.notna(comp_date):
            ch_after = ch[ch["event_date"] > comp_date]
        else:
            ch_after = ch

        target = ch_after[ch_after.get("domain") == target_domain]
        if target.empty:
            # если нет target_domain, возьмём просто следующую запись
            target = ch_after

        if target.empty:
            continue

        target_row = target.iloc[0]
        new_role = target_row.get("role_title")
        new_salary = target_row.get("salary_band")

        # time_to_job (если есть outcomes — берём оттуда)
        time_to_job = None
        salary_change = None
        if not outcomes.empty:
            oc_row = outcomes[(outcomes["student_id"] == sid) & (outcomes["course_id"] == course_id)]
            if not oc_row.empty:
                time_to_job = oc_row.iloc[0].get("time_to_target_job_months")
                salary_change = oc_row.iloc[0].get("salary_change")

        examples.append({
            "student_id": sid,
            "prev_role": prev_role,
            "new_role": new_role,
            "time_to_job_months": None if pd.isna(time_to_job) else float(time_to_job) if time_to_job is not None else None,
            "salary_change_levels": None if pd.isna(salary_change) else int(salary_change) if salary_change is not None else None,
            "project_url": proj_url if isinstance(proj_url, str) else f"https://github.com/fake/{sid.lower()}_{course_id.lower()}",
        })

    return examples


def explain_course_recommendation(
    llm: OllamaClient,
    user_profile: Dict[str, Any],
    course_row: Dict[str, Any],
    course_metrics: Dict[str, Any],
    alumni_examples: List[Dict[str, Any]],
) -> str:
    """
    Генерим текст объяснения через LLM.
    """
    system = (
        "Ты ассистент, который объясняет рекомендации онлайн-курсов.\n"
        "Пиши кратко, понятно, по делу, на русском.\n"
        "Структура:\n"
        "1) Почему курс подходит пользователю\n"
        "2) Метрики эффективности (проценты/медианы)\n"
        "3) 2-3 примера выпускников (prev_role -> new_role, сроки, ссылка)\n"
        "4) 1 строка 'кому особенно подойдёт'\n"
        "Не выдумывай цифры: используй только данные из контекста. Если чего-то нет — не пиши."
    )

    context = {
        "user_profile": user_profile,
        "course": course_row,
        "course_metrics": course_metrics,
        "alumni_examples": alumni_examples,
    }

    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Контекст (JSON):\n{json.dumps(context, ensure_ascii=False)}"},
    ]

    return llm.chat(msg, temperature=0.3)


def explain_top_courses(llm: OllamaClient, dfs, user_profile: Dict[str, Any], df_ranked) -> List[str]:
    """
    Для каждой строки df_ranked возвращает текст объяснения.
    """
    eff = dfs["course_effectiveness"].copy()

    explanations: List[str] = []
    for _, r in df_ranked.iterrows():
        cid = r["course_id"]

        # метрики курса (шаг 4)
        m = eff[eff["course_id"] == cid]
        course_metrics = m.iloc[0].to_dict() if not m.empty else {}

        # "сырые" поля из ранжирования
        course_row = r.to_dict()

        alumni = build_alumni_examples(dfs, cid, target_domain=user_profile.get("goal_domain", "data_analytics"), n=3)

        text = explain_course_recommendation(
            llm=llm,
            user_profile=user_profile,
            course_row=course_row,
            course_metrics=course_metrics,
            alumni_examples=alumni,
        )
        explanations.append(text)

    return explanations
