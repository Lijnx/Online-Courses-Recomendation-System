from typing import Dict, Any, List, Optional
import json
import re

from ollama_client import OllamaClient

REQUIRED_FIELDS = [
    "current_level",
    "hours_per_week",
    "budget",
    "need_certificate",
    "time_horizon_months",
    "goal_domain",
]

DEFAULT_PROFILE: Dict[str, Any] = {
    "current_level": None,
    "hours_per_week": None,
    "budget": None,
    "need_certificate": None,
    "time_horizon_months": None,
    "goal_domain": "data_analytics",
}

LEVELS = {"beginner", "intermediate", "advanced"}


def _safe_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"true", "yes", "y", "да", "нужен", "нужно"}:
            return True
        if t in {"false", "no", "n", "нет", "не нужен", "не нужно"}:
            return False
    return None


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def _normalize_level(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    t = v.strip().lower()
    # русские подсказки
    if t in {"новичок", "начальный", "beginner"}:
        return "beginner"
    if t in {"средний", "intermediate", "middle"}:
        return "intermediate"
    if t in {"продвинутый", "advanced", "senior"}:
        return "advanced"
    if t in LEVELS:
        return t
    return None


def _merge_profile(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if k not in out:
            continue
        if v is None:
            continue
        out[k] = v
    return out


def _missing_fields(profile: Dict[str, Any]) -> List[str]:
    return [k for k in REQUIRED_FIELDS if profile.get(k) is None]


def _regex_fallback_extract(text: str) -> Dict[str, Any]:
    """
    Минимальный fallback без LLM (на случай, если LLM вернул мусор):
    пытаемся достать числа и "сертификат" из текста.
    """
    t = text.lower()
    upd: Dict[str, Any] = {}

    # hours per week (например: "10 часов", "15h/week")
    m = re.search(r"(\d{1,2})\s*(час|hours|h)\b", t)
    if m:
        upd["hours_per_week"] = _safe_int(m.group(1))

    # budget (например: "до 400", "budget 300", "300$")
    m = re.search(r"(?:до|budget|бюджет)\s*[:=]?\s*(\d{2,5})", t)
    if m:
        upd["budget"] = _safe_float(m.group(1))

    # horizon months (например: "за 6 месяцев")
    m = re.search(r"(\d{1,2})\s*(?:мес|месяц|months)\b", t)
    if m:
        upd["time_horizon_months"] = _safe_float(m.group(1))

    # certificate
    if "сертифик" in t:
        if any(x in t for x in ["не нужен", "не важно", "без сертифик"]):
            upd["need_certificate"] = False
        else:
            upd["need_certificate"] = True

    # level
    if any(x in t for x in ["новичок", "с нуля", "начальный"]):
        upd["current_level"] = "beginner"
    if any(x in t for x in ["средний", "есть опыт"]):
        upd["current_level"] = "intermediate"
    if any(x in t for x in ["продвинутый"]):
        upd["current_level"] = "advanced"

    # goal
    if any(x in t for x in ["аналитик", "data analyst", "data analytics"]):
        upd["goal_domain"] = "data_analytics"

    return upd


def llm_extract_profile(llm: OllamaClient, user_text: str, current_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Просим LLM вернуть ТОЛЬКО JSON с полями профиля (частично).
    """
    system = (
        "Ты парсер пользовательского запроса для системы рекомендаций курсов.\n"
        "Извлеки параметры профиля пользователя из текста.\n"
        "Верни ТОЛЬКО валидный JSON без пояснений.\n"
        "Разрешённые ключи: current_level, hours_per_week, budget, need_certificate, time_horizon_months, goal_domain.\n"
        "current_level ∈ {beginner, intermediate, advanced}.\n"
        "hours_per_week: int.\n"
        "budget: float.\n"
        "need_certificate: bool.\n"
        "time_horizon_months: float.\n"
        "goal_domain: строка (например data_analytics).\n"
        "Если значение не указано — не включай ключ или ставь null."
    )

    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Текущий профиль: {json.dumps(current_profile, ensure_ascii=False)}\nТекст: {user_text}"},
    ]

    raw = llm.chat(msg, temperature=0.0)

    try:
        obj = json.loads(raw)
    except Exception:
        # fallback regex
        obj = _regex_fallback_extract(user_text)

    # валидация/нормализация
    upd: Dict[str, Any] = {}

    if "current_level" in obj:
        upd["current_level"] = _normalize_level(obj.get("current_level"))

    if "hours_per_week" in obj:
        v = _safe_int(obj.get("hours_per_week"))
        upd["hours_per_week"] = v if v is not None and 0 < v <= 60 else None

    if "budget" in obj:
        v = _safe_float(obj.get("budget"))
        upd["budget"] = v if v is not None and v >= 0 else None

    if "need_certificate" in obj:
        upd["need_certificate"] = _safe_bool(obj.get("need_certificate"))

    if "time_horizon_months" in obj:
        v = _safe_float(obj.get("time_horizon_months"))
        upd["time_horizon_months"] = v if v is not None and 0 < v <= 48 else None

    if "goal_domain" in obj and isinstance(obj.get("goal_domain"), str):
        upd["goal_domain"] = obj["goal_domain"].strip()

    # если LLM не вернул ключи, fallback regex может вернуть часть
    fallback = _regex_fallback_extract(user_text)
    upd = _merge_profile(upd, fallback)

    return upd


def llm_generate_followup_question(llm: OllamaClient, profile: Dict[str, Any], missing: List[str]) -> str:
    """
    Генерируем ОДИН уточняющий вопрос за раз, максимально коротко.
    """
    system = (
        "Ты помощник-опросник. Твоя задача — уточнить недостающие параметры профиля пользователя.\n"
        "Задавай ОДИН короткий вопрос за раз, без списков и лишнего текста.\n"
        "Сфокусируйся на самом важном из missing_fields."
    )

    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Профиль: {json.dumps(profile, ensure_ascii=False)}\nmissing_fields: {missing}"},
    ]
    q = llm.chat(msg, temperature=0.2)
    return q.strip() if q.strip() else "Сколько часов в неделю вы готовы учиться?"


def collect_user_profile_dialog(llm: OllamaClient) -> Dict[str, Any]:
    """
    Полноценный интерактивный диалог:
    1) первое свободное сообщение пользователя
    2) LLM извлекает поля
    3) пока не хватает — задаём уточняющие вопросы
    """
    profile = dict(DEFAULT_PROFILE)

    print("Опишите одним сообщением вашу цель и ограничения (например: хочу стать аналитиком данных, 10 ч/нед, бюджет до 400, нужен сертификат).")
    first = input("Вы: ").strip()

    profile = _merge_profile(profile, llm_extract_profile(llm, first, profile))

    while True:
        missing = _missing_fields(profile)
        if not missing:
            break

        q = llm_generate_followup_question(llm, profile, missing)
        ans = input(f"{q}\nВы: ").strip()
        profile = _merge_profile(profile, llm_extract_profile(llm, ans, profile))

    print("\nИтоговый профиль пользователя:")
    for k, v in profile.items():
        print(f"  {k}: {v}")

    return profile
