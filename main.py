from data_handler import load_and_clean_data
from nlp_step import run_nlp_step
from compute_metrics import compute_student_course_outcomes, compute_course_effectiveness_metrics
from ranking import train_xgboost_ranker, rank_courses_for_user

from ollama_client import OllamaClient
from dialog_manager import collect_user_profile_dialog
from explanations import explain_top_courses

dfs = load_and_clean_data("data")
dfs = run_nlp_step(dfs)
dfs = compute_student_course_outcomes(dfs, target_domain="data_analytics")
dfs = compute_course_effectiveness_metrics(dfs)

# 1) обучаем ранкер
model, feat_cols = train_xgboost_ranker(dfs, goal_domain="data_analytics")

# 2) LLM (Ollama)
llm = OllamaClient(model="llama3.1:8b")

# 3) интерактивный диалог -> профиль
user_profile = collect_user_profile_dialog(llm)

# 4) ранжирование
df_ranked = rank_courses_for_user(dfs, model, user_profile, feature_columns=feat_cols, top_n=5)

print("\n=== Топ-5 курсов под ваш профиль ===")
print(df_ranked[["course_id", "title", "score", "success_rate", "skill_overlap_with_goal"]])

# 5) Шаг 7: объяснения для каждого курса
texts = explain_top_courses(llm, dfs, user_profile, df_ranked)

print("\n=== Объяснения рекомендаций ===")
for i, (row, txt) in enumerate(zip(df_ranked.to_dict("records"), texts), start=1):
    print(f"\n[{i}] {row.get('title')} ({row.get('course_id')})")
    print(txt)
