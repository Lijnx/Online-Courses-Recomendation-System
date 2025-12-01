from data_handler import load_and_clean_data
from nlp_step import run_nlp_step
from compute_metrics import compute_student_course_outcomes, compute_course_effectiveness_metrics
from ranking import ask_user_profile_cli, build_course_feature_matrix
from ranking import (
    ask_user_profile_cli,
    train_xgboost_ranker,
    rank_courses_for_user,
)


dfs = load_and_clean_data("data")
dfs = run_nlp_step(dfs)
dfs = compute_student_course_outcomes(dfs, target_domain="data_analytics")
dfs = compute_course_effectiveness_metrics(dfs)

# print(dfs["course_effectiveness"].sort_values("success_rate", ascending=False).head())

# dfs["course_effectiveness"].to_csv("test.csv")

model, feat_cols = train_xgboost_ranker(dfs, goal_domain="data_analytics")


# 3. Спрашиваем пользователя
user_profile = ask_user_profile_cli(default_goal_domain="data_analytics")

# 4. Ранжируем курсы
df_ranked = rank_courses_for_user(dfs, model, user_profile, feature_columns=feat_cols, top_n=10)

print("\n=== Топ-10 курсов под ваш профиль ===")
print(df_ranked[["course_id", "title", "score", "success_rate", "skill_overlap_with_goal"]])


# # 1) спрашиваем пользователя
# user_profile = ask_user_profile_cli()
# print(user_profile)

# # 2) строим матрицу признаков для ранжирования
# df_features = build_course_feature_matrix(dfs, user_profile)

# print("\n=== Признаки курсов для данного пользователя ===")
# print(df_features.head())
# df_features.to_csv("df_features.csv")