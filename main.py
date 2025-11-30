from data_handler import load_and_clean_data
from nlp_step import run_nlp_step
from compute_metrics import compute_student_course_outcomes, compute_course_effectiveness_metrics

dfs = load_and_clean_data("data")
dfs = run_nlp_step(dfs)
dfs = compute_student_course_outcomes(dfs, target_domain="data_analytics")
dfs = compute_course_effectiveness_metrics(dfs)

print(dfs["course_effectiveness"].sort_values("success_rate", ascending=False).head())
