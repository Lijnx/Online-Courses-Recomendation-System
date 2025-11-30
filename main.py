from data_handler import load_and_clean_data
from nlp_step import run_nlp_step2, compute_student_course_outcomes

dfs = load_and_clean_data("data")
dfs = run_nlp_step2(dfs)
dfs = compute_student_course_outcomes(dfs)

print(dfs["student_course_outcomes"].head())
