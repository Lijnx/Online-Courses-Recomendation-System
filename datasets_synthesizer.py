import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ---------- Helper functions ----------

def random_date(start_date, end_date):
    delta = end_date - start_date
    days = random.randint(0, delta.days)
    return start_date + timedelta(days=days)

def salary_band():
    return random.choice(["low", "medium", "high"])

# ---------- 1. Courses ----------

course_titles = [
    "Intro to Data Analytics",
    "Python for Data Analysis",
    "SQL for Analytics",
    "Statistics for Data Science",
    "Data Visualization with Tableau",
    "Excel to Analytics Transition",
    "Product Analytics Fundamentals",
    "A/B Testing in Practice",
    "Business Intelligence with Power BI",
    "Machine Learning Basics",
    "Advanced SQL for Analysts",
    "Data Storytelling and Dashboards",
    "Practical Data Projects",
    "Analytics for Marketing",
    "Analytics for Finance",
    "Python & SQL Bootcamp",
    "Data Analytics Career Bootcamp",
    "ETL and Data Pipelines",
    "Experiment Design for Analysts",
    "Interview Preparation for Data Roles"
]

providers = ["DataCampX", "AnalyticaSchool", "PracticumLab", "InsightHub"]

course_levels = ["beginner", "intermediate", "advanced"]

skills_master = [
    "Python", "SQL", "Pandas", "NumPy", "Statistics", "Probability",
    "A/B testing", "Experiment design", "Tableau", "Power BI",
    "Excel", "Data visualization", "Dashboarding", "Git",
    "ETL", "Data pipelines", "Product analytics", "Marketing analytics",
    "Financial analytics", "Machine learning", "Communication"
]

courses = []
course_skills_rows = []

for i, title in enumerate(course_titles, start=1):
    course_id = f"C{i:03d}"
    duration_weeks = random.randint(4, 16)
    hours_per_week = random.choice([5, 7, 10, 12])
    price = random.choice([0, 99, 199, 299, 499])
    level = random.choice(course_levels)
    format_ = random.choice(["self-paced", "mentor-led", "bootcamp"])
    has_certificate = random.choice([True, False])
    provider = random.choice(providers)

    # Pick 4-7 skills for this course
    n_skills = random.randint(4, 7)
    course_skills = random.sample(skills_master, n_skills)

    courses.append({
        "course_id": course_id,
        "title": title,
        "provider": provider,
        "domain": "data_analytics",
        "level": level,
        "duration_weeks": duration_weeks,
        "hours_per_week": hours_per_week,
        "price": price,
        "format": format_,
        "has_certificate": has_certificate,
        "skills_taught": ", ".join(course_skills)
    })

    for skill in course_skills:
        course_skills_rows.append({
            "course_id": course_id,
            "skill_name": skill
        })

df_courses = pd.DataFrame(courses)
df_course_skills = pd.DataFrame(course_skills_rows)

# ---------- 2. Skills table ----------

df_skills = pd.DataFrame({
    "skill_id": [f"S{i:03d}" for i in range(1, len(skills_master) + 1)],
    "name": skills_master,
    "category": [
        "language" if s in ["Python", "SQL"]
        else "tool" if s in ["Tableau", "Power BI", "Excel", "Git"]
        else "concept"
        for s in skills_master
    ]
})

# Add skill_id to course_skills
skill_name_to_id = dict(zip(df_skills["name"], df_skills["skill_id"]))
df_course_skills["skill_id"] = df_course_skills["skill_name"].map(skill_name_to_id)

# ---------- 3. Students ----------

prev_roles = [
    "Waiter", "Sales Manager", "Economist", "Math Student",
    "Accountant", "Marketer", "Support Specialist", "Engineer",
    "HR Specialist", "Logistics Coordinator"
]

prev_domains = ["out_of_IT", "business", "IT_related"]

students = []
n_students = 50

for i in range(1, n_students + 1):
    student_id = f"S{i:03d}"
    age = random.randint(21, 40)
    country = random.choice(["RU", "UA", "KZ", "PL", "DE", "IN"])
    prev_role = random.choice(prev_roles)
    prev_domain = random.choice(prev_domains)
    prev_salary = salary_band()
    prev_skills = random.sample(
        ["Excel", "Basic SQL", "PowerPoint", "Python basics", "Customer support"],
        k=random.randint(1, 3)
    )
    students.append({
        "student_id": student_id,
        "age": age,
        "country": country,
        "prev_role_title": prev_role,
        "prev_domain": prev_domain,
        "prev_salary_band": prev_salary,
        "prev_skills": ", ".join(prev_skills)
    })

df_students = pd.DataFrame(students)

# ---------- 4. Enrollments / Completions ----------

start_period = datetime(2022, 1, 1)
end_period = datetime(2024, 6, 30)

enrollments = []

for student in df_students["student_id"]:
    # Each student enrolls in 1-3 courses
    course_ids = random.sample(list(df_courses["course_id"]), k=random.randint(1, 3))
    for cid in course_ids:
        enroll_date = random_date(start_period, end_period - timedelta(weeks=20))
        completion_status = random.choices(["completed", "dropped"], weights=[0.8, 0.2])[0]
        if completion_status == "completed":
            completion_date = enroll_date + timedelta(weeks=random.randint(4, 16))
        else:
            completion_date = None

        self_goal = random.choice([
            "become data analyst",
            "upskill in current job",
            "move to data science",
            "explore analytics"
        ])

        enrollments.append({
            "student_id": student,
            "course_id": cid,
            "enroll_date": enroll_date.date().isoformat(),
            "completion_date": completion_date.date().isoformat() if completion_date else None,
            "completion_status": completion_status,
            "final_project_url": f"https://github.com/user/{student.lower()}_{cid.lower()}_project"
            if completion_status == "completed" else None,
            "self_reported_goal": self_goal
        })

df_enrollments = pd.DataFrame(enrollments)

# ---------- 5. Career history ----------

career_history = []

for _, row in df_students.iterrows():
    sid = row["student_id"]
    # Pre-course event
    base_date = random_date(start_period - timedelta(days=365), start_period)
    career_history.append({
        "student_id": sid,
        "event_date": base_date.date().isoformat(),
        "role_title": row["prev_role_title"],
        "domain": row["prev_domain"],
        "salary_band": row["prev_salary_band"],
        "company_type": random.choice(["retail", "service", "corporate", "small_business"]),
        "is_first_job_in_domain": True
    })

    # Post-course: some students switch to data_analytics
    if random.random() < 0.7:  # 70% eventually have new event
        # Find latest completion date
        comp_dates = df_enrollments[
            (df_enrollments["student_id"] == sid) &
            (df_enrollments["completion_date"].notna())
        ]["completion_date"]

        if not comp_dates.empty:
            latest_completion = max(pd.to_datetime(comp_dates))
        else:
            latest_completion = random_date(start_period, end_period)

        # Job in target domain sometime after completion
        event_date = latest_completion + timedelta(weeks=random.randint(4, 24))
        success_domain = "data_analytics"
        level = random.choice(["Junior", "Middle"])
        new_role_title = f"{level} Data Analyst"
        new_salary = random.choice(["medium", "high"])
        career_history.append({
            "student_id": sid,
            "event_date": event_date.date().isoformat(),
            "role_title": new_role_title,
            "domain": success_domain,
            "salary_band": new_salary,
            "company_type": random.choice(["startup", "corporate", "agency"]),
            "is_first_job_in_domain": (row["prev_domain"] != success_domain)
        })

df_career_history = pd.DataFrame(career_history)

# ---------- 6. Social signals (posts / reviews) ----------

platforms = ["twitter", "telegram", "vk", "blog"]
social_signals = []
post_id_counter = 1

for _, enr in df_enrollments.iterrows():
    if enr["completion_status"] == "completed" and random.random() < 0.6:
        # 1-2 posts per completed course
        n_posts = random.randint(1, 2)
        for _ in range(n_posts):
            text_templates = [
                f"После курса {enr['course_id']} получил оффер на позицию Junior Data Analyst.",
                f"Курс {enr['course_id']} помог подтянуть SQL и Python, через 3 месяца вышел на новую работу.",
                f"Закончил {enr['course_id']}, сделал проект в портфолио и прошёл собес.",
                f"Курс {enr['course_id']} был полезен, но без практики работы не найти."
            ]
            text = random.choice(text_templates)
            post_date = (pd.to_datetime(enr["completion_date"]) +
                         timedelta(days=random.randint(7, 120))).date().isoformat()

            social_signals.append({
                "post_id": f"P{post_id_counter:04d}",
                "student_id": enr["student_id"],
                "course_id": enr["course_id"],
                "platform": random.choice(platforms),
                "date": post_date,
                "text": text
            })
            post_id_counter += 1

df_social_signals = pd.DataFrame(social_signals)

# ---------- 7. Repos (project portfolios) ----------

repos = []
repo_id_counter = 1

for _, enr in df_enrollments.iterrows():
    if enr["completion_status"] == "completed" and random.random() < 0.7:
        cid = enr["course_id"]
        sid = enr["student_id"]
        # Tech stack = some of the course skills
        cskills = df_course_skills[df_course_skills["course_id"] == cid]["skill_name"].tolist()
        tech_stack = random.sample(cskills, k=min(len(cskills), random.randint(2, 4)))
        repos.append({
            "repo_id": f"R{repo_id_counter:04d}",
            "student_id": sid,
            "course_id": cid,
            "tech_stack": ", ".join(tech_stack),
            "stars": random.randint(0, 25),
            "is_course_project": True
        })
        repo_id_counter += 1

df_repos = pd.DataFrame(repos)

# ---------- 8. Vacancies & vacancy_skills ----------

vacancy_titles = [
    "Junior Data Analyst",
    "Product Data Analyst",
    "Marketing Data Analyst",
    "BI Analyst",
    "Data Analytics Intern"
]

vacancies = []
vacancy_skills_rows = []

for i, vtitle in enumerate(vacancy_titles, start=1):
    vid = f"V{i:03d}"
    desc = f"{vtitle} role in data analytics focusing on SQL, dashboards and product insights."
    vacancies.append({
        "vacancy_id": vid,
        "title": vtitle,
        "description_text": desc,
        "target_domain": "data_analytics"
    })
    # Pick 5-8 skills per vacancy
    vskills = random.sample(skills_master, k=random.randint(5, 8))
    for s in vskills:
        vacancy_skills_rows.append({
            "vacancy_id": vid,
            "skill_id": skill_name_to_id[s],
            "skill_name": s
        })

df_vacancies = pd.DataFrame(vacancies)
df_vacancy_skills = pd.DataFrame(vacancy_skills_rows)

# ---------- 9. Student skills current (after courses) ----------

student_skills_current = []

for sid in df_students["student_id"]:
    # baseline: prev_skills mapped to main skill names where possible
    prev_row = df_students[df_students["student_id"] == sid].iloc[0]
    prev_skills_list = [s.strip() for s in prev_row["prev_skills"].split(",")]
    mapped_prev_skills = []
    for s in prev_skills_list:
        if "Excel" in s:
            mapped_prev_skills.append("Excel")
        elif "SQL" in s:
            mapped_prev_skills.append("SQL")
        elif "Python" in s:
            mapped_prev_skills.append("Python")
    # skills from completed courses
    completed_courses = df_enrollments[
        (df_enrollments["student_id"] == sid) &
        (df_enrollments["completion_status"] == "completed")
    ]["course_id"].tolist()

    course_skill_names = df_course_skills[
        df_course_skills["course_id"].isin(completed_courses)
    ]["skill_name"].unique().tolist()

    all_skills = set(mapped_prev_skills) | set(course_skill_names)
    for sk in all_skills:
        if sk in skill_name_to_id:
            student_skills_current.append({
                "student_id": sid,
                "skill_id": skill_name_to_id[sk],
                "skill_name": sk,
                "source": "course_or_prev"
            })

df_student_skills_current = pd.DataFrame(student_skills_current)

# ---------- Save to CSV in folder ./data ----------

# Создаём папку data, если она ещё не существует
os.makedirs("data", exist_ok=True)

filenames = {
    "courses.csv": df_courses,
    "course_skills.csv": df_course_skills,
    "skills.csv": df_skills,
    "students.csv": df_students,
    "enrollments.csv": df_enrollments,
    "career_history.csv": df_career_history,
    "social_signals.csv": df_social_signals,
    "repos.csv": df_repos,
    "vacancies.csv": df_vacancies,
    "vacancy_skills.csv": df_vacancy_skills,
    "student_skills_current.csv": df_student_skills_current,
}

for name, df in filenames.items():
    path = os.path.join("data", name)
    df.to_csv(path, index=False)

{ name: df.head() for name, df in filenames.items() }
