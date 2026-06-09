import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path("applications.db")

DEFAULT_USER_NAME = "Default Local User"


CREATE_APPLICATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT,
    job_title TEXT,
    location TEXT,
    job_url TEXT,
    fit_score INTEGER,
    recommendation TEXT,
    status TEXT,
    notes TEXT,
    created_at TEXT
)
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    created_at TEXT
)
"""

CREATE_COMPANIES_TABLE = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    created_at TEXT
)
"""

CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER,
    title TEXT,
    location TEXT,
    job_url TEXT,
    job_description TEXT,
    created_at TEXT,
    FOREIGN KEY (company_id) REFERENCES companies(id)
)
"""

CREATE_OUTCOMES_TABLE = """
CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER,
    outcome_status TEXT,
    updated_at TEXT,
    FOREIGN KEY (application_id) REFERENCES applications(id)
)
"""

CREATE_MODEL_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    company_id INTEGER,
    job_id INTEGER,
    model_name TEXT,
    fit_score INTEGER,
    skill_match INTEGER,
    experience_match INTEGER,
    domain_match INTEGER,
    career_goal_alignment INTEGER,
    growth_potential INTEGER,
    missing_skills TEXT,
    analysis_text TEXT,
    resume_tailoring_text TEXT,
    networking_messages_text TEXT,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (company_id) REFERENCES companies(id),
    FOREIGN KEY (job_id) REFERENCES jobs(id)
)
"""

CREATE_USER_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    model_run_id INTEGER,
    usefulness_rating INTEGER,
    used_resume_bullets TEXT,
    used_networking_message TEXT,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(id)
)
"""

CREATE_JOB_QUEUE_TABLE = """
CREATE TABLE IF NOT EXISTS job_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT,
    job_title TEXT,
    title TEXT,
    location TEXT,
    short_description TEXT,
    source TEXT,
    required_skills TEXT,
    years_experience TEXT,
    visa_sponsorship_hints TEXT,
    application_platform TEXT,
    fit_score INTEGER,
    apply_decision TEXT,
    decision_reason TEXT,
    reason TEXT,
    job_url TEXT,
    status TEXT,
    jd_text TEXT,
    analysis_text TEXT,
    resume_bullets TEXT,
    cover_letter TEXT,
    recruiter_message TEXT,
    application_checklist TEXT,
    pre_filter_score INTEGER,
    filter_stage TEXT,
    filtered_out_reason TEXT,
    sent_to_llm INTEGER,
    created_at TEXT,
    updated_at TEXT
)
"""

CREATE_CAREER_PROFILE_TABLE = """
CREATE TABLE IF NOT EXISTS career_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT,
    target_roles TEXT,
    visa_status TEXT,
    preferred_locations TEXT,
    salary_goal TEXT,
    years_experience TEXT,
    career_goal TEXT
)
"""

APPLICATION_COLUMN_DEFAULTS = {
    "user_id": "INTEGER",
    "company_id": "INTEGER",
    "job_id": "INTEGER",
    "outcome_status": "TEXT",
    "skill_match": "INTEGER",
    "experience_match": "INTEGER",
    "domain_match": "INTEGER",
    "career_goal_alignment": "INTEGER",
    "growth_potential": "INTEGER",
    "priority": "TEXT",
    "application_date": "TEXT",
    "follow_up_date": "TEXT",
    "recruiter_name": "TEXT",
    "next_action": "TEXT",
    "interview_stage": "TEXT",
}

JOB_QUEUE_COLUMN_DEFAULTS = {
    "job_title": "TEXT",
    "short_description": "TEXT",
    "source": "TEXT",
    "decision_reason": "TEXT",
    "pre_filter_score": "INTEGER",
    "filter_stage": "TEXT",
    "filtered_out_reason": "TEXT",
    "sent_to_llm": "INTEGER",
    "updated_at": "TEXT",
}


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def ensure_application_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(applications)").fetchall()
    }
    for column_name, column_type in APPLICATION_COLUMN_DEFAULTS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE applications ADD COLUMN {column_name} {column_type}")


def ensure_job_queue_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(job_queue)").fetchall()
    }
    for column_name, column_type in JOB_QUEUE_COLUMN_DEFAULTS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE job_queue ADD COLUMN {column_name} {column_type}")


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(CREATE_USERS_TABLE)
        conn.execute(CREATE_COMPANIES_TABLE)
        conn.execute(CREATE_JOBS_TABLE)
        conn.execute(CREATE_APPLICATIONS_TABLE)
        ensure_application_columns(conn)
        conn.execute(CREATE_OUTCOMES_TABLE)
        conn.execute(CREATE_MODEL_RUNS_TABLE)
        conn.execute(CREATE_USER_FEEDBACK_TABLE)
        conn.execute(CREATE_JOB_QUEUE_TABLE)
        ensure_job_queue_columns(conn)
        conn.execute(CREATE_CAREER_PROFILE_TABLE)
        conn.execute(
            """
            INSERT OR IGNORE INTO users (id, name, created_at)
            VALUES (1, ?, datetime('now'))
            """,
            (DEFAULT_USER_NAME,),
        )
        conn.commit()


def insert_application(application: dict) -> None:
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO applications (
                company,
                user_id,
                company_id,
                job_id,
                job_title,
                location,
                job_url,
                fit_score,
                skill_match,
                experience_match,
                domain_match,
                career_goal_alignment,
                growth_potential,
                recommendation,
                priority,
                status,
                outcome_status,
                application_date,
                follow_up_date,
                recruiter_name,
                next_action,
                interview_stage,
                notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                application["company"],
                application["user_id"],
                application["company_id"],
                application["job_id"],
                application["job_title"],
                application["location"],
                application["job_url"],
                application["fit_score"],
                application["skill_match"],
                application["experience_match"],
                application["domain_match"],
                application["career_goal_alignment"],
                application["growth_potential"],
                application["recommendation"],
                application["priority"],
                application["status"],
                application["outcome_status"],
                application["application_date"],
                application["follow_up_date"],
                application["recruiter_name"],
                application["next_action"],
                application["interview_stage"],
                application["notes"],
                application["created_at"],
            ),
        )
        conn.commit()
        return cursor.lastrowid


def fetch_applications() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                id,
                company,
                user_id,
                company_id,
                job_id,
                job_title,
                location,
                job_url,
                fit_score,
                skill_match,
                experience_match,
                domain_match,
                career_goal_alignment,
                growth_potential,
                recommendation,
                priority,
                status,
                outcome_status,
                application_date,
                follow_up_date,
                recruiter_name,
                next_action,
                interview_stage,
                notes,
                created_at
            FROM applications
            ORDER BY
                CASE priority
                    WHEN 'Must Apply' THEN 1
                    WHEN 'Good Opportunity' THEN 2
                    WHEN 'Low Priority' THEN 3
                    ELSE 4
                END,
                fit_score DESC,
                created_at DESC,
                id DESC
            """,
            conn,
        )


def upsert_career_profile(profile: dict) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO career_profile (
                id,
                name,
                target_roles,
                visa_status,
                preferred_locations,
                salary_goal,
                years_experience,
                career_goal
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                target_roles = excluded.target_roles,
                visa_status = excluded.visa_status,
                preferred_locations = excluded.preferred_locations,
                salary_goal = excluded.salary_goal,
                years_experience = excluded.years_experience,
                career_goal = excluded.career_goal
            """,
            (
                profile["name"],
                profile["target_roles"],
                profile["visa_status"],
                profile["preferred_locations"],
                profile["salary_goal"],
                profile["years_experience"],
                profile["career_goal"],
            ),
        )
        conn.commit()


def fetch_career_profile() -> dict:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                name,
                target_roles,
                visa_status,
                preferred_locations,
                salary_goal,
                years_experience,
                career_goal
            FROM career_profile
            WHERE id = 1
            """
        ).fetchone()

    if not row:
        return {
            "name": "",
            "target_roles": "",
            "visa_status": "",
            "preferred_locations": "",
            "salary_goal": "",
            "years_experience": "",
            "career_goal": "",
        }

    return {
        "name": row[0] or "",
        "target_roles": row[1] or "",
        "visa_status": row[2] or "",
        "preferred_locations": row[3] or "",
        "salary_goal": row[4] or "",
        "years_experience": row[5] or "",
        "career_goal": row[6] or "",
    }
