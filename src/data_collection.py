from datetime import datetime

import pandas as pd

from src.database import get_connection, init_db
from src.llm import get_openai_model


DEFAULT_USER_ID = 1
OUTCOME_STATUS_OPTIONS = ["No Response", "Rejected", "Interview", "Offer"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_or_create_company(company_name: str) -> int:
    init_db()
    clean_name = company_name.strip() or "Unknown Company"
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO companies (name, created_at)
            VALUES (?, ?)
            """,
            (clean_name, now_iso()),
        )
        company_id = conn.execute(
            "SELECT id FROM companies WHERE name = ?", (clean_name,)
        ).fetchone()[0]
        conn.commit()
    return company_id


def create_job(
    company_id: int,
    job_title: str,
    location: str,
    job_url: str,
    job_description: str,
) -> int:
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO jobs (
                company_id,
                title,
                location,
                job_url,
                job_description,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                job_title.strip() or "Untitled Job",
                location.strip(),
                job_url.strip(),
                job_description,
                now_iso(),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def extract_missing_skills(analysis: str) -> list:
    marker = "## Missing Skills"
    if marker not in analysis:
        return []

    section = analysis.split(marker, 1)[1]
    if "##" in section:
        section = section.split("##", 1)[0]

    skills = []
    for line in section.splitlines():
        clean_line = line.strip()
        if clean_line.startswith("-"):
            skill = clean_line.lstrip("-").strip()
            if skill:
                skills.append(skill)
    return skills


def create_model_run(
    user_id: int,
    company_id: int,
    job_id: int,
    fit_score: int,
    score_breakdown: dict,
    analysis_text: str,
    resume_tailoring_text: str,
    networking_messages_text: str,
    analysis_profile_hash: str = "",
) -> int:
    init_db()
    missing_skills = ", ".join(extract_missing_skills(analysis_text))
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_runs (
                user_id,
                company_id,
                job_id,
                model_name,
                fit_score,
                skill_match,
                experience_match,
                domain_match,
                career_goal_alignment,
                growth_potential,
                missing_skills,
                analysis_text,
                resume_tailoring_text,
                networking_messages_text,
                analysis_profile_hash,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                company_id,
                job_id,
                get_openai_model(),
                fit_score,
                score_breakdown.get("skill_match", 0),
                score_breakdown.get("experience_match", 0),
                score_breakdown.get("domain_match", 0),
                score_breakdown.get("career_goal_alignment", 0),
                score_breakdown.get("growth_potential", 0),
                missing_skills,
                analysis_text,
                resume_tailoring_text,
                networking_messages_text,
                analysis_profile_hash,
                now_iso(),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def record_analysis_run(
    company_name: str,
    job_title: str,
    location: str,
    job_url: str,
    job_description: str,
    fit_score: int,
    score_breakdown: dict,
    analysis_text: str,
    resume_tailoring_text: str,
    networking_messages_text: str,
    analysis_profile_hash: str = "",
) -> dict:
    company_id = get_or_create_company(company_name)
    job_id = create_job(company_id, job_title, location, job_url, job_description)
    model_run_id = create_model_run(
        DEFAULT_USER_ID,
        company_id,
        job_id,
        fit_score,
        score_breakdown,
        analysis_text,
        resume_tailoring_text,
        networking_messages_text,
        analysis_profile_hash,
    )
    return {
        "user_id": DEFAULT_USER_ID,
        "company_id": company_id,
        "job_id": job_id,
        "model_run_id": model_run_id,
    }


def create_outcome(application_id: int, outcome_status: str) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO outcomes (application_id, outcome_status, updated_at)
            VALUES (?, ?, ?)
            """,
            (application_id, outcome_status, now_iso()),
        )
        conn.execute(
            """
            UPDATE applications
            SET outcome_status = ?
            WHERE id = ?
            """,
            (outcome_status, application_id),
        )
        conn.commit()


def update_outcome(application_id: int, outcome_status: str) -> None:
    create_outcome(application_id, outcome_status)


def save_user_feedback(
    model_run_id: int,
    usefulness_rating: int,
    used_resume_bullets: str,
    used_networking_message: str,
) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_feedback (
                user_id,
                model_run_id,
                usefulness_rating,
                used_resume_bullets,
                used_networking_message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                DEFAULT_USER_ID,
                model_run_id,
                usefulness_rating,
                used_resume_bullets,
                used_networking_message,
                now_iso(),
            ),
        )
        conn.commit()


def fetch_analytics_summary() -> dict:
    init_db()
    with get_connection() as conn:
        total_analyzed_jobs = conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
        total_applications = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
        interviews = conn.execute(
            """
            SELECT COUNT(*)
            FROM applications
            WHERE outcome_status = 'Interview' OR status = 'Interview'
            """
        ).fetchone()[0]
        offers = conn.execute(
            """
            SELECT COUNT(*)
            FROM applications
            WHERE outcome_status = 'Offer' OR status = 'Offer'
            """
        ).fetchone()[0]

    interview_rate = interviews / total_applications if total_applications else 0
    offer_rate = offers / total_applications if total_applications else 0
    return {
        "total_analyzed_jobs": total_analyzed_jobs,
        "total_applications": total_applications,
        "interview_rate": interview_rate,
        "offer_rate": offer_rate,
    }


def fetch_average_fit_score_by_outcome() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                COALESCE(NULLIF(outcome_status, ''), status, 'No Response') AS outcome,
                AVG(fit_score) AS average_fit_score,
                COUNT(*) AS application_count
            FROM applications
            GROUP BY COALESCE(NULLIF(outcome_status, ''), status, 'No Response')
            ORDER BY average_fit_score DESC
            """,
            conn,
        )


def fetch_top_scoring_companies() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                companies.name AS company,
                AVG(model_runs.fit_score) AS average_fit_score,
                COUNT(model_runs.id) AS analyzed_jobs
            FROM model_runs
            JOIN companies ON companies.id = model_runs.company_id
            GROUP BY companies.id, companies.name
            ORDER BY average_fit_score DESC, analyzed_jobs DESC
            LIMIT 10
            """,
            conn,
        )


def fetch_most_common_missing_skills() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT missing_skills
            FROM model_runs
            WHERE missing_skills IS NOT NULL AND missing_skills != ''
            """
        ).fetchall()

    counts = {}
    for (missing_skills,) in rows:
        for skill in missing_skills.split(","):
            clean_skill = skill.strip()
            if clean_skill:
                counts[clean_skill] = counts.get(clean_skill, 0) + 1

    return pd.DataFrame(
        [{"missing_skill": skill, "count": count} for skill, count in counts.items()]
    ).sort_values("count", ascending=False).head(10) if counts else pd.DataFrame(
        columns=["missing_skill", "count"]
    )
