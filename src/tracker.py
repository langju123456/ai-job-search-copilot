from datetime import datetime
from typing import Optional

from src.database import fetch_applications, init_db, insert_application


STATUS_OPTIONS = ["Saved", "Applied", "Networking", "Interview", "Rejected", "Offer"]


def save_application(
    company: str,
    job_title: str,
    location: str,
    job_url: str,
    fit_score: int,
    recommendation: str,
    status: str,
    notes: str,
    score_breakdown: Optional[dict] = None,
    priority: str = "",
    application_date: str = "",
    follow_up_date: str = "",
    recruiter_name: str = "",
    next_action: str = "",
    interview_stage: str = "",
    user_id: int = 1,
    company_id: int = 0,
    job_id: int = 0,
    outcome_status: str = "No Response",
) -> int:
    score_breakdown = score_breakdown or {}
    return insert_application(
        {
            "company": company,
            "user_id": user_id,
            "company_id": company_id,
            "job_id": job_id,
            "job_title": job_title,
            "location": location,
            "job_url": job_url,
            "fit_score": fit_score,
            "skill_match": score_breakdown.get("skill_match", 0),
            "experience_match": score_breakdown.get("experience_match", 0),
            "domain_match": score_breakdown.get("domain_match", 0),
            "career_goal_alignment": score_breakdown.get("career_goal_alignment", 0),
            "growth_potential": score_breakdown.get("growth_potential", 0),
            "recommendation": recommendation,
            "priority": priority,
            "status": status,
            "outcome_status": outcome_status,
            "application_date": application_date,
            "follow_up_date": follow_up_date,
            "recruiter_name": recruiter_name,
            "next_action": next_action,
            "interview_stage": interview_stage,
            "notes": notes,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )


def get_applications():
    init_db()
    return fetch_applications()
