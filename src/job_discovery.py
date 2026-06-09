import json
import re
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

from src.database import get_connection, init_db
from src.job_analyzer import analyze_job
from src.llm import call_llm
from src.resume_tailor import tailor_resume
from src.scoring import parse_score_breakdown


JOB_QUEUE_STATUS_OPTIONS = [
    "Discovered",
    "Reviewed",
    "Ready to Apply",
    "Applied",
    "Skipped",
]

APPLY_DECISIONS = ["Apply", "Maybe", "Skip"]
REQUIRED_CSV_COLUMNS = ["company", "job_title", "location", "job_url", "jd_text"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def infer_application_platform(job_url: str) -> str:
    url = job_url.lower()
    if "linkedin.com" in url:
        return "LinkedIn"
    if "greenhouse.io" in url or "greenhouse" in url:
        return "Greenhouse"
    if "lever.co" in url or "lever" in url:
        return "Lever"
    if "workdayjobs.com" in url or "myworkdayjobs.com" in url or "workday" in url:
        return "Workday"
    if "ashbyhq.com" in url or "ashby" in url:
        return "Ashby"
    if job_url.strip():
        return "company website"
    return "unknown"


def infer_company_from_url(job_url: str) -> str:
    parsed = urlparse(job_url.strip())
    host = parsed.netloc.replace("www.", "")
    if not host:
        return ""
    return host.split(".")[0].replace("-", " ").title()


def split_raw_job_descriptions(raw_text: str) -> list:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*---+\s*\n", raw_text) if chunk.strip()]
    return chunks


def parse_job_urls(raw_urls: str) -> list:
    jobs = []
    for line in raw_urls.splitlines():
        url = line.strip()
        if not url:
            continue
        jobs.append(
            {
                "company": infer_company_from_url(url),
                "job_title": "",
                "location": "",
                "job_url": url,
                "jd_text": "",
            }
        )
    return jobs


def parse_csv_jobs(uploaded_file) -> list:
    df = pd.read_csv(uploaded_file)
    missing_columns = [column for column in REQUIRED_CSV_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {', '.join(missing_columns)}")

    jobs = []
    for row in df[REQUIRED_CSV_COLUMNS].fillna("").to_dict(orient="records"):
        jobs.append(
            {
                "company": row["company"],
                "job_title": row["job_title"],
                "location": row["location"],
                "job_url": row["job_url"],
                "jd_text": row["jd_text"],
            }
        )
    return jobs


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def extract_job_metadata(job: dict) -> dict:
    if not job.get("jd_text", "").strip():
        return {
            "company": job.get("company") or infer_company_from_url(job.get("job_url", "")),
            "title": job.get("job_title", ""),
            "location": job.get("location", ""),
            "required_skills": "",
            "years_experience": "",
            "visa_sponsorship_hints": "",
            "application_platform": infer_application_platform(job.get("job_url", "")),
        }

    system_prompt = """
You extract structured metadata from job descriptions for an AI job search tool.
Return strict JSON only. Do not include markdown.
"""
    user_prompt = f"""
Known company: {job.get("company", "")}
Known job title: {job.get("job_title", "")}
Known location: {job.get("location", "")}
Job URL: {job.get("job_url", "")}

Job description:
{job.get("jd_text", "")}

Return JSON with exactly these keys:
company, title, location, required_skills, years_experience, visa_sponsorship_hints.

Use concise string values. For required_skills, use a comma-separated string.
If something is unknown, use an empty string.
"""
    metadata = parse_json_object(call_llm(system_prompt, user_prompt))
    metadata["company"] = metadata.get("company") or job.get("company") or infer_company_from_url(job.get("job_url", ""))
    metadata["title"] = metadata.get("title") or job.get("job_title", "")
    metadata["location"] = metadata.get("location") or job.get("location", "")
    metadata["application_platform"] = infer_application_platform(job.get("job_url", ""))
    return metadata


def extract_missing_skills_from_analysis(analysis: str) -> list:
    if "## Missing Skills" not in analysis:
        return []
    section = analysis.split("## Missing Skills", 1)[1]
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


def decide_apply(
    fit_score: int,
    missing_skills: list,
    metadata: dict,
    career_profile: dict,
    analysis: str,
) -> dict:
    target_roles = career_profile.get("target_roles", "").lower()
    preferred_locations = career_profile.get("preferred_locations", "").lower()
    visa_status = career_profile.get("visa_status", "").lower()

    title = metadata.get("title", "").lower()
    location = metadata.get("location", "").lower()
    visa_hints = metadata.get("visa_sponsorship_hints", "").lower()

    role_match = not target_roles or any(
        role.strip() and role.strip() in title for role in target_roles.split(",")
    )
    location_match = not preferred_locations or any(
        loc.strip() and loc.strip() in location for loc in preferred_locations.split(",")
    )
    visa_risk = "sponsor" in visa_status and (
        "no sponsorship" in visa_hints or "unable to sponsor" in visa_hints
    )
    growth_signal = "Growth Potential:" in analysis and not re.search(
        r"Growth Potential:\s*[0-4]\s*/\s*10", analysis
    )
    portfolio_signal = "portfolio" in analysis.lower() or "project" in analysis.lower()

    if fit_score >= 78 and role_match and not visa_risk:
        decision = "Apply"
    elif fit_score >= 60 and not visa_risk:
        decision = "Maybe"
    else:
        decision = "Skip"

    if decision == "Apply" and not location_match:
        decision = "Maybe"
    if decision == "Apply" and len(missing_skills) >= 5:
        decision = "Maybe"

    reason_parts = [
        f"fit score {fit_score}",
        "target role aligned" if role_match else "target role alignment is weak",
        "location aligned" if location_match else "location preference is uncertain",
        "visa risk present" if visa_risk else "no clear visa blocker",
    ]
    if growth_signal:
        reason_parts.append("growth value is present")
    if portfolio_signal:
        reason_parts.append("portfolio alignment is present")
    if missing_skills:
        reason_parts.append(f"missing skills: {', '.join(missing_skills[:3])}")

    return {"apply_decision": decision, "reason": "; ".join(reason_parts)}


def generate_application_prep(user_profile: str, job_description: str, career_profile_text: str) -> dict:
    resume_text = tailor_resume(user_profile, job_description)
    system_prompt = """
You prepare application materials for an AI Engineer job application.
Return strict JSON only. Do not include markdown.
"""
    user_prompt = f"""
Career profile:
{career_profile_text}

Candidate profile/resume:
{user_profile}

Job description:
{job_description}

Return JSON with exactly these keys:
cover_letter, recruiter_message, application_checklist.

The cover_letter should be concise and specific.
The recruiter_message should be direct and not generic.
The application_checklist should be a short newline-separated checklist.
"""
    prep = parse_json_object(call_llm(system_prompt, user_prompt))
    return {
        "resume_bullets": resume_text,
        "cover_letter": prep.get("cover_letter", ""),
        "recruiter_message": prep.get("recruiter_message", ""),
        "application_checklist": prep.get("application_checklist", ""),
    }


def insert_job_queue_item(item: dict) -> int:
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO job_queue (
                company,
                title,
                location,
                required_skills,
                years_experience,
                visa_sponsorship_hints,
                application_platform,
                fit_score,
                apply_decision,
                reason,
                job_url,
                status,
                jd_text,
                analysis_text,
                resume_bullets,
                cover_letter,
                recruiter_message,
                application_checklist,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["company"],
                item["title"],
                item["location"],
                item["required_skills"],
                item["years_experience"],
                item["visa_sponsorship_hints"],
                item["application_platform"],
                item["fit_score"],
                item["apply_decision"],
                item["reason"],
                item["job_url"],
                item["status"],
                item["jd_text"],
                item["analysis_text"],
                item["resume_bullets"],
                item["cover_letter"],
                item["recruiter_message"],
                item["application_checklist"],
                now_iso(),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def process_discovered_job(
    job: dict,
    user_profile: str,
    career_profile: dict,
    career_profile_text: str,
) -> dict:
    metadata = extract_job_metadata(job)
    jd_text = job.get("jd_text", "")
    platform = metadata.get("application_platform") or infer_application_platform(job.get("job_url", ""))

    if not jd_text.strip():
        queue_item = {
            "company": metadata.get("company", ""),
            "title": metadata.get("title", ""),
            "location": metadata.get("location", ""),
            "required_skills": metadata.get("required_skills", ""),
            "years_experience": metadata.get("years_experience", ""),
            "visa_sponsorship_hints": metadata.get("visa_sponsorship_hints", ""),
            "application_platform": platform,
            "fit_score": 0,
            "apply_decision": "Maybe",
            "reason": "Job URL saved, but no job description text was provided. No scraping is performed in this MVP.",
            "job_url": job.get("job_url", ""),
            "status": "Discovered",
            "jd_text": "",
            "analysis_text": "",
            "resume_bullets": "",
            "cover_letter": "",
            "recruiter_message": "",
            "application_checklist": "",
        }
        insert_job_queue_item(queue_item)
        return queue_item

    analysis = analyze_job(user_profile, jd_text, career_profile_text)
    score_breakdown = parse_score_breakdown(analysis)
    fit_score = score_breakdown["fit_score"]
    missing_skills = extract_missing_skills_from_analysis(analysis)
    decision = decide_apply(fit_score, missing_skills, metadata, career_profile, analysis)

    prep = {
        "resume_bullets": "",
        "cover_letter": "",
        "recruiter_message": "",
        "application_checklist": "",
    }
    status = "Reviewed"
    if decision["apply_decision"] == "Apply":
        prep = generate_application_prep(user_profile, jd_text, career_profile_text)
        status = "Ready to Apply"
    elif decision["apply_decision"] == "Skip":
        status = "Skipped"

    queue_item = {
        "company": metadata.get("company", ""),
        "title": metadata.get("title", ""),
        "location": metadata.get("location", ""),
        "required_skills": metadata.get("required_skills", ""),
        "years_experience": metadata.get("years_experience", ""),
        "visa_sponsorship_hints": metadata.get("visa_sponsorship_hints", ""),
        "application_platform": platform,
        "fit_score": fit_score,
        "apply_decision": decision["apply_decision"],
        "reason": decision["reason"],
        "job_url": job.get("job_url", ""),
        "status": status,
        "jd_text": jd_text,
        "analysis_text": analysis,
        "resume_bullets": prep["resume_bullets"],
        "cover_letter": prep["cover_letter"],
        "recruiter_message": prep["recruiter_message"],
        "application_checklist": prep["application_checklist"],
    }
    insert_job_queue_item(queue_item)
    return queue_item


def fetch_job_queue() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                id,
                company,
                title,
                location,
                required_skills,
                years_experience,
                visa_sponsorship_hints,
                application_platform,
                fit_score,
                apply_decision,
                reason,
                job_url,
                status,
                resume_bullets,
                cover_letter,
                recruiter_message,
                application_checklist,
                created_at
            FROM job_queue
            ORDER BY
                CASE apply_decision
                    WHEN 'Apply' THEN 1
                    WHEN 'Maybe' THEN 2
                    WHEN 'Skip' THEN 3
                    ELSE 4
                END,
                fit_score DESC,
                created_at DESC
            """,
            conn,
        )


def update_job_queue_status(queue_id: int, status: str) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            "UPDATE job_queue SET status = ? WHERE id = ?",
            (status, queue_id),
        )
        conn.commit()
