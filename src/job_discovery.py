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
DEFAULT_PRE_FILTER_THRESHOLD = 40
DEFAULT_MAX_JOBS_PER_RUN = 100
DEFAULT_MAX_LLM_CALLS_PER_RUN = 10

POSITIVE_KEYWORDS = [
    "ai",
    "genai",
    "generative ai",
    "llm",
    "agent",
    "rag",
    "machine learning",
    "ml engineer",
    "ai engineer",
    "applied ai",
    "ai application",
    "automation",
    "solutions engineer",
    "sales engineer",
    "technical consultant",
    "python",
    "langchain",
    "langgraph",
    "fastapi",
    "aws",
    "vector database",
    "saas",
]

NEGATIVE_KEYWORDS = [
    "staff",
    "principal",
    "director",
    "vp",
    "8+ years",
    "10+ years",
    "commission only",
    "unpaid",
    "door-to-door",
    "insurance sales",
    "retail sales",
    "warehouse",
    "cashier",
]

HARD_NEGATIVE_KEYWORDS = [
    "commission only",
    "unpaid",
    "door-to-door",
    "insurance sales",
    "retail sales",
    "warehouse",
    "cashier",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(value).lower()).strip()


def infer_application_platform(job_url: str) -> str:
    url = (job_url or "").lower()
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
    if url.strip():
        return "company website"
    return "unknown"


def infer_company_from_url(job_url: str) -> str:
    parsed = urlparse((job_url or "").strip())
    host = parsed.netloc.replace("www.", "")
    if not host:
        return ""
    return host.split(".")[0].replace("-", " ").title()


def infer_title_from_jd(jd_text: str) -> str:
    for line in (jd_text or "").splitlines()[:8]:
        clean_line = normalize_text(line)
        if clean_line and len(clean_line) <= 90:
            return clean_line
    return ""


def short_description_from_jd(jd_text: str, max_chars: int = 700) -> str:
    text = normalize_text(jd_text)
    return text[:max_chars]


def parse_manual_jobs(raw_text: str) -> list:
    jobs = []
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*---+\s*\n", raw_text or "") if chunk.strip()]
    for chunk in chunks:
        fields = {
            "company": "",
            "job_title": "",
            "location": "",
            "job_url": "",
            "jd_text": "",
            "source": "manual",
        }
        description_lines = []
        active_field = None
        for line in chunk.splitlines():
            if re.match(r"^\s*company\s*:", line, re.IGNORECASE):
                fields["company"] = line.split(":", 1)[1].strip()
                active_field = None
            elif re.match(r"^\s*title\s*:", line, re.IGNORECASE):
                fields["job_title"] = line.split(":", 1)[1].strip()
                active_field = None
            elif re.match(r"^\s*location\s*:", line, re.IGNORECASE):
                fields["location"] = line.split(":", 1)[1].strip()
                active_field = None
            elif re.match(r"^\s*url\s*:", line, re.IGNORECASE):
                fields["job_url"] = line.split(":", 1)[1].strip()
                active_field = None
            elif re.match(r"^\s*description\s*:", line, re.IGNORECASE):
                first_line = line.split(":", 1)[1].strip()
                if first_line:
                    description_lines.append(first_line)
                active_field = "description"
            elif active_field == "description":
                description_lines.append(line)
        fields["jd_text"] = "\n".join(description_lines).strip() or chunk
        jobs.append(fields)
    return jobs


def parse_job_urls(raw_urls: str) -> list:
    jobs = []
    for line in (raw_urls or "").splitlines():
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
                "source": "url",
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
                "source": "csv",
            }
        )
    return jobs


def normalize_job(job: dict) -> dict:
    jd_text = str(job.get("jd_text", "") or "").strip()
    job_url = normalize_text(job.get("job_url", ""))
    company = normalize_text(job.get("company", "")) or infer_company_from_url(job_url)
    job_title = normalize_text(job.get("job_title", "")) or infer_title_from_jd(jd_text)
    location = normalize_text(job.get("location", ""))
    return {
        "company": company,
        "job_title": job_title,
        "location": location,
        "job_url": job_url,
        "jd_text": jd_text,
        "short_description": short_description_from_jd(jd_text),
        "source": normalize_text(job.get("source", "")) or "manual",
    }


def job_identity(job: dict) -> tuple:
    return (
        normalize_key(job.get("company", "")),
        normalize_key(job.get("job_title", "")),
        normalize_key(job.get("location", "")),
    )


def existing_queue_keys() -> tuple:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT company, COALESCE(job_title, title, ''), location, job_url
            FROM job_queue
            """
        ).fetchall()
    identity_keys = set()
    url_keys = set()
    for company, job_title, location, job_url in rows:
        identity_keys.add((normalize_key(company), normalize_key(job_title), normalize_key(location)))
        if job_url:
            url_keys.add(normalize_text(job_url).lower())
    return identity_keys, url_keys


def deduplicate_jobs(jobs: list) -> tuple:
    existing_identities, existing_urls = existing_queue_keys()
    seen_identities = set()
    seen_urls = set()
    unique_jobs = []
    duplicate_jobs = []

    for raw_job in jobs:
        job = normalize_job(raw_job)
        identity = job_identity(job)
        url_key = job["job_url"].lower()
        is_duplicate = identity in existing_identities or identity in seen_identities
        if url_key:
            is_duplicate = is_duplicate or url_key in existing_urls or url_key in seen_urls

        if is_duplicate:
            duplicate_jobs.append({**job, "duplicate_reason": "Already exists in queue or current batch"})
        else:
            unique_jobs.append(job)
            seen_identities.add(identity)
            if url_key:
                seen_urls.add(url_key)
    return unique_jobs, duplicate_jobs


def combined_text(job: dict) -> str:
    return " ".join(
        [
            job.get("company", ""),
            job.get("job_title", ""),
            job.get("location", ""),
            job.get("job_url", ""),
            job.get("short_description", ""),
            job.get("jd_text", "")[:1200],
        ]
    ).lower()


def rule_based_filter(job: dict, career_profile: dict) -> dict:
    text = combined_text(job)
    title = job.get("job_title", "").lower()
    location = job.get("location", "").lower()
    preferred_locations = career_profile.get("preferred_locations", "").lower()

    for keyword in HARD_NEGATIVE_KEYWORDS:
        if keyword in text:
            return {"passed": False, "reason": f"Filtered out by keyword: {keyword}"}

    if any(keyword in title for keyword in ["staff", "principal", "director", "vp"]):
        return {"passed": False, "reason": "Filtered out by seniority"}

    if re.search(r"(8|9|10)\+?\s*(?:years|yrs)", text):
        return {"passed": False, "reason": "Filtered out by experience requirement"}

    if preferred_locations and location:
        location_terms = [loc.strip() for loc in preferred_locations.split(",") if loc.strip()]
        remote_ok = "remote" in location or "remote" in preferred_locations
        if location_terms and not remote_ok and not any(loc in location for loc in location_terms):
            return {"passed": False, "reason": "Filtered out by location preference"}

    return {"passed": True, "reason": ""}


def keyword_pre_score(job: dict, career_profile: dict) -> int:
    text = combined_text(job)
    title = job.get("job_title", "").lower()
    target_roles = career_profile.get("target_roles", "").lower()
    score = 0

    target_role_terms = [role.strip() for role in target_roles.split(",") if role.strip()]
    if target_role_terms and any(role in title for role in target_role_terms):
        score += 25

    if any(keyword in text for keyword in ["llm", "genai", "generative ai", "agent", "rag"]):
        score += 20
    if any(keyword in text for keyword in ["python", "fastapi", "langchain", "langgraph"]):
        score += 15
    if any(keyword in text for keyword in ["aws", "cloud", "vector db", "vector database"]):
        score += 10
    if any(keyword in text for keyword in ["solutions engineer", "sales engineer"]):
        if "ai" in text or "saas" in text:
            score += 10

    if any(keyword in title for keyword in ["staff", "principal", "director", "vp"]):
        score -= 30
    if re.search(r"(8|10)\+?\s*(?:years|yrs)", text):
        score -= 40
    if any(keyword in text for keyword in ["commission only", "unpaid", "door-to-door"]):
        score -= 50

    return max(0, min(100, score))


def top_relevant_keyword_lines(jd_text: str, max_lines: int = 8) -> str:
    lines = [normalize_text(line) for line in (jd_text or "").splitlines()]
    relevant_lines = []
    keywords = POSITIVE_KEYWORDS + ["requirements", "responsibilities", "qualifications"]
    for line in lines:
        lower_line = line.lower()
        if line and any(keyword in lower_line for keyword in keywords):
            relevant_lines.append(line)
        if len(relevant_lines) >= max_lines:
            break
    if not relevant_lines:
        relevant_lines = [line for line in lines if line][:max_lines]
    return "\n".join(relevant_lines)


def compressed_job_text(job: dict) -> str:
    return f"""
Company: {job.get("company", "")}
Job title: {job.get("job_title", "")}
Location: {job.get("location", "")}
Short description: {job.get("short_description", "")}
Relevant JD lines:
{top_relevant_keyword_lines(job.get("jd_text", ""))}
"""


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def extract_missing_skills_from_analysis(analysis: str) -> list:
    if "## Missing Skills" not in analysis:
        return []
    section = analysis.split("## Missing Skills", 1)[1]
    if "##" in section:
        section = section.split("##", 1)[0]
    return [
        line.strip().lstrip("-").strip()
        for line in section.splitlines()
        if line.strip().startswith("-") and line.strip().lstrip("-").strip()
    ]


def decide_apply(fit_score: int, missing_skills: list, job: dict, career_profile: dict, analysis: str) -> dict:
    preferred_locations = career_profile.get("preferred_locations", "").lower()
    visa_status = career_profile.get("visa_status", "").lower()
    location = job.get("location", "").lower()
    text = combined_text(job)
    visa_risk = "sponsor" in visa_status and (
        "no sponsorship" in text or "unable to sponsor" in text
    )
    location_uncertain = bool(preferred_locations and location and "remote" not in location)
    portfolio_alignment = any(term in analysis.lower() for term in ["portfolio", "project", "application"])

    if fit_score >= 78 and not visa_risk:
        decision = "Apply"
    elif fit_score >= 60 and not visa_risk:
        decision = "Maybe"
    else:
        decision = "Skip"

    if decision == "Apply" and location_uncertain:
        decision = "Maybe"
    if decision == "Apply" and len(missing_skills) >= 5:
        decision = "Maybe"

    reason = [
        f"fit score {fit_score}",
        "visa risk present" if visa_risk else "no clear visa blocker",
        "portfolio alignment present" if portfolio_alignment else "portfolio alignment unclear",
    ]
    if missing_skills:
        reason.append(f"missing skills: {', '.join(missing_skills[:3])}")
    return {"apply_decision": decision, "decision_reason": "; ".join(reason)}


def generate_application_prep(user_profile: str, job_description: str, career_profile_text: str) -> dict:
    resume_text = tailor_resume(user_profile, job_description)
    system_prompt = """
You prepare concise application materials for an AI Engineer job application.
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

The cover_letter should be an outline, not a full polished letter.
The recruiter_message should be direct and specific.
The application_checklist should be a short newline-separated checklist.
"""
    prep = parse_json_object(call_llm(system_prompt, user_prompt))
    return {
        "resume_bullets": resume_text,
        "cover_letter": prep.get("cover_letter", ""),
        "recruiter_message": prep.get("recruiter_message", ""),
        "application_checklist": prep.get("application_checklist", ""),
    }


def empty_queue_item(job: dict) -> dict:
    return {
        "company": job.get("company", ""),
        "job_title": job.get("job_title", ""),
        "location": job.get("location", ""),
        "job_url": job.get("job_url", ""),
        "jd_text": job.get("jd_text", ""),
        "short_description": job.get("short_description", ""),
        "source": job.get("source", ""),
        "pre_filter_score": 0,
        "fit_score": 0,
        "apply_decision": "Skip",
        "decision_reason": "",
        "status": "Skipped",
        "analysis_text": "",
        "resume_bullets": "",
        "cover_letter": "",
        "recruiter_message": "",
        "application_checklist": "",
        "sent_to_llm": 0,
    }


def insert_job_queue_item(item: dict) -> int:
    init_db()
    timestamp = now_iso()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO job_queue (
                company,
                job_title,
                title,
                location,
                job_url,
                jd_text,
                short_description,
                source,
                pre_filter_score,
                fit_score,
                apply_decision,
                decision_reason,
                reason,
                status,
                analysis_text,
                resume_bullets,
                cover_letter,
                recruiter_message,
                application_checklist,
                sent_to_llm,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["company"],
                item["job_title"],
                item["job_title"],
                item["location"],
                item["job_url"],
                item["jd_text"],
                item["short_description"],
                item["source"],
                item["pre_filter_score"],
                item["fit_score"],
                item["apply_decision"],
                item["decision_reason"],
                item["decision_reason"],
                item["status"],
                item.get("analysis_text", ""),
                item.get("resume_bullets", ""),
                item.get("cover_letter", ""),
                item.get("recruiter_message", ""),
                item.get("application_checklist", ""),
                item.get("sent_to_llm", 0),
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def evaluate_job_without_llm(job: dict, career_profile: dict, pre_filter_threshold: int) -> dict:
    item = empty_queue_item(job)
    rule_result = rule_based_filter(job, career_profile)
    if not rule_result["passed"]:
        item["decision_reason"] = rule_result["reason"]
        return item

    pre_filter_score = keyword_pre_score(job, career_profile)
    item["pre_filter_score"] = pre_filter_score
    if pre_filter_score < pre_filter_threshold:
        item["decision_reason"] = f"Pre-filter score {pre_filter_score} is below threshold {pre_filter_threshold}."
        return item

    if not job.get("jd_text", "").strip():
        item["apply_decision"] = "Maybe"
        item["status"] = "Discovered"
        item["decision_reason"] = "Passed pre-score, but no JD text is available. No scraping is performed."
        return item

    item["apply_decision"] = "Maybe"
    item["status"] = "Reviewed"
    item["decision_reason"] = "Passed pre-score and is eligible for LLM deep analysis."
    return item


def evaluate_job_with_llm(job: dict, item: dict, user_profile: str, career_profile: dict, career_profile_text: str) -> dict:
    analysis = analyze_job(user_profile, compressed_job_text(job), career_profile_text)
    score_breakdown = parse_score_breakdown(analysis)
    fit_score = score_breakdown["fit_score"]
    missing_skills = extract_missing_skills_from_analysis(analysis)
    decision = decide_apply(fit_score, missing_skills, job, career_profile, analysis)

    item["fit_score"] = fit_score
    item["apply_decision"] = decision["apply_decision"]
    item["decision_reason"] = decision["decision_reason"]
    item["status"] = "Ready to Apply" if decision["apply_decision"] == "Apply" else "Reviewed"
    if decision["apply_decision"] == "Skip":
        item["status"] = "Skipped"
    item["analysis_text"] = analysis
    item["sent_to_llm"] = 1
    return item


def process_discovered_jobs(
    jobs: list,
    user_profile: str,
    career_profile: dict,
    career_profile_text: str,
    pre_filter_threshold: int = DEFAULT_PRE_FILTER_THRESHOLD,
    max_llm_calls_per_run: int = DEFAULT_MAX_LLM_CALLS_PER_RUN,
    max_jobs_per_run: int = DEFAULT_MAX_JOBS_PER_RUN,
) -> dict:
    selected_jobs = jobs[:max_jobs_per_run]
    normalized_jobs = [normalize_job(job) for job in selected_jobs]
    unique_jobs, duplicate_jobs = deduplicate_jobs(normalized_jobs)

    metrics = {
        "jobs_discovered": len(jobs),
        "jobs_filtered_out": len(duplicate_jobs),
        "jobs_pre_scored": 0,
        "jobs_sent_to_llm": 0,
        "estimated_token_savings": 0,
        "imported_jobs": normalized_jobs,
        "filtered_out_jobs": duplicate_jobs,
        "jobs_sent_to_llm_rows": [],
        "processed_jobs": [],
    }

    if len(jobs) > max_jobs_per_run:
        over_limit = len(jobs) - max_jobs_per_run
        metrics["jobs_filtered_out"] += over_limit
        metrics["estimated_token_savings"] += over_limit * 750

    for duplicate_job in duplicate_jobs:
        duplicate_item = empty_queue_item(duplicate_job)
        duplicate_item["decision_reason"] = duplicate_job["duplicate_reason"]
        metrics["processed_jobs"].append(duplicate_item)

    for job in unique_jobs:
        item = evaluate_job_without_llm(job, career_profile, pre_filter_threshold)
        if item["pre_filter_score"] > 0 or item["status"] in ["Reviewed", "Discovered"]:
            metrics["jobs_pre_scored"] += 1

        eligible_for_llm = (
            item["pre_filter_score"] >= pre_filter_threshold
            and bool(job.get("jd_text", "").strip())
            and metrics["jobs_sent_to_llm"] < max_llm_calls_per_run
        )

        if eligible_for_llm:
            item = evaluate_job_with_llm(job, item, user_profile, career_profile, career_profile_text)
            metrics["jobs_sent_to_llm"] += 1
            metrics["jobs_sent_to_llm_rows"].append(item)
        else:
            metrics["jobs_filtered_out"] += 1
            if not item["sent_to_llm"]:
                metrics["estimated_token_savings"] += max(250, len(job.get("jd_text", "")) // 4)

        insert_job_queue_item(item)
        metrics["processed_jobs"].append(item)

    return metrics


def fetch_job_queue() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                id,
                company,
                COALESCE(job_title, title, '') AS job_title,
                location,
                job_url,
                jd_text,
                short_description,
                source,
                pre_filter_score,
                fit_score,
                apply_decision,
                COALESCE(decision_reason, reason, '') AS decision_reason,
                status,
                resume_bullets,
                cover_letter,
                recruiter_message,
                application_checklist,
                created_at,
                updated_at
            FROM job_queue
            ORDER BY
                CASE apply_decision
                    WHEN 'Apply' THEN 1
                    WHEN 'Maybe' THEN 2
                    WHEN 'Skip' THEN 3
                    ELSE 4
                END,
                fit_score DESC,
                pre_filter_score DESC,
                created_at DESC
            """,
            conn,
        )


def update_job_queue_status(queue_id: int, status: str) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE job_queue
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, now_iso(), queue_id),
        )
        conn.commit()


def update_application_prep(queue_id: int, prep: dict) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE job_queue
            SET
                resume_bullets = ?,
                cover_letter = ?,
                recruiter_message = ?,
                application_checklist = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                prep.get("resume_bullets", ""),
                prep.get("cover_letter", ""),
                prep.get("recruiter_message", ""),
                prep.get("application_checklist", ""),
                now_iso(),
                queue_id,
            ),
        )
        conn.commit()
