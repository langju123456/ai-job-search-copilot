import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import pandas as pd

from src.database import get_connection, init_db
from src.job_analyzer import analyze_job
from src.llm import call_llm
from src.profile import compute_career_profile_hash
from src.resume_assets import resume_assets_to_text, retrieve_relevant_resume_assets
from src.resume_tailor import tailor_resume
from src.scoring import (
    compact_career_profile_text,
    parse_score_breakdown,
    select_relevant_projects,
    split_terms,
    top_skills,
)


JOB_QUEUE_STATUS_OPTIONS = [
    "Discovered",
    "Reviewed",
    "Ready to Apply",
    "Applied",
    "Skipped",
]

APPLY_DECISIONS = ["Apply", "Maybe", "Skip"]
REQUIRED_CSV_COLUMNS = ["company", "job_title", "location", "job_url", "jd_text"]
OPTIONAL_CSV_COLUMNS = ["post_time", "job_level", "work_mode"]
DEFAULT_PRE_FILTER_THRESHOLD = 40
DEFAULT_MAX_JOBS_PER_RUN = 100
DEFAULT_MAX_LLM_CALLS_PER_RUN = 10
SAMPLE_DISCOVERED_JOBS_PATH = Path("data/sample_discovered_jobs.csv")
LLM_TOKEN_ESTIMATE_PER_FULL_JOB = 750

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

SENIOR_LEVELS = ["Staff", "Principal", "Director"]
QUEUE_CATEGORIES = ["Apply", "Maybe", "Filtered", "Rejected"]


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


def source_name_from_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.replace("www.", "")
    return host or "public_html"


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
    available_columns = REQUIRED_CSV_COLUMNS + [
        column for column in OPTIONAL_CSV_COLUMNS if column in df.columns
    ]
    for row in df[available_columns].fillna("").to_dict(orient="records"):
        jobs.append(
            {
                "company": row["company"],
                "job_title": row["job_title"],
                "location": row["location"],
                "job_url": row["job_url"],
                "jd_text": row["jd_text"],
                "source": "csv",
                "post_time": row.get("post_time", ""),
                "job_level": row.get("job_level", ""),
                "work_mode": row.get("work_mode", ""),
            }
        )
    return jobs


def fetch_public_html(url: str) -> str:
    clean_url = normalize_text(url)
    if not clean_url:
        raise ValueError("Source URL is empty.")
    if "linkedin.com" in clean_url.lower():
        raise ValueError("LinkedIn scraping is not supported in this MVP. Use CSV or manual paste instead.")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is not installed. Run pip install -r requirements.txt.") from exc

    response = requests.get(
        clean_url,
        timeout=15,
        headers={
            "User-Agent": "AIJobSearchCopilot/0.1 (+local Streamlit MVP)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    response.raise_for_status()
    return response.text


def _get_soup(html: str):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is not installed. Run pip install -r requirements.txt.") from exc
    return BeautifulSoup(html or "", "html.parser")


def extract_job_links_from_html(html: str, base_url: str) -> list:
    soup = _get_soup(html)
    jobs = []
    job_terms = [
        "job",
        "career",
        "opening",
        "position",
        "role",
        "greenhouse",
        "lever",
        "workday",
        "ashby",
    ]

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        text = normalize_text(anchor.get_text(" "))
        combined = f"{href} {text}".lower()
        if not text and not href:
            continue
        if not any(term in combined for term in job_terms):
            continue

        absolute_url = urljoin(base_url, href)
        jobs.append(
            {
                "company": infer_company_from_url(absolute_url) or source_name_from_url(base_url),
                "job_title": text[:120],
                "location": "",
                "job_url": absolute_url,
                "jd_text": text,
                "short_description": text,
                "source": "public_html",
            }
        )
    return jobs


def _looks_like_job_card(element) -> bool:
    attrs = " ".join(
        [
            " ".join(element.get("class", [])),
            element.get("id", ""),
            element.get("data-testid", ""),
        ]
    ).lower()
    text = normalize_text(element.get_text(" ")).lower()
    return any(term in attrs for term in ["job", "opening", "position", "role", "listing"]) or (
        any(term in text for term in POSITIVE_KEYWORDS)
        and any(term in text for term in ["apply", "remote", "engineer", "consultant", "location"])
    )


def _extract_location_from_text(text: str) -> str:
    location_patterns = [
        r"\b(Remote(?:\s*-\s*[A-Za-z ,]+)?)\b",
        r"\b(San Francisco|New York|Seattle|Austin|Boston|Los Angeles|Chicago|Atlanta|Dallas|Denver|Toronto|London)\b",
        r"\b([A-Z][a-z]+,\s*(?:CA|NY|WA|TX|MA|IL|GA|CO|FL|NJ))\b",
    ]
    for pattern in location_patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_text(match.group(1))
    return ""


def extract_job_cards_from_html(html: str, base_url: str) -> list:
    soup = _get_soup(html)
    jobs = []
    candidates = soup.find_all(["article", "li", "div", "section"])

    for element in candidates:
        if not _looks_like_job_card(element):
            continue

        card_text = normalize_text(element.get_text(" "))
        if len(card_text) < 20:
            continue

        anchor = element.find("a", href=True)
        job_url = urljoin(base_url, anchor.get("href", "")) if anchor else ""
        anchor_text = normalize_text(anchor.get_text(" ")) if anchor else ""
        heading = element.find(["h1", "h2", "h3", "h4"])
        heading_text = normalize_text(heading.get_text(" ")) if heading else ""
        title = heading_text or anchor_text or card_text[:90]

        jobs.append(
            {
                "company": source_name_from_url(base_url),
                "job_title": title[:120],
                "location": _extract_location_from_text(card_text),
                "job_url": job_url,
                "jd_text": card_text,
                "short_description": card_text[:700],
                "source": "public_html",
            }
        )
    return jobs


def normalize_discovered_job(raw_job: dict) -> dict:
    return normalize_job(raw_job)


def discover_from_public_url(url: str, settings=None) -> list:
    settings = settings or {}
    html = fetch_public_html(url)
    jobs = extract_job_cards_from_html(html, url)
    jobs.extend(extract_job_links_from_html(html, url))

    keywords = normalize_text(settings.get("keywords", "")).lower()
    excluded_keywords = normalize_text(settings.get("excluded_keywords", "")).lower()
    keyword_terms = [term.strip() for term in re.split(r"[,;]", keywords) if term.strip()]
    excluded_terms = [term.strip() for term in re.split(r"[,;]", excluded_keywords) if term.strip()]

    normalized_jobs = [normalize_discovered_job(job) for job in jobs]
    if keyword_terms:
        normalized_jobs = [
            job for job in normalized_jobs if any(term in combined_text(job) for term in keyword_terms)
        ]
    if excluded_terms:
        normalized_jobs = [
            job for job in normalized_jobs if not any(term in combined_text(job) for term in excluded_terms)
        ]
    return normalized_jobs


def load_sample_discovered_jobs() -> list:
    df = pd.read_csv(SAMPLE_DISCOVERED_JOBS_PATH)
    missing_columns = [column for column in REQUIRED_CSV_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Sample dataset is missing columns: {', '.join(missing_columns)}")
    return [
        {
            "company": row["company"],
            "job_title": row["job_title"],
            "location": row["location"],
            "job_url": row["job_url"],
            "jd_text": row["jd_text"],
            "source": "sample",
            "post_time": row.get("post_time", ""),
            "job_level": row.get("job_level", ""),
            "work_mode": row.get("work_mode", ""),
        }
        for row in df.fillna("").to_dict(orient="records")
    ]


def upsert_discovered_source(source_name: str, source_url: str, source_type: str, enabled: int = 1) -> None:
    init_db()
    timestamp = now_iso()
    with get_connection() as conn:
        existing = conn.execute(
            """
            SELECT id
            FROM discovered_sources
            WHERE source_url = ? AND source_type = ?
            """,
            (source_url, source_type),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE discovered_sources
                SET source_name = ?, enabled = ?, updated_at = ?
                WHERE id = ?
                """,
                (source_name, enabled, timestamp, existing[0]),
            )
        else:
            conn.execute(
                """
                INSERT INTO discovered_sources (
                    source_name,
                    source_url,
                    source_type,
                    enabled,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_name, source_url, source_type, enabled, timestamp, timestamp),
            )
        conn.commit()


def record_discovery_run(
    source_name: str,
    source_type: str,
    search_keywords: str,
    target_locations: str,
    metrics: dict,
) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO job_discovery_runs (
                source_name,
                source_type,
                search_keywords,
                target_locations,
                total_discovered,
                total_new_jobs,
                total_duplicates,
                total_filtered_out,
                total_sent_to_llm,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_name,
                source_type,
                search_keywords,
                target_locations,
                metrics.get("jobs_discovered", 0),
                metrics.get("total_new_jobs", 0),
                metrics.get("duplicates_removed", 0),
                metrics.get("jobs_filtered_out", 0),
                metrics.get("jobs_sent_to_llm", 0),
                now_iso(),
            ),
        )
        conn.commit()


def infer_job_level(title: str, jd_text: str) -> str:
    text = f"{title} {jd_text}".lower()
    if any(term in text for term in ["internship", "intern "]):
        return "Internship"
    if any(term in text for term in ["entry level", "new grad", "graduate"]):
        return "Entry"
    if "junior" in text:
        return "Junior"
    if "staff" in text:
        return "Staff"
    if "principal" in text:
        return "Principal"
    if any(term in text for term in ["director", "vp", "vice president"]):
        return "Director"
    if "senior" in text or re.search(r"\b(sr|sr\.)\b", text):
        return "Senior"
    if re.search(r"(4|5|6|7)\+?\s*(?:years|yrs)", text):
        return "Mid"
    if re.search(r"(0|1|2|3)\+?\s*(?:years|yrs)", text):
        return "Junior"
    return "Unknown"


def infer_work_mode(location: str, jd_text: str) -> str:
    text = f"{location} {jd_text}".lower()
    if "remote" in text:
        return "Remote"
    if "hybrid" in text:
        return "Hybrid"
    if any(term in text for term in ["onsite", "on-site", "in office", "in-office"]):
        return "Onsite"
    return "Unknown"


def infer_post_time(raw_text: str) -> str:
    text = normalize_text(raw_text).lower()
    patterns = [
        r"posted\s+((?:today|yesterday|\d+\s+(?:hour|hours|day|days|week|weeks|month|months)\s+ago))",
        r"\b((?:today|yesterday|\d+\s+(?:hour|hours|day|days|week|weeks|month|months)\s+ago))\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).title()
    return "Unknown"


def should_filter_job(job: dict, settings: dict) -> dict:
    text = combined_text(job)
    allow_senior_roles = bool(settings.get("allow_senior_roles", False))
    career_profile = settings.get("career_profile", {}) or {}
    job_level = job.get("job_level", "Unknown")

    excluded_roles = [role.lower() for role in split_terms(career_profile.get("excluded_roles", ""))]
    for role in excluded_roles:
        if role and role in text:
            return {
                "filtered": True,
                "category": "Filtered",
                "reason": f"Filtered by excluded role: {role}",
            }

    preferred_locations = [location.lower() for location in split_terms(career_profile.get("preferred_locations", ""))]
    location_text = job.get("location", "").lower()
    if preferred_locations and location_text and "remote" not in location_text:
        if not any(location in location_text for location in preferred_locations):
            return {
                "filtered": True,
                "category": "Filtered",
                "reason": "Filtered by preferred location mismatch",
            }

    for keyword in HARD_NEGATIVE_KEYWORDS:
        if keyword in text:
            return {
                "filtered": True,
                "category": "Rejected",
                "reason": f"Rejected by low-quality keyword: {keyword}",
            }

    if not allow_senior_roles and job_level in SENIOR_LEVELS:
        return {
            "filtered": True,
            "category": "Filtered",
            "reason": f"Filtered by seniority: {job_level}",
        }

    if not allow_senior_roles and any(keyword in text for keyword in ["vp", "vice president"]):
        return {
            "filtered": True,
            "category": "Filtered",
            "reason": "Filtered by seniority: VP",
        }

    return {"filtered": False, "category": "", "reason": ""}


def assign_queue_category(item: dict) -> str:
    decision = item.get("apply_decision", "")
    if decision in ["Apply", "Maybe"]:
        return decision
    if item.get("rejection_reason"):
        return "Rejected"
    return "Filtered"


def normalize_job(job: dict) -> dict:
    jd_text = str(job.get("jd_text", "") or "").strip()
    job_url = normalize_text(job.get("job_url", ""))
    company = normalize_text(job.get("company", "")) or infer_company_from_url(job_url)
    job_title = normalize_text(job.get("job_title", "")) or infer_title_from_jd(jd_text)
    location = normalize_text(job.get("location", ""))
    raw_text = " ".join([job_title, location, jd_text, normalize_text(job.get("post_time", ""))])
    job_level = normalize_text(job.get("job_level", "")) or infer_job_level(job_title, jd_text)
    work_mode = normalize_text(job.get("work_mode", "")) or infer_work_mode(location, jd_text)
    post_time = normalize_text(job.get("post_time", "")) or infer_post_time(raw_text)
    return {
        "company": company,
        "job_title": job_title,
        "location": location,
        "job_url": job_url,
        "jd_text": jd_text,
        "short_description": short_description_from_jd(jd_text),
        "source": normalize_text(job.get("source", "")) or "manual",
        "post_time": post_time,
        "job_level": job_level,
        "work_mode": work_mode,
    }


def job_identity(job: dict) -> tuple:
    return (
        normalize_key(job.get("company", "")),
        normalize_key(job.get("job_title", "")),
        normalize_key(job.get("location", "")),
    )


def job_cache_key(job: dict) -> tuple:
    return (
        normalize_key(job.get("company", "")),
        normalize_key(job.get("job_title", "")),
        normalize_key(job.get("location", "")),
        normalize_text(job.get("job_url", "")).lower(),
    )


def existing_queue_index() -> dict:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                company,
                COALESCE(job_title, title, '') AS job_title,
                location,
                job_url,
                fit_score,
                apply_decision,
                COALESCE(decision_reason, reason, '') AS decision_reason,
                analysis_text,
                pre_filter_score,
                COALESCE(queue_category, apply_decision, '') AS queue_category,
                COALESCE(analysis_profile_hash, '') AS analysis_profile_hash,
                COALESCE(post_time, 'Unknown') AS post_time,
                COALESCE(job_level, 'Unknown') AS job_level,
                COALESCE(work_mode, 'Unknown') AS work_mode,
                updated_at,
                id
            FROM job_queue
            """
        ).fetchall()
    cache = {}

    def cache_priority(item: dict) -> tuple:
        return (
            1 if int(item.get("fit_score", 0) or 0) > 0 else 0,
            1 if item.get("apply_decision", "") in ["Apply", "Maybe"] else 0,
            sum(
                1
                for field in ["post_time", "job_level", "work_mode"]
                if str(item.get(field, "") or "") not in ["", "Unknown"]
            ),
            str(item.get("updated_at", "") or ""),
            int(item.get("id", 0) or 0),
        )

    for row in rows:
        company, job_title, location, job_url = row[:4]
        item = {
            "company": company or "",
            "job_title": job_title or "",
            "location": location or "",
            "job_url": job_url or "",
            "fit_score": row[4] or 0,
            "apply_decision": row[5] or "",
            "decision_reason": row[6] or "",
            "analysis_text": row[7] or "",
            "pre_filter_score": row[8] or 0,
            "queue_category": row[9] or "",
            "analysis_profile_hash": row[10] or "",
            "post_time": row[11] or "Unknown",
            "job_level": row[12] or "Unknown",
            "work_mode": row[13] or "Unknown",
            "updated_at": row[14] or "",
            "id": row[15],
        }
        exact_key = job_cache_key(item)
        existing_exact = cache.get(exact_key)
        if not existing_exact or cache_priority(item) >= cache_priority(existing_exact):
            cache[exact_key] = item
        if item["job_url"]:
            url_key = ("", "", "", item["job_url"].lower())
            existing_url = cache.get(url_key)
            if not existing_url or cache_priority(item) >= cache_priority(existing_url):
                cache[url_key] = item
    return cache


def deduplicate_jobs(jobs: list, current_profile_hash: str, force_refresh: bool = False) -> tuple:
    existing_jobs = existing_queue_index()
    seen_identities = set()
    seen_urls = set()
    unique_jobs = []
    known_jobs = []
    stale_jobs = []
    duplicate_jobs = []

    for raw_job in jobs:
        job = normalize_job(raw_job)
        identity = job_identity(job)
        url_key = job["job_url"].lower()
        exact_key = job_cache_key(job)
        url_cache_key = ("", "", "", url_key)
        cached_job = existing_jobs.get(exact_key) or existing_jobs.get(url_cache_key)
        cache_matches_profile = bool(cached_job) and (
            not cached_job.get("analysis_profile_hash")
            or cached_job.get("analysis_profile_hash") == current_profile_hash
        )
        is_known = bool(cached_job) and cache_matches_profile and not force_refresh
        is_duplicate = identity in seen_identities
        if url_key:
            is_duplicate = is_duplicate or url_key in seen_urls

        if is_known:
            known_jobs.append(
                {
                    **job,
                    "existing_reason": "Already known from historical queue",
                    "cached_result": cached_job,
                }
            )
        elif bool(cached_job) and not force_refresh:
            stale_jobs.append(
                {
                    **job,
                    "stale_reason": "Profile changed; re-analysis recommended",
                    "cached_result": cached_job,
                }
            )
        elif is_duplicate:
            duplicate_jobs.append({**job, "duplicate_reason": "Duplicate within current run"})
        else:
            unique_jobs.append(job)
            seen_identities.add(identity)
            if url_key:
                seen_urls.add(url_key)
    return unique_jobs, known_jobs, stale_jobs, duplicate_jobs


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


def rule_based_filter(job: dict, career_profile: dict, allow_senior_roles: bool = False) -> dict:
    text = combined_text(job)
    title = job.get("job_title", "").lower()
    location = job.get("location", "").lower()
    preferred_locations = career_profile.get("preferred_locations", "").lower()
    excluded_roles = [role.lower() for role in split_terms(career_profile.get("excluded_roles", ""))]

    for keyword in HARD_NEGATIVE_KEYWORDS:
        if keyword in text:
            return {"passed": False, "reason": f"Filtered out by keyword: {keyword}"}

    if any(role in text for role in excluded_roles):
        return {"passed": False, "reason": "Filtered out by excluded role"}

    if not allow_senior_roles and any(keyword in title for keyword in ["staff", "principal", "director", "vp"]):
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
    target_roles = career_profile.get("target_roles", "")
    acceptable_roles = career_profile.get("acceptable_roles", "")
    preferred_locations = career_profile.get("preferred_locations", "").lower()
    score = 0

    target_role_terms = [role.lower() for role in split_terms(target_roles)]
    if target_role_terms and any(role in title for role in target_role_terms):
        score += 25
    acceptable_role_terms = [role.lower() for role in split_terms(acceptable_roles)]
    if acceptable_role_terms and any(role in title for role in acceptable_role_terms):
        score += 12

    if any(keyword in text for keyword in ["llm", "genai", "generative ai", "agent", "rag"]):
        score += 20
    if any(keyword in text for keyword in ["python", "fastapi", "langchain", "langgraph"]):
        score += 15
    if any(keyword in text for keyword in ["aws", "cloud", "vector db", "vector database"]):
        score += 10
    if any(keyword in text for keyword in ["solutions engineer", "sales engineer"]):
        if "ai" in text or "saas" in text:
            score += 10
    if preferred_locations and (
        "remote" in preferred_locations
        and ("remote" in text or "hybrid" in text)
    ):
        score += 5

    profile_skills = [skill.lower() for skill in top_skills(career_profile, limit=12)]
    score += min(
        15,
        3 * sum(1 for skill in profile_skills if skill and skill in text),
    )

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
Job level: {job.get("job_level", "Unknown")}
Work mode: {job.get("work_mode", "Unknown")}
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

    role_hits = []
    for label, raw_value in [
        ("target roles", career_profile.get("target_roles", "")),
        ("acceptable roles", career_profile.get("acceptable_roles", "")),
    ]:
        hits = [role for role in split_terms(raw_value) if role.lower() in text]
        if hits:
            role_hits.append(f"{label}: {', '.join(hits[:3])}")

    relevant_projects = select_relevant_projects(career_profile, text, limit=2)
    project_names = [project.get("project_name", "") for project in relevant_projects if project.get("project_name")]
    active_constraints = [
        row
        for row in (career_profile.get("constraints", []) or [])
        if row.get("constraint_type") or row.get("constraint_value")
    ]

    reason = [
        f"fit score {fit_score}",
        "visa risk present" if visa_risk else "no clear visa blocker",
        "portfolio alignment present" if portfolio_alignment else "portfolio alignment unclear",
    ]
    if role_hits:
        reason.append("; ".join(role_hits))
    if project_names:
        reason.append(f"supporting projects: {', '.join(project_names)}")
    if missing_skills:
        reason.append(f"missing skills: {', '.join(missing_skills[:3])}")
    if active_constraints:
        reason.append(
            "constraints considered: "
            + ", ".join(
                row.get("constraint_type", "") or row.get("constraint_value", "")
                for row in active_constraints[:2]
            )
        )
    return {"apply_decision": decision, "decision_reason": "; ".join(reason)}


def generate_application_prep(
    user_profile: str,
    job_description: str,
    career_profile: dict,
    career_profile_text: str,
    selected_assets: Optional[list[dict]] = None,
    mode: str = "asset_first",
) -> dict:
    selected_assets = selected_assets or []
    selected_assets_text = resume_assets_to_text(selected_assets)
    candidate_context = (
        career_profile_text if mode == "asset_first" else user_profile
    )
    resume_text = tailor_resume(
        candidate_context,
        job_description,
        career_profile_text,
        selected_assets_text,
        mode,
    )
    system_prompt = """
You prepare concise application materials for an AI Engineer job application.
Return strict JSON only. Do not include markdown.
"""
    user_prompt = f"""
Career profile:
{career_profile_text}

Candidate profile/resume:
{candidate_context}

Selected resume assets:
{selected_assets_text}

Job description:
{job_description}

Return JSON with exactly these keys:
cover_letter, recruiter_message, application_checklist.

The cover_letter should be an outline, not a full polished letter.
The recruiter_message should be direct and specific.
The application_checklist should be a short newline-separated checklist.
"""
    if mode == "asset_first" and selected_assets_text.strip():
        user_prompt += """

Rules:
- Use the selected resume assets as the evidence base.
- Rewrite or polish those assets for this job instead of inventing new experience.
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
        "post_time": job.get("post_time", "Unknown"),
        "job_level": job.get("job_level", "Unknown"),
        "work_mode": job.get("work_mode", "Unknown"),
        "pre_filter_score": 0,
        "fit_score": 0,
        "apply_decision": "Skip",
        "decision_reason": "",
        "rejection_reason": "",
        "queue_category": "Filtered",
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
                analysis_profile_hash,
                sent_to_llm,
                post_time,
                job_level,
                work_mode,
                rejection_reason,
                queue_category,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                item.get("analysis_profile_hash", ""),
                item.get("sent_to_llm", 0),
                item.get("post_time", "Unknown"),
                item.get("job_level", "Unknown"),
                item.get("work_mode", "Unknown"),
                item.get("rejection_reason", ""),
                item.get("queue_category", assign_queue_category(item)),
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def evaluate_job_without_llm(
    job: dict,
    career_profile: dict,
    pre_filter_threshold: int,
    allow_senior_roles: bool = False,
) -> dict:
    item = empty_queue_item(job)
    rule_result = rule_based_filter(job, career_profile, allow_senior_roles)
    if not rule_result["passed"]:
        item["decision_reason"] = rule_result["reason"]
        item["rejection_reason"] = rule_result["reason"]
        item["queue_category"] = "Filtered"
        return item

    pre_filter_score = keyword_pre_score(job, career_profile)
    item["pre_filter_score"] = pre_filter_score
    if pre_filter_score < pre_filter_threshold:
        item["decision_reason"] = f"Pre-filter score {pre_filter_score} is below threshold {pre_filter_threshold}."
        item["rejection_reason"] = item["decision_reason"]
        item["queue_category"] = "Filtered"
        return item

    if not job.get("jd_text", "").strip():
        item["apply_decision"] = "Maybe"
        item["status"] = "Discovered"
        item["queue_category"] = "Maybe"
        item["decision_reason"] = "Passed pre-score, but no JD text is available. No scraping is performed."
        return item

    item["apply_decision"] = "Maybe"
    item["status"] = "Reviewed"
    item["queue_category"] = "Maybe"
    item["decision_reason"] = "Passed pre-score and is eligible for LLM deep analysis."
    return item


def evaluate_job_with_llm(job: dict, item: dict, user_profile: str, career_profile: dict, career_profile_text: str) -> dict:
    compact_profile = compact_career_profile_text(career_profile, combined_text(job))
    selected_assets = retrieve_relevant_resume_assets(job.get("jd_text", ""), career_profile, limit=5)
    current_profile_hash = compute_career_profile_hash(career_profile)
    analysis = analyze_job(
        compact_profile or career_profile_text or user_profile,
        compressed_job_text(job),
        compact_profile or career_profile_text,
        resume_assets_to_text(selected_assets),
    )
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
    item["queue_category"] = assign_queue_category(item)
    item["analysis_text"] = analysis
    item["analysis_profile_hash"] = current_profile_hash
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
    allow_senior_roles: bool = False,
    force_refresh: bool = False,
) -> dict:
    selected_jobs = jobs[:max_jobs_per_run]
    normalized_jobs = [normalize_job(job) for job in selected_jobs]
    current_profile_hash = compute_career_profile_hash(career_profile)
    unique_jobs, known_jobs, stale_jobs, duplicate_jobs = deduplicate_jobs(
        normalized_jobs, current_profile_hash, force_refresh
    )

    metrics = {
        "jobs_discovered": len(jobs),
        "selected_jobs": len(selected_jobs),
        "already_known_jobs": len(known_jobs),
        "stale_cache_hits": len(stale_jobs),
        "new_jobs": len(unique_jobs),
        "jobs_rejected_by_rules": 0,
        "jobs_below_threshold": 0,
        "jobs_filtered_rejected": len(duplicate_jobs),
        "jobs_filtered_out": len(duplicate_jobs),
        "duplicates_removed": len(duplicate_jobs),
        "total_new_jobs": len(unique_jobs),
        "jobs_pre_scored": 0,
        "jobs_sent_to_llm": 0,
        "jobs_added_to_queue": 0,
        "actual_llm_calls_used": 0,
        "skipped_llm_calls": 0,
        "cache_hits": len(known_jobs),
        "cache_misses": len(unique_jobs),
        "llm_calls_avoided": len(duplicate_jobs) + len(known_jobs),
        "estimated_token_savings": 0,
        "estimated_tokens_saved": 0,
        "token_savings_formula": f"llm_calls_avoided * {LLM_TOKEN_ESTIMATE_PER_FULL_JOB} tokens",
        "profile_changed_reanalysis_recommended": len(stale_jobs),
        "imported_jobs": normalized_jobs,
        "known_jobs": known_jobs,
        "stale_jobs": stale_jobs,
        "duplicate_jobs": duplicate_jobs,
        "rejected_by_rules_jobs": [],
        "below_threshold_jobs": [],
        "filtered_out_jobs": duplicate_jobs,
        "pre_scored_jobs": [],
        "jobs_sent_to_llm_rows": [],
        "jobs_skipped_llm_rows": [],
        "queued_jobs": [],
        "processed_jobs": [],
    }

    if len(jobs) > max_jobs_per_run:
        over_limit = len(jobs) - max_jobs_per_run
        metrics["jobs_filtered_rejected"] += over_limit
        metrics["jobs_filtered_out"] += over_limit
        metrics["llm_calls_avoided"] += over_limit
        metrics["estimated_token_savings"] += over_limit * LLM_TOKEN_ESTIMATE_PER_FULL_JOB

    for duplicate_job in duplicate_jobs:
        duplicate_item = empty_queue_item(duplicate_job)
        duplicate_item["decision_reason"] = duplicate_job["duplicate_reason"]
        duplicate_item["rejection_reason"] = duplicate_job["duplicate_reason"]
        duplicate_item["queue_category"] = "Filtered"
        metrics["processed_jobs"].append(duplicate_item)

    for known_job in known_jobs:
        cached_result = known_job.get("cached_result", {})
        known_item = empty_queue_item(known_job)
        known_item["fit_score"] = cached_result.get("fit_score", 0)
        known_item["pre_filter_score"] = cached_result.get("pre_filter_score", 0)
        known_item["apply_decision"] = cached_result.get("apply_decision", "")
        known_item["decision_reason"] = cached_result.get("decision_reason", "Reused cached analysis.")
        known_item["analysis_text"] = cached_result.get("analysis_text", "")
        known_item["queue_category"] = cached_result.get("queue_category", known_item["apply_decision"])
        known_item["status"] = "Reviewed"
        metrics["processed_jobs"].append(known_item)

    for stale_job in stale_jobs:
        stale_item = empty_queue_item(stale_job)
        stale_item["fit_score"] = stale_job.get("cached_result", {}).get("fit_score", 0)
        stale_item["pre_filter_score"] = stale_job.get("cached_result", {}).get("pre_filter_score", 0)
        stale_item["apply_decision"] = stale_job.get("cached_result", {}).get("apply_decision", "")
        stale_item["decision_reason"] = stale_job["stale_reason"]
        stale_item["analysis_text"] = stale_job.get("cached_result", {}).get("analysis_text", "")
        stale_item["queue_category"] = stale_job.get("cached_result", {}).get("queue_category", "")
        stale_item["status"] = "Reviewed"
        stale_item["rejection_reason"] = stale_job["stale_reason"]
        metrics["processed_jobs"].append(stale_item)

    for job in unique_jobs:
        structured_filter = should_filter_job(
            job,
            {
                "allow_senior_roles": allow_senior_roles,
                "career_profile": career_profile,
            },
        )
        if structured_filter["filtered"]:
            item = empty_queue_item(job)
            item["decision_reason"] = structured_filter["reason"]
            item["rejection_reason"] = structured_filter["reason"]
            item["queue_category"] = structured_filter["category"]
            metrics["jobs_rejected_by_rules"] += 1
            metrics["jobs_filtered_rejected"] += 1
            metrics["jobs_filtered_out"] += 1
            metrics["llm_calls_avoided"] += 1
            metrics["rejected_by_rules_jobs"].append(item)
            metrics["filtered_out_jobs"].append(item)
            metrics["processed_jobs"].append(item)
            continue

        item = evaluate_job_without_llm(job, career_profile, pre_filter_threshold, allow_senior_roles)
        if item["pre_filter_score"] > 0 or item["status"] in ["Reviewed", "Discovered"]:
            metrics["jobs_pre_scored"] += 1
            metrics["pre_scored_jobs"].append(item)

        if item["pre_filter_score"] < pre_filter_threshold:
            metrics["jobs_below_threshold"] += 1
            metrics["jobs_filtered_rejected"] += 1
            metrics["jobs_filtered_out"] += 1
            metrics["llm_calls_avoided"] += 1
            metrics["below_threshold_jobs"].append(item)
            metrics["filtered_out_jobs"].append(item)
            metrics["processed_jobs"].append(item)
            continue

        eligible_for_llm = (
            item["pre_filter_score"] >= pre_filter_threshold
            and bool(job.get("jd_text", "").strip())
            and metrics["jobs_sent_to_llm"] < max_llm_calls_per_run
        )

        if eligible_for_llm:
            item = evaluate_job_with_llm(job, item, user_profile, career_profile, career_profile_text)
            metrics["jobs_sent_to_llm"] += 1
            metrics["actual_llm_calls_used"] += 1
            metrics["jobs_sent_to_llm_rows"].append(item)
        else:
            llm_was_skipped = (
                item["pre_filter_score"] >= pre_filter_threshold
                and bool(job.get("jd_text", "").strip())
                and metrics["jobs_sent_to_llm"] >= max_llm_calls_per_run
            )
            if llm_was_skipped:
                metrics["skipped_llm_calls"] += 1
                metrics["jobs_skipped_llm_rows"].append(item)
            if not item["sent_to_llm"]:
                metrics["llm_calls_avoided"] += 1
                metrics["estimated_token_savings"] += LLM_TOKEN_ESTIMATE_PER_FULL_JOB

        item["queue_category"] = assign_queue_category(item)
        if item["queue_category"] in ["Apply", "Maybe"]:
            insert_job_queue_item(item)
            metrics["jobs_added_to_queue"] += 1
            metrics["queued_jobs"].append(item)
        else:
            item["rejection_reason"] = item["decision_reason"]
            metrics["jobs_filtered_rejected"] += 1
            metrics["jobs_filtered_out"] += 1
            metrics["filtered_out_jobs"].append(item)
        metrics["processed_jobs"].append(item)

    metrics["estimated_tokens_saved"] = metrics["llm_calls_avoided"] * LLM_TOKEN_ESTIMATE_PER_FULL_JOB
    metrics["estimated_token_savings"] = metrics["estimated_tokens_saved"]
    return metrics


def run_discovery_pipeline(jobs: list, settings: dict) -> dict:
    return process_discovered_jobs(
        jobs,
        settings.get("user_profile", ""),
        settings.get("career_profile", {}),
        settings.get("career_profile_text", ""),
        int(settings.get("pre_filter_threshold", DEFAULT_PRE_FILTER_THRESHOLD)),
        int(settings.get("max_llm_calls_per_run", DEFAULT_MAX_LLM_CALLS_PER_RUN)),
        int(settings.get("max_jobs_per_run", DEFAULT_MAX_JOBS_PER_RUN)),
        bool(settings.get("allow_senior_roles", False)),
        bool(settings.get("force_refresh", False)),
    )


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
                COALESCE(post_time, 'Unknown') AS post_time,
                COALESCE(job_level, 'Unknown') AS job_level,
                COALESCE(work_mode, 'Unknown') AS work_mode,
                COALESCE(NULLIF(queue_category, ''), apply_decision, '') AS queue_category,
                COALESCE(rejection_reason, '') AS rejection_reason,
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
            WHERE COALESCE(NULLIF(queue_category, ''), apply_decision, '') IN ('Apply', 'Maybe')
            ORDER BY
                CASE COALESCE(NULLIF(queue_category, ''), apply_decision, '')
                    WHEN 'Apply' THEN 1
                    WHEN 'Maybe' THEN 2
                    ELSE 4
                END,
                fit_score DESC,
                pre_filter_score DESC,
                created_at DESC
            """,
            conn,
        )


def deduplicate_job_queue() -> dict:
    init_db()
    with get_connection() as conn:
        rows = pd.read_sql_query(
            """
            SELECT
                id,
                company,
                COALESCE(job_title, title, '') AS job_title,
                location,
                job_url,
                fit_score,
                apply_decision,
                COALESCE(post_time, 'Unknown') AS post_time,
                COALESCE(job_level, 'Unknown') AS job_level,
                COALESCE(work_mode, 'Unknown') AS work_mode,
                COALESCE(updated_at, created_at, '') AS updated_at
            FROM job_queue
            """,
            conn,
        )
        if rows.empty:
            return {"duplicates_archived": 0, "active_records": 0}

        def dedup_key(row):
            url = normalize_text(row["job_url"]).lower()
            if url:
                return ("url", url)
            return (
                normalize_key(row["company"]),
                normalize_key(row["job_title"]),
                normalize_key(row["location"]),
            )

        def rank(row):
            return (
                1 if int(row.get("fit_score", 0) or 0) > 0 else 0,
                1 if row.get("apply_decision", "") in ["Apply", "Maybe"] else 0,
                sum(
                    1
                    for field in ["post_time", "job_level", "work_mode"]
                    if str(row.get(field, "") or "") not in ["", "Unknown"]
                ),
                str(row.get("updated_at", "") or ""),
                int(row.get("id", 0) or 0),
            )

        rows["dedup_key"] = rows.apply(dedup_key, axis=1)
        archived_ids = []
        for _, group in rows.groupby("dedup_key"):
            if len(group) <= 1:
                continue
            best_index = max(group.index.tolist(), key=lambda idx: rank(group.loc[idx]))
            archived_ids.extend(
                int(group.loc[idx, "id"])
                for idx in group.index.tolist()
                if idx != best_index
            )

        if archived_ids:
            placeholders = ", ".join(["?"] * len(archived_ids))
            conn.execute(
                f"""
                UPDATE job_queue
                SET
                    queue_category = 'Filtered',
                    status = 'Skipped',
                    decision_reason = CASE
                        WHEN COALESCE(decision_reason, '') = '' THEN 'Archived duplicate queue record'
                        ELSE decision_reason || '; Archived duplicate queue record'
                    END,
                    rejection_reason = 'Archived duplicate queue record',
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (now_iso(), *archived_ids),
            )
            conn.commit()
        active_records = conn.execute(
            """
            SELECT COUNT(*)
            FROM job_queue
            WHERE COALESCE(NULLIF(queue_category, ''), apply_decision, '') IN ('Apply', 'Maybe')
            """
        ).fetchone()[0]
    return {"duplicates_archived": len(archived_ids), "active_records": active_records}


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
