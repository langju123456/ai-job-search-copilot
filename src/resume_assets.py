import json
import re
from datetime import datetime

import pandas as pd

from src.database import get_connection, init_db
from src.llm import call_llm


ASSET_TYPES = [
    "bullet",
    "project_summary",
    "technical_achievement",
    "business_impact",
    "leadership_example",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def split_terms(value: str) -> list[str]:
    if not value:
        return []
    return [term.strip() for term in re.split(r"[,;\n]", str(value or "")) if term.strip()]


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def fetch_resume_assets() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                id,
                asset_type,
                title,
                content,
                skills,
                projects,
                target_roles,
                evidence_strength,
                created_at,
                updated_at
            FROM resume_assets
            ORDER BY updated_at DESC, id DESC
            """,
            conn,
        )


def save_resume_assets(rows: list[dict]) -> None:
    init_db()
    timestamp = now_iso()
    clean_rows = []
    for row in rows or []:
        clean_row = {
            "id": row.get("id", ""),
            "asset_type": normalize_text(row.get("asset_type", "")),
            "title": normalize_text(row.get("title", "")),
            "content": str(row.get("content", "") or "").strip(),
            "skills": str(row.get("skills", "") or "").strip(),
            "projects": str(row.get("projects", "") or "").strip(),
            "target_roles": str(row.get("target_roles", "") or "").strip(),
            "evidence_strength": normalize_text(row.get("evidence_strength", "")),
            "created_at": str(row.get("created_at", "") or "").strip(),
            "updated_at": timestamp,
        }
        if any(
            clean_row[key]
            for key in [
                "asset_type",
                "title",
                "content",
                "skills",
                "projects",
                "target_roles",
                "evidence_strength",
            ]
        ):
            clean_rows.append(clean_row)

    with get_connection() as conn:
        existing_ids = {
            row[0] for row in conn.execute("SELECT id FROM resume_assets").fetchall()
        }
        kept_ids = set()
        for row in clean_rows:
            row_id = row.get("id")
            if str(row_id).strip():
                kept_ids.add(int(row_id))
                conn.execute(
                    """
                    UPDATE resume_assets
                    SET
                        asset_type = ?,
                        title = ?,
                        content = ?,
                        skills = ?,
                        projects = ?,
                        target_roles = ?,
                        evidence_strength = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        row["asset_type"],
                        row["title"],
                        row["content"],
                        row["skills"],
                        row["projects"],
                        row["target_roles"],
                        row["evidence_strength"],
                        row["updated_at"],
                        int(row_id),
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO resume_assets (
                        asset_type,
                        title,
                        content,
                        skills,
                        projects,
                        target_roles,
                        evidence_strength,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["asset_type"],
                        row["title"],
                        row["content"],
                        row["skills"],
                        row["projects"],
                        row["target_roles"],
                        row["evidence_strength"],
                        row["created_at"] or timestamp,
                        row["updated_at"],
                    ),
                )
                kept_ids.add(int(cursor.lastrowid))

        removed_ids = existing_ids - kept_ids
        if removed_ids:
            placeholders = ", ".join(["?"] * len(removed_ids))
            conn.execute(
                f"DELETE FROM resume_assets WHERE id IN ({placeholders})",
                tuple(sorted(removed_ids)),
            )
        conn.commit()


def generate_assets_from_career_profile(profile: dict) -> list[dict]:
    projects = profile.get("projects", []) or []
    if not projects:
        return []

    generated_assets = []
    system_prompt = """
You create reusable resume assets for AI Engineer and AI Application Engineer job searches.
Return strict JSON only.
"""
    for project in projects:
        project_name = project.get("project_name", "") or "Untitled Project"
        user_prompt = f"""
Career profile:
- Headline: {profile.get('headline', '') or profile.get('summary', '')}
- Target roles: {profile.get('target_roles', '')}
- Skills: {profile.get('skills', '')}

Project:
- Project name: {project_name}
- Project type: {project.get('project_type', '')}
- Business problem: {project.get('business_problem', '')}
- Technical stack: {project.get('technical_stack', '')}
- AI methods: {project.get('ai_methods', '')}
- Business impact: {project.get('business_impact', '')}
- Supported roles: {project.get('target_roles_supported', '')}
- Existing resume bullets: {project.get('resume_bullets', '')}

Return JSON with one key: assets.
assets must be a list of exactly 6 items:
- 3 technical_achievement assets
- 2 business_impact assets
- 1 leadership_example asset

Each asset item must contain:
asset_type, title, content, skills, projects, target_roles, evidence_strength.

Rules:
- Make assets reusable across future jobs.
- Keep content to one bullet-length sentence.
- skills should be a comma-separated list.
- projects should be the project name.
- target_roles should be a comma-separated list.
- evidence_strength should be High, Medium, or Low.
"""
        payload = parse_json_object(call_llm(system_prompt, user_prompt))
        for asset in payload.get("assets", []):
            generated_assets.append(
                {
                    "asset_type": asset.get("asset_type", "bullet"),
                    "title": asset.get("title", project_name),
                    "content": asset.get("content", ""),
                    "skills": asset.get("skills", project.get("technical_stack", "")),
                    "projects": asset.get("projects", project_name),
                    "target_roles": asset.get(
                        "target_roles",
                        project.get("target_roles_supported", profile.get("target_roles", "")),
                    ),
                    "evidence_strength": asset.get("evidence_strength", "Medium"),
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                }
            )
    return generated_assets


def extract_keywords(text: str) -> set[str]:
    stopwords = {
        "and",
        "the",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "your",
        "role",
        "team",
        "work",
        "using",
        "build",
        "will",
        "are",
        "you",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\+\#\-]{1,}", str(text or "").lower())
    return {word for word in words if word not in stopwords}


def score_resume_asset(asset: dict, job_description: str, career_profile: dict) -> int:
    asset_text = " ".join(
        [
            str(asset.get("title", "") or ""),
            str(asset.get("content", "") or ""),
            str(asset.get("skills", "") or ""),
            str(asset.get("projects", "") or ""),
            str(asset.get("target_roles", "") or ""),
        ]
    )
    asset_keywords = extract_keywords(asset_text)
    job_keywords = extract_keywords(job_description)
    profile_roles = extract_keywords(
        " ".join(
            [
                career_profile.get("target_roles", ""),
                career_profile.get("acceptable_roles", ""),
            ]
        )
    )
    profile_skills = extract_keywords(career_profile.get("skills", ""))

    score = 0
    score += min(40, 5 * len(asset_keywords & job_keywords))
    score += min(20, 4 * len(extract_keywords(asset.get("skills", "")) & job_keywords))
    score += min(15, 5 * len(extract_keywords(asset.get("target_roles", "")) & profile_roles))
    score += min(15, 3 * len(extract_keywords(asset.get("projects", "")) & job_keywords))
    score += min(10, 2 * len(extract_keywords(asset.get("skills", "")) & profile_skills))

    evidence_bonus = {
        "high": 10,
        "medium": 6,
        "low": 2,
    }
    score += evidence_bonus.get(str(asset.get("evidence_strength", "")).lower(), 0)
    return score


def retrieve_relevant_resume_assets(
    job_description: str, career_profile: dict, limit: int = 5
) -> list[dict]:
    assets_df = fetch_resume_assets()
    if assets_df.empty:
        return []

    rows = assets_df.to_dict("records")
    scored_assets = []
    for row in rows:
        score = score_resume_asset(row, job_description, career_profile)
        if score > 0:
            row["match_score"] = score
            scored_assets.append(row)
    scored_assets.sort(key=lambda item: item.get("match_score", 0), reverse=True)
    return scored_assets[:limit]


def resume_assets_to_text(assets: list[dict]) -> str:
    if not assets:
        return ""
    lines = []
    for index, asset in enumerate(assets[:5], start=1):
        lines.append(
            f"{index}. [{asset.get('asset_type', 'bullet')}] {asset.get('title', '')}\n"
            f"Content: {asset.get('content', '')}\n"
            f"Skills: {asset.get('skills', '')}\n"
            f"Projects: {asset.get('projects', '')}\n"
            f"Target roles: {asset.get('target_roles', '')}\n"
            f"Evidence strength: {asset.get('evidence_strength', '')}"
        )
    return "\n\n".join(lines)
