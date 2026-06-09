import json
import re
from hashlib import sha256
from datetime import datetime

from src.database import (
    fetch_career_feedback,
    fetch_career_profile,
    fetch_constraints,
    fetch_preferences,
    fetch_projects,
    fetch_skills,
    replace_constraints,
    replace_preferences,
    replace_projects,
    replace_skills,
    upsert_career_profile,
)
from src.llm import call_llm


CORE_PROFILE_KEYS = [
    "name",
    "headline",
    "summary",
    "education",
    "skills",
    "target_roles",
    "acceptable_roles",
    "preferred_locations",
    "excluded_roles",
    "visa_status",
    "salary_goal",
    "years_experience",
    "career_goal",
    "missing_skills",
    "suggested_locations",
    "suggested_career_paths",
]


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def clean_list(items) -> list[str]:
    if not items:
        return []
    if isinstance(items, str):
        items = re.split(r"[,;\n]", items)
    return [str(item).strip() for item in items if str(item).strip()]


def clean_rows(rows: list[dict], required_keys: list[str]) -> list[dict]:
    cleaned = []
    for row in rows or []:
        item = {key: str(row.get(key, "") or "").strip() for key in required_keys}
        if any(item.values()):
            cleaned.append(item)
    return cleaned


def derive_skill_summary(skill_rows: list[dict]) -> str:
    if not skill_rows:
        return ""
    grouped = {}
    for row in skill_rows:
        category = row.get("category", "General") or "General"
        grouped.setdefault(category, []).append(row.get("skill_name", ""))
    lines = []
    for category, skills in grouped.items():
        skill_names = ", ".join(skill for skill in skills if skill)
        if skill_names:
            lines.append(f"{category}: {skill_names}")
    return "\n".join(lines)


def compute_career_profile_hash(profile: dict) -> str:
    base = {
        "headline": str(profile.get("headline", "") or profile.get("summary", "") or "").strip(),
        "target_roles": str(profile.get("target_roles", "") or "").strip(),
        "acceptable_roles": str(profile.get("acceptable_roles", "") or "").strip(),
        "excluded_roles": str(profile.get("excluded_roles", "") or "").strip(),
        "preferred_locations": str(profile.get("preferred_locations", "") or "").strip(),
        "skills": str(profile.get("skills", "") or "").strip(),
        "constraints": [
            {
                "constraint_type": str(row.get("constraint_type", "") or "").strip(),
                "constraint_value": str(row.get("constraint_value", "") or "").strip(),
                "severity": str(row.get("severity", "") or "").strip(),
            }
            for row in (profile.get("constraints", []) or [])
            if any(str(row.get(key, "") or "").strip() for key in ["constraint_type", "constraint_value", "severity"])
        ],
    }
    payload = json.dumps(base, sort_keys=True, ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def normalize_generated_profile(payload: dict) -> dict:
    timestamp = datetime.now().isoformat(timespec="seconds")
    core = {}
    for key in CORE_PROFILE_KEYS:
        value = payload.get(key, "")
        if isinstance(value, list):
            core[key] = ", ".join(clean_list(value))
        else:
            core[key] = str(value or "").strip()

    skill_rows = clean_rows(
        payload.get("skills_inventory", []),
        ["category", "skill_name", "proficiency", "evidence"],
    )
    project_rows = clean_rows(
        payload.get("projects", []),
        [
            "project_name",
            "project_type",
            "business_problem",
            "technical_stack",
            "ai_methods",
            "business_impact",
            "target_roles_supported",
            "resume_bullets",
        ],
    )
    preference_rows = clean_rows(
        payload.get("preferences", []),
        ["preference_type", "preference_value", "weight"],
    )
    constraint_rows = clean_rows(
        payload.get("constraints", []),
        ["constraint_type", "constraint_value", "severity"],
    )

    if not core["skills"]:
        core["skills"] = derive_skill_summary(skill_rows)

    if not core["summary"]:
        core["summary"] = core.get("headline", "")

    core["generated_at"] = timestamp
    core["updated_at"] = timestamp
    return {
        **core,
        "skills_inventory": skill_rows,
        "projects": project_rows,
        "preferences": preference_rows,
        "constraints": constraint_rows,
        "profile_hash": "",
    }


def generate_career_profile_from_resume(resume_text: str) -> dict:
    system_prompt = """
You are an AI career strategist for early-career AI Engineer, GenAI Engineer,
AI Application Engineer, AI Solutions Engineer, and applied AI roles.

Extract a structured career intelligence profile from the resume.
Return strict JSON only. Be conservative and avoid inventing facts.
"""
    user_prompt = f"""
Resume text:
{resume_text}

Return JSON with exactly this shape:
{{
  "name": "",
  "headline": "",
  "summary": "",
  "education": "",
  "skills": "",
  "target_roles": ["", ""],
  "acceptable_roles": ["", ""],
  "preferred_locations": ["", ""],
  "excluded_roles": ["", ""],
  "visa_status": "",
  "salary_goal": "",
  "years_experience": "",
  "career_goal": "",
  "missing_skills": "",
  "suggested_locations": ["", ""],
  "suggested_career_paths": ["", ""],
  "skills_inventory": [
    {{
      "category": "",
      "skill_name": "",
      "proficiency": "",
      "evidence": ""
    }}
  ],
  "projects": [
    {{
      "project_name": "",
      "project_type": "",
      "business_problem": "",
      "technical_stack": "",
      "ai_methods": "",
      "business_impact": "",
      "target_roles_supported": "",
      "resume_bullets": ""
    }}
  ],
  "preferences": [
    {{
      "preference_type": "",
      "preference_value": "",
      "weight": ""
    }}
  ],
  "constraints": [
    {{
      "constraint_type": "",
      "constraint_value": "",
      "severity": ""
    }}
  ]
}}

Rules:
- target_roles: best-fit roles to pursue now.
- acceptable_roles: adjacent roles still worth considering.
- excluded_roles: roles that should usually be avoided.
- skills_inventory: include technical, AI, cloud, product, and business-facing skills when present.
- projects: focus on practical projects that prove AI application ability.
- preferences: location, team type, domain, work mode, compensation, learning goals if supported.
- constraints: visa, seniority, location, industry, or other blockers.
- If unknown, use an empty string instead of guessing.
"""
    payload = parse_json_object(call_llm(system_prompt, user_prompt))
    return normalize_generated_profile(payload)


def get_career_profile() -> dict:
    profile = fetch_career_profile()
    skill_rows = fetch_skills()
    project_rows = fetch_projects()
    preference_rows = fetch_preferences()
    constraint_rows = fetch_constraints()

    if not profile.get("skills"):
        profile["skills"] = derive_skill_summary(skill_rows)

    profile["skills_inventory"] = skill_rows
    profile["projects"] = project_rows
    profile["preferences"] = preference_rows
    profile["constraints"] = constraint_rows
    meaningful_values = [
        profile.get("name", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("skills", ""),
        profile.get("target_roles", ""),
        profile.get("acceptable_roles", ""),
        profile.get("preferred_locations", ""),
        profile.get("excluded_roles", ""),
    ]
    if any(str(value or "").strip() for value in meaningful_values) or skill_rows or project_rows or preference_rows or constraint_rows:
        profile["profile_hash"] = profile.get("profile_hash", "") or compute_career_profile_hash(profile)
    else:
        profile["profile_hash"] = ""
    return profile


def save_career_profile(profile: dict) -> None:
    core_profile = {key: str(profile.get(key, "") or "").strip() for key in CORE_PROFILE_KEYS}
    core_profile["summary"] = core_profile["summary"] or core_profile.get("headline", "")
    core_profile["generated_at"] = profile.get("generated_at", "")
    core_profile["updated_at"] = datetime.now().isoformat(timespec="seconds")

    if not core_profile["skills"]:
        core_profile["skills"] = derive_skill_summary(profile.get("skills_inventory", []))

    profile_with_hash = {
        **profile,
        **core_profile,
    }
    core_profile["profile_hash"] = compute_career_profile_hash(profile_with_hash)
    upsert_career_profile(core_profile)
    replace_skills(
        clean_rows(
            profile.get("skills_inventory", []),
            ["category", "skill_name", "proficiency", "evidence"],
        )
    )
    replace_projects(
        clean_rows(
            profile.get("projects", []),
            [
                "project_name",
                "project_type",
                "business_problem",
                "technical_stack",
                "ai_methods",
                "business_impact",
                "target_roles_supported",
                "resume_bullets",
            ],
        )
    )
    replace_preferences(
        clean_rows(
            profile.get("preferences", []),
            ["preference_type", "preference_value", "weight"],
        )
    )
    replace_constraints(
        clean_rows(
            profile.get("constraints", []),
            ["constraint_type", "constraint_value", "severity"],
        )
    )


def get_career_feedback_history():
    return fetch_career_feedback()
