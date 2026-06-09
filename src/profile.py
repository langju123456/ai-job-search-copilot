import json
import re
from datetime import datetime

from src.database import fetch_career_profile, upsert_career_profile
from src.llm import call_llm


PROFILE_KEYS = [
    "name",
    "summary",
    "education",
    "skills",
    "target_roles",
    "preferred_locations",
    "excluded_roles",
    "visa_status",
    "years_experience",
    "career_goal",
    "missing_skills",
    "suggested_locations",
    "suggested_career_paths",
]


def get_career_profile() -> dict:
    return fetch_career_profile()


def save_career_profile(profile: dict) -> None:
    upsert_career_profile(profile)


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def generate_career_profile_from_resume(resume_text: str) -> dict:
    system_prompt = """
You are an AI career strategist for early-career AI Engineer, GenAI Engineer,
AI Application Engineer, AI Solutions Engineer, and AI Application Builder roles.
Extract a practical career profile from the resume. Return strict JSON only.
"""
    user_prompt = f"""
Resume text:
{resume_text}

Return JSON with exactly these keys:
name,
summary,
education,
skills,
target_roles,
preferred_locations,
excluded_roles,
visa_status,
years_experience,
career_goal,
missing_skills,
suggested_locations,
suggested_career_paths.

Guidance:
- summary: concise professional summary.
- skills: grouped inventory including Technical Skills, AI Skills, Cloud Skills.
- missing_skills: skills to build for AI Engineer / GenAI Engineer roles.
- target_roles: comma-separated recommended roles.
- preferred_locations: infer from resume if possible, otherwise include Remote.
- excluded_roles: roles that do not match the stated path.
- visa_status: infer only if explicitly mentioned; otherwise Unknown.
- years_experience: estimate conservatively from resume content.
- suggested_career_paths: practical next paths toward AI application engineering.
"""
    generated = parse_json_object(call_llm(system_prompt, user_prompt))
    timestamp = datetime.now().isoformat(timespec="seconds")
    profile = {key: str(generated.get(key, "") or "") for key in PROFILE_KEYS}
    profile["generated_at"] = timestamp
    profile["updated_at"] = timestamp
    return profile
