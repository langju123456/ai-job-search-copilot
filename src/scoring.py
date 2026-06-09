import re

import pandas as pd


SCORE_COMPONENTS = {
    "skill_match": ("Skill Match", 30),
    "experience_match": ("Experience Match", 25),
    "domain_match": ("Domain Match", 20),
    "career_goal_alignment": ("Career Goal Alignment", 15),
    "growth_potential": ("Growth Potential", 10),
}


def clamp_score(value: int, maximum: int) -> int:
    return max(0, min(maximum, value))


def parse_score_breakdown(analysis: str) -> dict:
    scores = {}
    for key, (label, maximum) in SCORE_COMPONENTS.items():
        pattern = rf"{re.escape(label)}\s*:\s*(\d{{1,3}})\s*/\s*{maximum}"
        match = re.search(pattern, analysis, re.IGNORECASE)
        scores[key] = clamp_score(int(match.group(1)), maximum) if match else 0
    scores["fit_score"] = sum(scores.values())
    return scores


def score_breakdown_dataframe(scores: dict) -> pd.DataFrame:
    rows = []
    for key, (label, maximum) in SCORE_COMPONENTS.items():
        rows.append(
            {
                "Component": label,
                "Score": scores.get(key, 0),
                "Max": maximum,
            }
        )
    return pd.DataFrame(rows)


def determine_priority(fit_score: int, recommendation: str) -> str:
    if fit_score >= 80 and recommendation == "Apply":
        return "Must Apply"
    if fit_score >= 60:
        return "Good Opportunity"
    return "Low Priority"


def career_profile_to_text(profile: dict) -> str:
    if not profile:
        return ""

    lines = [
        f"Name: {profile.get('name', '')}",
        f"Professional summary: {profile.get('summary', '')}",
        f"Education: {profile.get('education', '')}",
        f"Skills: {profile.get('skills', '')}",
        f"Target roles: {profile.get('target_roles', '')}",
        f"Preferred locations: {profile.get('preferred_locations', '')}",
        f"Excluded roles: {profile.get('excluded_roles', '')}",
        f"Visa status: {profile.get('visa_status', '')}",
        f"Salary goal: {profile.get('salary_goal', '')}",
        f"Years of experience: {profile.get('years_experience', '')}",
        f"Career goal: {profile.get('career_goal', '')}",
        f"Missing skills: {profile.get('missing_skills', '')}",
        f"Suggested locations: {profile.get('suggested_locations', '')}",
        f"Suggested career paths: {profile.get('suggested_career_paths', '')}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))
