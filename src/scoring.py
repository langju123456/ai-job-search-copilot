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


def split_terms(value: str) -> list[str]:
    if not value:
        return []
    return [term.strip() for term in re.split(r"[,;\n]", value) if term.strip()]


def select_relevant_projects(profile: dict, job_text: str, limit: int = 3) -> list[dict]:
    projects = profile.get("projects", []) or []
    if not projects:
        return []

    job_text_lower = (job_text or "").lower()
    scored_projects = []
    for project in projects:
        blob = " ".join(
            [
                str(project.get("project_name", "") or ""),
                str(project.get("project_type", "") or ""),
                str(project.get("technical_stack", "") or ""),
                str(project.get("ai_methods", "") or ""),
                str(project.get("business_impact", "") or ""),
                str(project.get("target_roles_supported", "") or ""),
            ]
        ).lower()
        overlap = sum(1 for token in set(re.findall(r"[a-z0-9\+\#\.]+", blob)) if token and token in job_text_lower)
        scored_projects.append((overlap, project))

    scored_projects.sort(key=lambda item: item[0], reverse=True)
    selected = [project for score, project in scored_projects if score > 0][:limit]
    if selected:
        return selected
    return projects[:limit]


def top_skills(profile: dict, limit: int = 8) -> list[str]:
    rows = profile.get("skills_inventory", []) or []
    if rows:
        skills = [str(row.get("skill_name", "") or "").strip() for row in rows]
        return [skill for skill in skills if skill][:limit]
    return split_terms(profile.get("skills", ""))[:limit]


def compact_career_profile_text(profile: dict, job_text: str = "") -> str:
    if not profile:
        return ""

    relevant_projects = select_relevant_projects(profile, job_text)
    preferences = profile.get("preferences", []) or []
    constraints = profile.get("constraints", []) or []

    lines = [
        f"Name: {profile.get('name', '')}",
        f"Headline: {profile.get('headline', '') or profile.get('summary', '')}",
        f"Education: {profile.get('education', '')}",
        f"Target roles: {profile.get('target_roles', '')}",
        f"Acceptable roles: {profile.get('acceptable_roles', '')}",
        f"Preferred locations: {profile.get('preferred_locations', '')}",
        f"Excluded roles: {profile.get('excluded_roles', '')}",
        f"Visa status: {profile.get('visa_status', '')}",
        f"Salary goal: {profile.get('salary_goal', '')}",
        f"Years of experience: {profile.get('years_experience', '')}",
        f"Career goal: {profile.get('career_goal', '')}",
        f"Missing skills: {profile.get('missing_skills', '')}",
        f"Top skills: {', '.join(top_skills(profile))}",
    ]
    if preferences:
        lines.append(
            "Preferences: "
            + "; ".join(
                f"{row.get('preference_type', '')}: {row.get('preference_value', '')} ({row.get('weight', '')})"
                for row in preferences[:5]
                if row.get("preference_type") or row.get("preference_value")
            )
        )
    if constraints:
        lines.append(
            "Constraints: "
            + "; ".join(
                f"{row.get('constraint_type', '')}: {row.get('constraint_value', '')} ({row.get('severity', '')})"
                for row in constraints[:5]
                if row.get("constraint_type") or row.get("constraint_value")
            )
        )
    if relevant_projects:
        project_lines = []
        for project in relevant_projects:
            project_lines.append(
                f"{project.get('project_name', '')}: "
                f"{project.get('technical_stack', '')}; "
                f"{project.get('ai_methods', '')}; "
                f"{project.get('business_impact', '')}"
            )
        lines.append("Relevant projects: " + " | ".join(project_lines))
    return "\n".join(line for line in lines if not line.endswith(": "))


def career_profile_to_text(profile: dict) -> str:
    return compact_career_profile_text(profile)
