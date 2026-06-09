from src.llm import call_llm


SYSTEM_PROMPT = """
You are an AI career strategist for early-career AI Engineer, GenAI Engineer,
and AI Application Engineer roles.

Create resume content that is specific to the candidate and job description.
Focus on AI application engineering, practical systems, measurable business
impact, product thinking, and portfolio alignment. Avoid generic advice.
"""


def tailor_resume(
    user_profile: str,
    job_description: str,
    career_profile: str = "",
    selected_assets_text: str = "",
    mode: str = "full_llm",
) -> str:
    user_prompt = f"""
Career profile:
{career_profile}

Candidate profile/resume:
{user_profile}

Job description:
{job_description}

Resume assets:
{selected_assets_text}

Generate tailored resume content using exactly these sections:

## Tailored Resume Bullet Points
- [bullet 1]
- [bullet 2]
- [bullet 3]
- [bullet 4]
- [bullet 5]

## Project Highlights To Emphasize
- [project highlight 1]
- [project highlight 2]
- [project highlight 3]

## Resume Keywords
- [keyword 1]
- [keyword 2]
- [keyword 3]
- [keyword 4]
- [keyword 5]
"""
    if mode == "asset_first" and selected_assets_text.strip():
        user_prompt += """

Rules:
- Reuse and polish the supplied resume assets first.
- Do not invent a brand-new resume story when the assets already cover the evidence.
- Prefer the strongest matching assets and rewrite them to fit the job.
"""
    return call_llm(SYSTEM_PROMPT, user_prompt)
