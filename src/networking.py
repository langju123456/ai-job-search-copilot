from src.llm import call_llm


SYSTEM_PROMPT = """
You write personalized networking messages for early-career AI Engineer,
GenAI Engineer, and AI Application Engineer job searches.

Tone must be direct, professional, ambitious, not desperate, and focused on AI
application engineering and real business impact. Avoid generic AI wording,
empty enthusiasm, and vague claims.
"""


def generate_networking_messages(
    user_profile: str, job_description: str, career_profile: str = ""
) -> str:
    user_prompt = f"""
Career profile:
{career_profile}

Candidate profile/resume:
{user_profile}

Job description:
{job_description}

Generate personalized networking messages using exactly these sections:

## Recruiter Message
[message]

## Hiring Manager Message
[message]

## Engineer / Alumni Message
[message]
"""
    return call_llm(SYSTEM_PROMPT, user_prompt)
