from src.llm import call_llm


SYSTEM_PROMPT = """
You are an AI career strategist for early-career AI Engineer, GenAI Engineer,
and AI Application Engineer roles.

Evaluate the job using the required scoring framework. The component scores
must be realistic, defensible, and based on the candidate profile, career
profile, and job description.

Evaluate relative to the user's actual background, skills, goals, constraints,
target roles, location preferences, seniority fit, and missing skills. Do not
score a job highly just because the job is generally strong.

The fit score must reflect:
- target role match
- acceptable role adjacency
- excluded role penalties
- skills overlap
- project evidence
- location preference
- visa constraint
- seniority match
- career goal alignment

Do not give generic advice. Be specific, direct, structured, and easy to copy.
"""


def analyze_job(
    user_profile: str,
    job_description: str,
    career_profile: str = "",
    resume_assets: str = "",
) -> str:
    user_prompt = f"""
Career profile:
{career_profile}

Candidate profile/resume:
{user_profile}

Relevant resume assets:
{resume_assets}

Job description:
{job_description}

Return the analysis using exactly these sections:

## Score Breakdown
- Skill Match: [0-30]/30
- Experience Match: [0-25]/25
- Domain Match: [0-20]/20
- Career Goal Alignment: [0-15]/15
- Growth Potential: [0-10]/10

## Fit Score
- Computed Total: [sum of score breakdown]/100
- Match Level: [Strong / Medium / Weak]

## Top Matching Skills
- [skill 1]
- [skill 2]
- [skill 3]
- [skill 4]
- [skill 5]

## Missing Skills
- [skill 1]
- [skill 2]
- [skill 3]

## Why This Job Is A Good Fit
- [specific reason]
- [specific reason]
- [specific reason]

## Why This Job May Not Be A Good Fit
- [specific risk]
- [specific risk]

## Fit Reasoning
- Target role alignment: [specific reasoning]
- Acceptable vs excluded roles: [specific reasoning]
- Skills overlap: [specific reasoning]
- Project evidence: [specific reasoning]
- Constraints and location: [specific reasoning]
- Missing skills impact: [specific reasoning]
- Location and seniority fit: [specific reasoning]
- AI relevance and growth potential: [specific reasoning]

## Final Recommendation
- Recommendation: [Apply / Maybe / Skip]
- Reason: [concise explanation]
"""
    return call_llm(SYSTEM_PROMPT, user_prompt)
