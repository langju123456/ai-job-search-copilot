# SPEC.md — AI Job Search Copilot v0.1

## Project Goal

Build a local MVP of an AI Job Search Copilot that helps a job seeker analyze job descriptions, evaluate job fit, identify skill gaps, generate tailored resume bullets, draft networking messages, and track applications.

This project is designed for:

1. Helping the user find AI Engineer / GenAI Engineer / AI Application Engineer roles.
2. Serving as a GitHub portfolio project.
3. Demonstrating AI application engineering skills.

## Tech Stack

Use:

* Python
* Streamlit
* SQLite
* OpenAI API
* Pandas
* python-dotenv

Do not use:

* LangGraph
* FastAPI
* Postgres
* AWS
* Docker
* Web scraping
* Auto-apply features

Those are for later versions.

## File Structure

```text
ai-job-search-copilot/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── SPEC.md
├── data/
│   └── sample_jobs.csv
├── src/
│   ├── llm.py
│   ├── database.py
│   ├── job_analyzer.py
│   ├── resume_tailor.py
│   ├── networking.py
│   └── tracker.py
└── docs/
    ├── business_architecture.md
    ├── decision_architecture.md
    └── mvp_scope.md
```

## Core Features

### Feature 1: User Profile Input

The app should allow the user to paste their profile/resume into a text area.

Store this profile in session state.

### Feature 2: Job Description Analyzer

The user pastes a job description.

The app calls the LLM and returns:

* Fit score from 0 to 100
* Match level: Strong / Medium / Weak
* Top matching skills
* Missing skills
* Why this job is a good fit
* Why this job may not be a good fit
* Final recommendation: Apply / Maybe / Skip

### Feature 3: Resume Tailor

Based on the user profile and job description, generate:

* 5 tailored resume bullet points
* 3 project highlights to emphasize
* 5 keywords to include in the resume

### Feature 4: Networking Message Generator

Generate:

* LinkedIn connection request
* Recruiter message
* Follow-up message

Tone:

* Direct
* Professional
* Ambitious
* Not desperate
* Focused on AI application engineering and real business impact

### Feature 5: Application Tracker

Use SQLite to save:

* Company
* Job title
* Location
* Job URL
* Fit score
* Recommendation
* Status
* Notes
* Created timestamp

Status options:

* Saved
* Applied
* Networking
* Interview
* Rejected
* Offer

The app should display saved jobs in a table.

## SQLite Schema

Create a table named `applications`.

Fields:

```sql
id INTEGER PRIMARY KEY AUTOINCREMENT,
company TEXT,
job_title TEXT,
location TEXT,
job_url TEXT,
fit_score INTEGER,
recommendation TEXT,
status TEXT,
notes TEXT,
created_at TEXT
```

## OpenAI Setup

Use `.env`:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Create `src/llm.py` with a helper function:

```python
call_llm(system_prompt: str, user_prompt: str) -> str
```

## Streamlit UI

The app should have these sections:

1. Header: AI Job Search Copilot
2. User Profile / Resume input
3. Job Description input
4. Analyze Job button
5. Resume Tailoring output
6. Networking Messages output
7. Save Application form
8. Application Tracker table

## Prompt Requirements

The LLM should behave like an AI career strategist for early-career AI Engineer roles.

It should evaluate jobs based on:

* AI relevance
* Skill match
* Growth potential
* Portfolio alignment
* Practical probability of getting interviews
* Whether the role helps the user move toward AI Engineer / GenAI Engineer / AI Application Engineer

The LLM should not give generic advice.

Outputs should be structured and easy to copy.

## Error Handling

If API key is missing, show a clear Streamlit warning.

If profile or JD is empty, ask the user to fill it in.

If LLM call fails, show the error message in the app.

## README Requirements

README should include:

* Project overview
* Problem solved
* Features
* Tech stack
* Architecture
* How to run locally
* Screenshots placeholder
* Future roadmap

## Run Command

The project should run with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Success Criteria

The MVP is complete when:

1. User can paste profile and JD.
2. App generates job fit analysis.
3. App generates tailored resume bullets.
4. App generates networking messages.
5. User can save a job application.
6. Saved jobs appear in a tracker table.
7. Project can be pushed to GitHub and run locally.
