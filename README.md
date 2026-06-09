# AI Job Search Copilot

AI Job Search Copilot is a local Streamlit MVP that helps a job seeker evaluate AI Engineer, GenAI Engineer, and AI Application Engineer roles.

## Problem Solved

Job seekers often spend too much time deciding whether a role is worth applying to, how to tailor their resume, and what to say when networking. This app turns a resume, a structured career profile, and job data into structured job-fit analysis, token-efficient discovery decisions, tailored resume content, networking messages, and a saved application tracker.

## Features

- Paste a user profile or resume into session state
- Upload PDF, DOCX, or TXT resume/profile files
- Upload PDF, DOCX, or TXT job description files
- Analyze a job description for fit score, match level, skill match, missing skills, fit risks, and recommendation
- Generate tailored resume bullets, project highlights, and resume keywords
- Generate LinkedIn, recruiter, and follow-up networking messages
- Save applications to SQLite
- View saved applications in a tracker table
- Store a career profile locally in SQLite
- Generate a structured career profile from uploaded or pasted resume text
- Edit structured career memory for skills, projects, preferences, and constraints
- Use the saved career profile to guide profile-aware job discovery, fit analysis, and queue decisions
- Compute structured fit scores from skill, experience, domain, career alignment, and growth components
- Prioritize saved jobs into Must Apply, Good Opportunity, and Low Priority buckets
- Track CRM-style fields such as recruiter name, follow-up date, next action, and interview stage
- Export resume suggestions and networking messages as `.txt` files
- Store local data collection records for users, companies, jobs, model runs, outcomes, and feedback
- Update application outcomes and capture lightweight analysis feedback
- Maintain a local outcome and future-learning foundation through outcomes, model runs, and user feedback
- View analytics for analyzed jobs, applications, interview rate, offer rate, missing skills, and top scoring companies
- Discover jobs from pasted URLs, pasted job descriptions, or CSV uploads
- Discover jobs from public career/job board pages with requests and BeautifulSoup
- Run sample discovery mode with a local sample dataset
- Score discovered jobs and route them into an apply decision queue
- Generate application prep for Apply decisions without submitting applications
- Build and reuse a Resume Asset Library from structured projects and reusable bullets
- Run asset-first application prep that retrieves top resume assets before using an LLM rewrite call
- Run token-efficient batch evaluation with normalization, deduplication, rule-based filtering, keyword pre-scoring, and capped LLM deep analysis
- Export discovered jobs and the Apply queue as CSV files
- Reuse cached queue results for already-known jobs and avoid repeated LLM analysis

## Tech Stack

- Python
- Streamlit
- SQLite
- OpenAI API
- Pandas
- python-dotenv
- requests
- BeautifulSoup

## Architecture

- `app.py` contains the Streamlit interface
- `src/llm.py` loads environment variables and calls the OpenAI API
- `src/job_analyzer.py` creates job-fit analysis prompts
- `src/resume_tailor.py` creates tailored resume prompts
- `src/networking.py` creates networking message prompts
- `src/database.py` manages SQLite setup and queries
- `src/tracker.py` provides application tracking helpers
- `src/scoring.py` parses structured score components and computes priority
- `src/profile.py` reads and writes career profile memory
- `src/profile.py` generates and persists structured career intelligence
- `src/data_collection.py` records local learning data and analytics queries
- `src/job_discovery.py` powers job discovery, apply decisions, queue status, and application prep

## MVP Data Note

Data is stored locally in SQLite. User data is only sent to the configured LLM API for analysis. There is no multi-user login or authentication in this MVP. The local database acts as memory for career profile data, applications, outcomes, and future learning signals.

## Job Discovery CSV Format

The Job Discovery page accepts CSV uploads with these columns:

```text
company,job_title,location,job_url,jd_text
```

The app can parse public HTML pages, but it does not scrape LinkedIn directly, bypass captchas, use browser automation, require login, or submit applications automatically.

## Token-Efficient Job Queue

The Job Discovery page avoids sending every job to the LLM. It first normalizes and deduplicates jobs, then applies rule-based filtering and weighted keyword pre-scoring. Only jobs above the configured pre-filter threshold and within the `max_llm_calls_per_run` budget are sent to OpenAI for deep analysis.

## Public Job Discovery

The app supports public career pages, public job board/search result pages, CSV uploads, pasted jobs, pasted URLs, and sample dataset mode. Discovered jobs are normalized, deduplicated, filtered cheaply, and then routed into the same Job Queue pipeline.

## Career Profile Engine

The Career Profile page can parse a PDF, DOCX, TXT, or pasted resume and generate a structured local profile with headline, summary, education, skills, target roles, acceptable roles, preferred locations, missing skills, projects, preferences, constraints, and career paths. Job discovery uses this saved profile so queue decisions are based on the user's actual background, goals, and constraints.

## Resume Asset Library

The Resume Assets page stores reusable bullets, project summaries, technical achievements, business impact statements, and leadership examples in local SQLite memory. Application Prep can run in asset-first mode so the app retrieves the top matching assets first, then uses the LLM only to polish the selected material for a specific job.

## Local Memory And Data Flywheel

SQLite acts as local memory for the career profile, job queue, resume assets, applications, outcomes, model runs, and user feedback. That creates a local outcome-learning foundation and future data flywheel without adding external services or multi-user auth.

## How To Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file:

```bash
cp .env.example .env
```

3. Add your OpenAI API key to `.env`:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

4. Start the app:

```bash
streamlit run app.py
```

## Screenshots

Screenshots placeholder.

## Future Roadmap

- Role scoring history
- Resume version management
- Company/contact tracking
- Interview preparation
- More structured exports
