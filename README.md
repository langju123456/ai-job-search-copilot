# AI Job Search Copilot

AI Job Search Copilot is a local Streamlit MVP that helps a job seeker evaluate AI Engineer, GenAI Engineer, and AI Application Engineer roles.

## Problem Solved

Job seekers often spend too much time deciding whether a role is worth applying to, how to tailor their resume, and what to say when networking. This app turns a pasted resume and job description into structured job-fit analysis, tailored resume content, networking messages, and a saved application tracker.

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
- Compute structured fit scores from skill, experience, domain, career alignment, and growth components
- Prioritize saved jobs into Must Apply, Good Opportunity, and Low Priority buckets
- Track CRM-style fields such as recruiter name, follow-up date, next action, and interview stage
- Export resume suggestions and networking messages as `.txt` files
- Store local data collection records for users, companies, jobs, model runs, outcomes, and feedback
- Update application outcomes and capture lightweight analysis feedback
- View analytics for analyzed jobs, applications, interview rate, offer rate, missing skills, and top scoring companies
- Discover jobs from pasted URLs, pasted job descriptions, or CSV uploads
- Score discovered jobs and route them into an apply decision queue
- Generate application prep for Apply decisions without submitting applications

## Tech Stack

- Python
- Streamlit
- SQLite
- OpenAI API
- Pandas
- python-dotenv

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
- `src/data_collection.py` records local learning data and analytics queries
- `src/job_discovery.py` powers job discovery, apply decisions, queue status, and application prep

## MVP Data Note

Data is stored locally in SQLite. User data is only sent to the configured LLM API for analysis. There is no multi-user login or authentication in this MVP.

## Job Discovery CSV Format

The Job Discovery page accepts CSV uploads with these columns:

```text
company,job_title,location,job_url,jd_text
```

The app does not scrape job URLs and does not submit applications automatically.

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
