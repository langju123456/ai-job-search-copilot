import re
from datetime import datetime
from json import JSONDecodeError

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data_collection import (
    OUTCOME_STATUS_OPTIONS,
    create_job,
    create_outcome,
    fetch_analytics_summary,
    fetch_average_fit_score_by_outcome,
    fetch_most_common_missing_skills,
    fetch_top_scoring_companies,
    get_or_create_company,
    record_analysis_run,
    save_user_feedback,
    update_outcome,
)
from src.file_parser import extract_text_from_uploaded_file
from src.job_analyzer import analyze_job
from src.database import get_connection, init_db
from src.job_discovery import (
    DEFAULT_MAX_LLM_CALLS_PER_RUN,
    DEFAULT_MAX_JOBS_PER_RUN,
    DEFAULT_PRE_FILTER_THRESHOLD,
    JOB_QUEUE_STATUS_OPTIONS,
    deduplicate_job_queue,
    discover_from_public_url,
    fetch_job_queue,
    generate_application_prep,
    load_sample_discovered_jobs,
    parse_manual_jobs,
    parse_csv_jobs,
    parse_job_urls,
    record_discovery_run,
    run_discovery_pipeline,
    update_application_prep,
    update_job_queue_status,
    upsert_discovered_source,
)
from src.llm import get_openai_api_key, get_openai_model
from src.networking import generate_networking_messages
from src.profile import (
    compute_career_profile_hash,
    generate_career_profile_from_resume,
    get_career_feedback_history,
    get_career_profile,
    save_career_profile,
)
from src.resume_assets import (
    ASSET_TYPES,
    fetch_resume_assets,
    generate_assets_from_career_profile,
    resume_assets_to_text,
    retrieve_relevant_resume_assets,
    save_resume_assets,
)
from src.resume_tailor import tailor_resume
from src.scoring import (
    compact_career_profile_text,
    career_profile_to_text,
    determine_priority,
    parse_score_breakdown,
    score_breakdown_dataframe,
)
from src.tracker import STATUS_OPTIONS, get_applications, save_application


load_dotenv()


st.set_page_config(page_title="AI Job Search Copilot", page_icon="AI", layout="wide")


def extract_recommendation(analysis: str) -> str:
    match = re.search(r"Recommendation:\s*(Apply|Maybe|Skip)", analysis, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).capitalize()


def populate_text_from_upload(uploaded_file, session_key: str, label: str) -> None:
    if not uploaded_file:
        return

    upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    signature_key = f"{session_key}_upload_signature"
    if st.session_state.get(signature_key) == upload_signature:
        return

    try:
        extracted_text = extract_text_from_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"Could not parse {label}: {exc}")
        return

    if not extracted_text:
        st.warning(f"No text could be extracted from the uploaded {label}.")
        return

    st.session_state[session_key] = extracted_text
    st.session_state[signature_key] = upload_signature
    st.success(f"Loaded text from {uploaded_file.name}.")


def initialize_session_state() -> None:
    defaults = {
        "user_profile": "",
        "job_description": "",
        "analysis": "",
        "resume_tailoring": "",
        "networking_messages": "",
        "career_profile_resume_text": "",
        "score_breakdown": {},
        "computed_fit_score": 0,
        "priority": "",
        "analysis_company": "",
        "analysis_job_title": "",
        "analysis_location": "",
        "analysis_job_url": "",
        "analysis_user_id": 1,
        "analysis_company_id": 0,
        "analysis_job_id": 0,
        "analysis_model_run_id": 0,
        "last_discovery_metrics": {},
        "last_discovery_settings": {},
        "selected_prep_job_id": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def text_to_rows(text: str, row_key: str) -> list[dict]:
    rows = []
    for line in (text or "").splitlines():
        value = line.strip()
        if value:
            rows.append({row_key: value})
    return rows


def dataframe_editor_rows(dataframe: pd.DataFrame) -> list[dict]:
    cleaned_df = dataframe.fillna("")
    if cleaned_df.empty:
        return []
    rows = []
    for row in cleaned_df.to_dict("records"):
        if any(str(value).strip() for value in row.values()):
            rows.append({key: str(value).strip() for key, value in row.items()})
    return rows


def compact_text(value: str, fallback: str = "Not set") -> str:
    cleaned = str(value or "").strip()
    return cleaned if cleaned else fallback


def format_list_block(value: str, fallback: str = "Not set") -> str:
    items = [item.strip() for item in re.split(r"[,;\n]", str(value or "")) if item.strip()]
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items[:8])


def format_constraints_summary(constraints: list[dict]) -> str:
    rows = []
    for item in constraints or []:
        label = str(item.get("constraint_type", "") or "").strip()
        value = str(item.get("constraint_value", "") or "").strip()
        severity = str(item.get("severity", "") or "").strip()
        if label or value:
            line = f"{label}: {value}".strip(": ")
            if severity:
                line = f"{line} ({severity})"
            rows.append(line)
    if not rows:
        return "Not set"
    return "\n".join(f"- {row}" for row in rows[:5])


def top_skill_names(profile: dict, limit: int = 8) -> str:
    rows = profile.get("skills_inventory", []) or []
    skill_names = [str(row.get("skill_name", "") or "").strip() for row in rows]
    skill_names = [name for name in skill_names if name]
    if skill_names:
        return ", ".join(skill_names[:limit])
    return compact_text(profile.get("skills", ""), "Not set")


def render_labeled_markdown_grid(items: list[tuple[str, str]], columns: int = 2) -> None:
    for index in range(0, len(items), columns):
        row = st.columns(columns)
        for col, (label, value) in zip(row, items[index : index + columns]):
            with col:
                st.markdown(f"**{label}**")
                st.markdown(value)


def reason_to_bullets(reason: str) -> str:
    parts = [part.strip() for part in str(reason or "").split(";") if part.strip()]
    if not parts:
        return "No reason recorded."
    return "\n".join(f"- {part}" for part in parts)


def format_decision_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    formatted = dataframe.copy()
    if "decision_reason" in formatted.columns:
        formatted["decision_reason"] = formatted["decision_reason"].apply(reason_to_bullets)
    if "rejection_reason" in formatted.columns:
        formatted["rejection_reason"] = formatted["rejection_reason"].apply(
            lambda value: compact_text(value, "")
        )
    return formatted


def format_asset_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    formatted = dataframe.copy()
    if "content" in formatted.columns:
        formatted["content"] = formatted["content"].astype(str).str.strip()
    return formatted


def current_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def profile_display_name(profile: dict) -> str:
    name = str(profile.get("name", "") or "").strip()
    if name:
        return name.split()[0]
    return "there"


def profile_strength_score(profile: dict) -> int:
    checks = [
        bool(str(profile.get("headline", "") or profile.get("summary", "")).strip()),
        bool(str(profile.get("target_roles", "")).strip()),
        bool(str(profile.get("preferred_locations", "")).strip()),
        bool(str(profile.get("visa_status", "")).strip()),
        bool(str(profile.get("career_goal", "")).strip()),
        bool(str(profile.get("skills", "")).strip() or (profile.get("skills_inventory", []) or [])),
        bool(profile.get("projects", []) or []),
        bool(profile.get("constraints", []) or []),
    ]
    return round(sum(checks) / len(checks) * 100)


def fetch_queue_records(include_all: bool = False) -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        query = """
            SELECT
                id,
                company,
                COALESCE(job_title, title, '') AS job_title,
                location,
                job_url,
                jd_text,
                short_description,
                source,
                COALESCE(post_time, 'Unknown') AS post_time,
                COALESCE(job_level, 'Unknown') AS job_level,
                COALESCE(work_mode, 'Unknown') AS work_mode,
                COALESCE(NULLIF(queue_category, ''), apply_decision, '') AS queue_category,
                COALESCE(rejection_reason, '') AS rejection_reason,
                pre_filter_score,
                fit_score,
                COALESCE(apply_decision, '') AS apply_decision,
                COALESCE(decision_reason, reason, '') AS decision_reason,
                status,
                resume_bullets,
                cover_letter,
                recruiter_message,
                application_checklist,
                created_at,
                updated_at
            FROM job_queue
        """
        if not include_all:
            query += """
            WHERE COALESCE(NULLIF(queue_category, ''), apply_decision, '') IN ('Apply', 'Maybe')
            """
        query += """
            ORDER BY
                CASE COALESCE(NULLIF(queue_category, ''), apply_decision, '')
                    WHEN 'Apply' THEN 1
                    WHEN 'Maybe' THEN 2
                    WHEN 'Filtered' THEN 3
                    WHEN 'Rejected' THEN 4
                    ELSE 5
                END,
                fit_score DESC,
                pre_filter_score DESC,
                created_at DESC
        """
        dataframe = pd.read_sql_query(query, conn)

    if dataframe.empty:
        return dataframe

    defaults = {
        "post_time": "Unknown",
        "job_level": "Unknown",
        "work_mode": "Unknown",
        "queue_category": "",
        "rejection_reason": "",
        "decision_reason": "",
        "status": "",
        "fit_score": 0,
        "pre_filter_score": 0,
    }
    for column_name, default_value in defaults.items():
        if column_name not in dataframe.columns:
            dataframe[column_name] = default_value
        dataframe[column_name] = dataframe[column_name].fillna(default_value)

    dataframe["queue_category"] = dataframe["queue_category"].where(
        dataframe["queue_category"].astype(str).str.strip() != "",
        dataframe["apply_decision"],
    )
    return dataframe


def archive_queue_job(queue_id: int) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE job_queue
            SET
                apply_decision = 'Skip',
                queue_category = 'Filtered',
                status = 'Skipped',
                decision_reason = CASE
                    WHEN COALESCE(decision_reason, '') = '' THEN 'Skipped by user'
                    ELSE decision_reason || '; Skipped by user'
                END,
                rejection_reason = 'Skipped by user'
            WHERE id = ?
            """,
            (queue_id,),
        )
        conn.commit()


def render_profile_summary_card(profile: dict) -> None:
    st.subheader("Profile Summary")
    if not any(profile.values()):
        st.info("Create a profile to personalize job matching.")
        return

    render_labeled_markdown_grid(
        [
            ("Headline", compact_text(profile.get("headline") or profile.get("summary"))),
            ("Target Roles", format_list_block(profile.get("target_roles", ""))),
            ("Top Skills", compact_text(top_skill_names(profile))),
            (
                "Preferred Locations",
                format_list_block(
                    profile.get("preferred_locations", "")
                    or profile.get("suggested_locations", "")
                ),
            ),
            ("Constraints", format_constraints_summary(profile.get("constraints", []))),
        ]
    )


def render_compact_profile_strip(profile: dict) -> None:
    if not any(profile.values()):
        st.warning("Create a Career Profile first for better job matching.")
        return
    render_labeled_markdown_grid(
        [
            ("Profile Headline", compact_text(profile.get("headline") or profile.get("summary"))),
            ("Target Roles", format_list_block(profile.get("target_roles", ""))),
            ("Excluded Roles", format_list_block(profile.get("excluded_roles", ""))),
            ("Preferred Locations", format_list_block(profile.get("preferred_locations", ""))),
            ("Top Skills", compact_text(top_skill_names(profile))),
        ]
    )


def recommend_next_action(profile: dict, queue: pd.DataFrame, applications: pd.DataFrame) -> str:
    if queue.empty:
        if not any(profile.values()):
            return "Create your career profile so the app can start finding better matches."
        return "Run job discovery to surface fresh matches and build your opportunities list."

    apply_jobs = queue[queue["queue_category"] == "Apply"]
    if not apply_jobs.empty:
        top_job = apply_jobs.sort_values(["fit_score", "pre_filter_score"], ascending=False).iloc[0]
        if not str(top_job.get("resume_bullets", "") or "").strip():
            return f"Generate an application pack for {top_job['job_title']} at {top_job['company']}."
        return f"Apply to {top_job['job_title']} at {top_job['company']} while the match is still warm."

    if not applications.empty:
        pending = applications[
            applications["status"].isin(["Saved", "Applied", "Networking"])
        ]
        if not pending.empty:
            row = pending.iloc[0]
            return f"Follow up on {row['job_title']} at {row['company']} and keep momentum going."

    return "Review your Maybe jobs and decide which one deserves a tailored application next."


def render_home_page() -> None:
    profile = get_career_profile()
    queue = fetch_queue_records()
    applications = get_applications()

    st.header(f"{current_greeting()}, {profile_display_name(profile)}.")
    st.caption("Your AI job search workspace is tuned for better matches, clearer tradeoffs, and the next action that matters.")

    summary_col1, summary_col2, summary_col3 = st.columns([1.1, 1, 1])
    with summary_col1:
        render_profile_summary_card(profile)
    with summary_col2:
        st.subheader("Career Focus")
        st.metric("Profile Strength", f"{profile_strength_score(profile)}/100")
        st.markdown("**Target Roles**")
        st.markdown(format_list_block(profile.get("target_roles", "")))
        st.markdown("**Preferred Locations**")
        st.markdown(format_list_block(profile.get("preferred_locations", "")))
    with summary_col3:
        st.subheader("Snapshot")
        st.metric("Best Matches", int(len(queue)))
        st.metric("Applications", int(len(applications)))
        st.metric("Interviews", int(len(applications[applications["status"] == "Interview"])))

    st.subheader("Today's Best Opportunities")
    if queue.empty:
        st.info("No strong opportunities yet. Start with your profile, then run discovery.")
    else:
        top_opportunities = (
            queue.sort_values(["fit_score", "pre_filter_score"], ascending=False)
            .head(3)[["company", "job_title", "location", "fit_score", "decision_reason"]]
            .rename(
                columns={
                    "job_title": "Title",
                    "fit_score": "Match Score",
                    "decision_reason": "Why It Matches",
                    "company": "Company",
                    "location": "Location",
                }
            )
        )
        st.dataframe(format_decision_table(top_opportunities), width="stretch", hide_index=True)

    st.subheader("Recommended Next Action")
    st.info(recommend_next_action(profile, queue, applications))


def render_setup_page() -> None:
    st.header("Career Profile Setup")
    profile = get_career_profile()

    with st.form("career_profile_form"):
        name = st.text_input("Name", value=profile["name"])
        target_roles = st.text_input("Target roles", value=profile["target_roles"])
        visa_status = st.text_input("Visa status", value=profile["visa_status"])
        preferred_locations = st.text_input(
            "Preferred locations", value=profile["preferred_locations"]
        )
        salary_goal = st.text_input("Salary goal", value=profile["salary_goal"])
        years_experience = st.text_input(
            "Years experience", value=profile["years_experience"]
        )
        career_goal = st.text_area(
            "Career goal", value=profile["career_goal"], height=140
        )

        if st.form_submit_button("Save Career Profile"):
            save_career_profile(
                {
                    "name": name,
                    "target_roles": target_roles,
                    "visa_status": visa_status,
                    "preferred_locations": preferred_locations,
                    "salary_goal": salary_goal,
                    "years_experience": years_experience,
                    "career_goal": career_goal,
                }
            )
            st.success("Career profile saved.")

    saved_profile = get_career_profile()
    if any(saved_profile.values()):
        st.subheader("Saved Profile")
        st.json(saved_profile)


def render_career_profile_summary(profile: dict) -> None:
    st.subheader("Profile Summary")
    if not any(profile.values()):
        st.info("No career profile saved yet.")
        return

    render_labeled_markdown_grid(
        [
            ("Headline", compact_text(profile.get("headline") or profile.get("summary"))),
            ("Target Roles", format_list_block(profile.get("target_roles", ""))),
            ("Top Skills", compact_text(top_skill_names(profile))),
            (
                "Preferred Locations",
                format_list_block(
                    profile.get("preferred_locations", "")
                    or profile.get("suggested_locations", "")
                ),
            ),
            ("Constraints", format_constraints_summary(profile.get("constraints", []))),
            ("Visa Status", compact_text(profile.get("visa_status", ""))),
        ]
    )


def render_career_profile_page(show_header: bool = True) -> None:
    if show_header:
        st.header("Career Profile")
    else:
        st.subheader("Career Profile")
    st.caption("Generate and edit the structured profile that powers job discovery, fit scoring, and application prep.")

    st.subheader("Resume Input")
    profile_upload = st.file_uploader(
        "Upload resume",
        type=["pdf", "docx", "txt"],
        key="career_profile_upload",
    )
    populate_text_from_upload(profile_upload, "career_profile_resume_text", "career profile resume")

    st.text_area(
        "Paste resume text",
        key="career_profile_resume_text",
        height=260,
    )

    if st.button("Generate Structured Profile from Resume", type="primary"):
        resume_text = st.session_state.get("career_profile_resume_text", "").strip()
        if not resume_text:
            st.warning("Upload or paste your resume before generating a career profile.")
        elif not get_openai_api_key():
            st.warning("OPENAI_API_KEY is missing. Add it to your .env file before generating a profile.")
        else:
            try:
                with st.spinner("Generating career profile..."):
                    profile = generate_career_profile_from_resume(resume_text)
                    save_career_profile(profile)
                    st.session_state.user_profile = resume_text
                st.success("Career profile generated and saved.")
            except JSONDecodeError:
                st.error("Profile generation failed because the model did not return valid structured JSON. Please try again.")
            except Exception as exc:
                st.error(f"Profile generation failed: {exc}")

    saved_profile = get_career_profile()
    if not any(saved_profile.values()):
        st.warning("Create a Career Profile first for better job matching.")
    render_career_profile_summary(saved_profile)

    if any(saved_profile.values()):
        st.divider()
        with st.form("career_profile_core_form"):
            st.subheader("Core Profile")
            st.caption("Edit the high-signal profile fields used across fit scoring, discovery, and prep.")
            core_col1, core_col2 = st.columns(2)
            with core_col1:
                name = st.text_input("Name", value=saved_profile.get("name", ""))
                headline = st.text_input("Headline", value=saved_profile.get("headline", ""))
                visa_status = st.text_input("Visa status", value=saved_profile.get("visa_status", ""))
                years_experience = st.text_input(
                    "Years experience",
                    value=saved_profile.get("years_experience", ""),
                )
                preferred_locations = st.text_area(
                    "Preferred locations",
                    value=saved_profile.get("preferred_locations", ""),
                    height=90,
                )
            with core_col2:
                target_roles = st.text_area(
                    "Target roles",
                    value=saved_profile.get("target_roles", ""),
                    height=90,
                )
                acceptable_roles = st.text_area(
                    "Acceptable roles",
                    value=saved_profile.get("acceptable_roles", ""),
                    height=90,
                )
                excluded_roles = st.text_area(
                    "Excluded roles",
                    value=saved_profile.get("excluded_roles", ""),
                    height=90,
                )
                salary_goal = st.text_input("Salary goal", value=saved_profile.get("salary_goal", ""))

            summary = st.text_area(
                "Professional summary",
                value=saved_profile.get("summary", ""),
                height=120,
            )
            education = st.text_area(
                "Education",
                value=saved_profile.get("education", ""),
                height=100,
            )
            career_goal = st.text_area(
                "Career goal",
                value=saved_profile.get("career_goal", ""),
                height=100,
            )
            missing_skills = st.text_area(
                "Missing skills",
                value=saved_profile.get("missing_skills", ""),
                height=90,
            )
            if st.form_submit_button("Save Core Profile"):
                updated_profile = {
                    **saved_profile,
                    "name": name,
                    "headline": headline,
                    "summary": summary,
                    "education": education,
                    "target_roles": target_roles,
                    "acceptable_roles": acceptable_roles,
                    "preferred_locations": preferred_locations,
                    "excluded_roles": excluded_roles,
                    "visa_status": visa_status,
                    "salary_goal": salary_goal,
                    "years_experience": years_experience,
                    "career_goal": career_goal,
                    "missing_skills": missing_skills,
                }
                save_career_profile(updated_profile)
                st.success("Core profile saved.")
                st.rerun()

        st.divider()
        st.subheader("Skill Graph")
        st.caption("Keep the skill list concise and evidence-backed so matching stays sharp.")
        skills_df = pd.DataFrame(
            saved_profile.get("skills_inventory", []),
            columns=["category", "skill_name", "proficiency", "evidence"],
        )
        edited_skills = st.data_editor(
            skills_df,
            num_rows="dynamic",
            width="stretch",
            key="career_profile_skills_editor",
        )
        if st.button("Save Skills"):
            updated_profile = {**saved_profile, "skills_inventory": dataframe_editor_rows(edited_skills)}
            save_career_profile(updated_profile)
            st.success("Skills saved.")
            st.rerun()

        st.divider()
        st.subheader("Project Memory")
        st.caption("These projects are the strongest evidence layer for job-fit reasoning and resume tailoring.")
        projects_df = pd.DataFrame(
            saved_profile.get("projects", []),
            columns=[
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
        edited_projects = st.data_editor(
            projects_df,
            num_rows="dynamic",
            width="stretch",
            key="career_profile_projects_editor",
        )
        if st.button("Save Projects"):
            updated_profile = {**saved_profile, "projects": dataframe_editor_rows(edited_projects)}
            save_career_profile(updated_profile)
            st.success("Projects saved.")
            st.rerun()

        st.divider()
        pref_col1, pref_col2 = st.columns(2)
        with pref_col1:
            st.subheader("Preferences")
            preferences_df = pd.DataFrame(
                saved_profile.get("preferences", []),
                columns=["preference_type", "preference_value", "weight"],
            )
            edited_preferences = st.data_editor(
                preferences_df,
                num_rows="dynamic",
                width="stretch",
                key="career_profile_preferences_editor",
            )
            if st.button("Save Preferences"):
                updated_profile = {
                    **saved_profile,
                    "preferences": dataframe_editor_rows(edited_preferences),
                }
                save_career_profile(updated_profile)
                st.success("Preferences saved.")
                st.rerun()

        with pref_col2:
            st.subheader("Constraints")
            constraints_df = pd.DataFrame(
                saved_profile.get("constraints", []),
                columns=["constraint_type", "constraint_value", "severity"],
            )
            edited_constraints = st.data_editor(
                constraints_df,
                num_rows="dynamic",
                width="stretch",
                key="career_profile_constraints_editor",
            )
            if st.button("Save Constraints"):
                updated_profile = {
                    **saved_profile,
                    "constraints": dataframe_editor_rows(edited_constraints),
                }
                save_career_profile(updated_profile)
                st.success("Constraints saved.")
                st.rerun()

        st.divider()
        st.subheader("Feedback History")
        feedback_history = get_career_feedback_history()
        if feedback_history.empty:
            st.info("No career feedback history yet.")
        else:
            st.dataframe(feedback_history, width="stretch", hide_index=True)


def render_resume_assets_page(show_header: bool = True) -> None:
    if show_header:
        st.header("Resume Assets")
    else:
        st.subheader("Resume Assets")
    st.caption("Build a reusable bullet library so application prep can retrieve evidence first and only rewrite what matters.")

    profile = get_career_profile()
    assets_df = fetch_resume_assets()

    top_profile_skills = top_skill_names(profile)
    render_labeled_markdown_grid(
        [
            ("Profile Headline", compact_text(profile.get("headline") or profile.get("summary"))),
            ("Target Roles", format_list_block(profile.get("target_roles", ""))),
            ("Projects Available", str(len(profile.get("projects", []) or []))),
            ("Stored Assets", str(len(assets_df))),
            ("Top Skills", compact_text(top_profile_skills)),
        ]
    )

    st.subheader("Generate From Career Profile Projects")
    if not (profile.get("projects", []) or []):
        st.warning("Add projects in Career Profile before generating resume assets.")
    else:
        st.caption("For each saved project, the app will generate 3 technical bullets, 2 business impact bullets, and 1 leadership/collaboration bullet.")
        if st.button("Generate Resume Assets From Projects", type="primary"):
            if not get_openai_api_key():
                st.warning("OPENAI_API_KEY is missing. Add it to your .env file before generating assets.")
            else:
                try:
                    with st.spinner("Generating reusable resume assets..."):
                        generated_assets = generate_assets_from_career_profile(profile)
                        existing_assets = assets_df.to_dict("records")
                        save_resume_assets(existing_assets + generated_assets)
                    st.success(f"Generated {len(generated_assets)} resume assets.")
                    st.rerun()
                except JSONDecodeError:
                    st.error("Resume asset generation failed because the model did not return valid structured JSON.")
                except Exception as exc:
                    st.error(f"Could not generate resume assets: {exc}")

    st.divider()
    st.subheader("Asset Library")
    st.caption("Add, edit, or delete assets here. Removing a row and saving deletes it from the local library.")
    asset_columns = [
        "id",
        "asset_type",
        "title",
        "content",
        "skills",
        "projects",
        "target_roles",
        "evidence_strength",
        "created_at",
        "updated_at",
    ]
    editable_assets = st.data_editor(
        assets_df.reindex(columns=asset_columns),
        num_rows="dynamic",
        width="stretch",
        key="resume_assets_editor",
        column_config={
            "asset_type": st.column_config.SelectboxColumn(
                "asset_type",
                options=ASSET_TYPES,
            ),
            "content": st.column_config.TextColumn("content", width="large"),
        },
        disabled=["created_at", "updated_at"],
    )
    if st.button("Save Resume Assets"):
        rows = dataframe_editor_rows(editable_assets)
        save_resume_assets(rows)
        st.success("Resume assets saved.")
        st.rerun()


def render_score_breakdown() -> None:
    scores = st.session_state.score_breakdown
    if not scores:
        return

    st.subheader("Structured Fit Score")
    st.metric("Computed Fit Score", f"{st.session_state.computed_fit_score}/100")
    score_df = score_breakdown_dataframe(scores)
    st.bar_chart(score_df.set_index("Component")["Score"])
    st.dataframe(score_df, width="stretch", hide_index=True)


def render_analysis_page(show_header: bool = True) -> None:
    profile = get_career_profile()
    career_profile_text = career_profile_to_text(profile)

    if not get_openai_api_key():
        st.warning("OPENAI_API_KEY is missing. Add it to your .env file before running AI analysis.")

    if show_header:
        st.header("Analyze A Job")
    st.subheader("User Profile / Resume")
    resume_upload = st.file_uploader(
        "Upload your profile or resume",
        type=["pdf", "docx", "txt"],
        key="resume_upload",
    )
    populate_text_from_upload(resume_upload, "user_profile", "resume/profile file")

    st.text_area(
        "Paste your profile or resume",
        key="user_profile",
        height=220,
    )

    st.subheader("Job Description")
    st.caption("Paste one specific opportunity when you want a full deep-dive, tailored bullets, and networking help.")
    st.markdown("**Job Metadata**")
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.text_input("Company", key="analysis_company")
        st.text_input("Job title", key="analysis_job_title")
    with meta_col2:
        st.text_input("Location", key="analysis_location")
        st.text_input("Job URL", key="analysis_job_url")

    jd_upload = st.file_uploader(
        "Upload the job description",
        type=["pdf", "docx", "txt"],
        key="jd_upload",
    )
    populate_text_from_upload(jd_upload, "job_description", "job description file")

    st.text_area(
        "Paste the job description",
        key="job_description",
        height=260,
    )

    if st.button("Analyze Job", type="primary"):
        if not st.session_state.user_profile.strip():
            st.warning("Please paste or upload your profile/resume before analyzing the job.")
        elif not st.session_state.job_description.strip():
            st.warning("Please paste or upload the job description before analyzing the job.")
        elif not get_openai_api_key():
            st.warning("OPENAI_API_KEY is missing. Add it to your .env file before running AI analysis.")
        else:
            try:
                selected_assets = retrieve_relevant_resume_assets(
                    st.session_state.job_description,
                    profile,
                    limit=5,
                )
                selected_assets_text = resume_assets_to_text(selected_assets)
                with st.spinner("Analyzing job fit..."):
                    st.session_state.analysis = analyze_job(
                        st.session_state.user_profile,
                        st.session_state.job_description,
                        career_profile_text,
                        selected_assets_text,
                    )
                    st.session_state.score_breakdown = parse_score_breakdown(
                        st.session_state.analysis
                    )
                    st.session_state.computed_fit_score = st.session_state.score_breakdown[
                        "fit_score"
                    ]
                    st.session_state.priority = determine_priority(
                        st.session_state.computed_fit_score,
                        extract_recommendation(st.session_state.analysis),
                    )
                with st.spinner("Generating resume tailoring..."):
                    st.session_state.resume_tailoring = tailor_resume(
                        st.session_state.user_profile,
                        st.session_state.job_description,
                        career_profile_text,
                        selected_assets_text,
                        "asset_first" if selected_assets else "full_llm",
                    )
                with st.spinner("Generating networking messages..."):
                    st.session_state.networking_messages = generate_networking_messages(
                        st.session_state.user_profile,
                        st.session_state.job_description,
                        career_profile_text,
                    )
                data_links = record_analysis_run(
                    st.session_state.analysis_company,
                    st.session_state.analysis_job_title,
                    st.session_state.analysis_location,
                    st.session_state.analysis_job_url,
                    st.session_state.job_description,
                    st.session_state.computed_fit_score,
                    st.session_state.score_breakdown,
                    st.session_state.analysis,
                    st.session_state.resume_tailoring,
                    st.session_state.networking_messages,
                    compute_career_profile_hash(profile),
                )
                st.session_state.analysis_user_id = data_links["user_id"]
                st.session_state.analysis_company_id = data_links["company_id"]
                st.session_state.analysis_job_id = data_links["job_id"]
                st.session_state.analysis_model_run_id = data_links["model_run_id"]
            except Exception as exc:
                st.error(f"LLM call failed: {exc}")

    if st.session_state.analysis:
        st.subheader("Job Fit Analysis")
        render_score_breakdown()
        if st.session_state.priority:
            st.info(f"Recommended priority: {st.session_state.priority}")
        st.markdown(st.session_state.analysis)

    if st.session_state.resume_tailoring:
        st.subheader("Resume Tailoring")
        st.markdown(st.session_state.resume_tailoring)
        st.download_button(
            "Export Resume Suggestions (.txt)",
            data=st.session_state.resume_tailoring,
            file_name="resume_suggestions.txt",
            mime="text/plain",
        )

    if st.session_state.networking_messages:
        st.subheader("Networking Messages")
        st.markdown(st.session_state.networking_messages)
        st.download_button(
            "Export Networking Messages (.txt)",
            data=st.session_state.networking_messages,
            file_name="networking_messages.txt",
            mime="text/plain",
        )

    render_feedback_form()
    render_save_application_form()


def render_feedback_form() -> None:
    if not st.session_state.analysis_model_run_id:
        return

    st.header("Analysis Feedback")
    with st.form("analysis_feedback_form"):
        usefulness_rating = st.slider("Was this analysis useful?", 1, 5, 4)
        used_resume_bullets = st.radio(
            "Did you use the generated resume bullets?",
            ["yes", "no"],
            horizontal=True,
        )
        used_networking_message = st.radio(
            "Did you use the networking message?",
            ["yes", "no"],
            horizontal=True,
        )

        if st.form_submit_button("Save Feedback"):
            save_user_feedback(
                st.session_state.analysis_model_run_id,
                usefulness_rating,
                used_resume_bullets,
                used_networking_message,
            )
            st.success("Feedback saved.")


def ensure_application_links(company: str, job_title: str, location: str, job_url: str) -> dict:
    if st.session_state.analysis_company_id and st.session_state.analysis_job_id:
        return {
            "user_id": st.session_state.analysis_user_id,
            "company_id": st.session_state.analysis_company_id,
            "job_id": st.session_state.analysis_job_id,
        }

    company_id = get_or_create_company(company)
    job_id = create_job(
        company_id,
        job_title,
        location,
        job_url,
        st.session_state.job_description,
    )
    return {"user_id": 1, "company_id": company_id, "job_id": job_id}


def render_save_application_form() -> None:
    st.header("Save Application")
    with st.form("save_application_form"):
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input(
                "Company",
                value=st.session_state.analysis_company,
                key="save_company",
            )
            job_title = st.text_input(
                "Job title",
                value=st.session_state.analysis_job_title,
                key="save_job_title",
            )
            location = st.text_input(
                "Location",
                value=st.session_state.analysis_location,
                key="save_location",
            )
            job_url = st.text_input(
                "Job URL",
                value=st.session_state.analysis_job_url,
                key="save_job_url",
            )
            recruiter_name = st.text_input("Recruiter Name", key="save_recruiter_name")
            application_date = st.text_input(
                "Application Date",
                placeholder="YYYY-MM-DD",
                key="save_application_date",
            )
        with col2:
            inferred_fit_score = st.session_state.computed_fit_score
            inferred_recommendation = extract_recommendation(st.session_state.analysis)
            fit_score = st.number_input(
                "Fit score",
                min_value=0,
                max_value=100,
                value=inferred_fit_score,
                step=1,
            )
            recommendation = st.selectbox(
                "Recommendation",
                ["", "Apply", "Maybe", "Skip"],
                index=["", "Apply", "Maybe", "Skip"].index(inferred_recommendation)
                if inferred_recommendation in ["Apply", "Maybe", "Skip"]
                else 0,
            )
            priority = st.selectbox(
                "Pipeline Priority",
                ["", "Must Apply", "Good Opportunity", "Low Priority"],
                index=["", "Must Apply", "Good Opportunity", "Low Priority"].index(
                    st.session_state.priority
                )
                if st.session_state.priority
                in ["Must Apply", "Good Opportunity", "Low Priority"]
                else 0,
            )
            status = st.selectbox("Status", STATUS_OPTIONS)
            outcome_status = st.selectbox("Outcome Status", OUTCOME_STATUS_OPTIONS)
            follow_up_date = st.text_input(
                "Follow-up Date",
                placeholder="YYYY-MM-DD",
                key="save_follow_up_date",
            )
            interview_stage = st.text_input("Interview Stage", key="save_interview_stage")
        next_action = st.text_input("Next Action", key="save_next_action")
        notes = st.text_area("Notes", height=120, key="save_notes")

        submitted = st.form_submit_button("Save Application")
        if submitted:
            links = ensure_application_links(company, job_title, location, job_url)
            application_id = save_application(
                company=company,
                job_title=job_title,
                location=location,
                job_url=job_url,
                fit_score=int(fit_score),
                recommendation=recommendation,
                status=status,
                notes=notes,
                score_breakdown=st.session_state.score_breakdown,
                priority=priority,
                application_date=application_date,
                follow_up_date=follow_up_date,
                recruiter_name=recruiter_name,
                next_action=next_action,
                interview_stage=interview_stage,
                user_id=links["user_id"],
                company_id=links["company_id"],
                job_id=links["job_id"],
                outcome_status=outcome_status,
            )
            create_outcome(application_id, outcome_status)
            st.success("Application saved.")


def render_pipeline_page() -> None:
    st.header("Job Pipeline")
    applications = get_applications()

    if applications.empty:
        st.info("No saved applications yet.")
        return

    priority_order = ["Must Apply", "Good Opportunity", "Low Priority"]
    for priority in priority_order:
        st.subheader(priority)
        priority_jobs = applications[applications["priority"] == priority]
        if priority_jobs.empty:
            st.caption("No jobs in this bucket.")
        else:
            st.dataframe(priority_jobs, width="stretch", hide_index=True)

    uncategorized = applications[~applications["priority"].isin(priority_order)]
    if not uncategorized.empty:
        st.subheader("Uncategorized")
        st.dataframe(uncategorized, width="stretch", hide_index=True)


def render_tracker_page(show_header: bool = True) -> None:
    if show_header:
        st.header("Application Tracker")
    applications = get_applications()

    total_applications = len(applications)
    interviews = len(applications[applications["status"] == "Interview"])
    rejections = len(applications[applications["status"] == "Rejected"])
    offers = len(applications[applications["status"] == "Offer"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", total_applications)
    col2.metric("Interviews", interviews)
    col3.metric("Rejections", rejections)
    col4.metric("Offers", offers)

    st.dataframe(applications, width="stretch", hide_index=True)

    if applications.empty:
        return

    st.subheader("Update Outcome")
    with st.form("outcome_update_form"):
        application_options = {
            f"{row.company} - {row.job_title} (#{row.id})": int(row.id)
            for row in applications.itertuples()
        }
        selected_application = st.selectbox(
            "Application",
            list(application_options.keys()),
        )
        outcome_status = st.selectbox("Outcome Status", OUTCOME_STATUS_OPTIONS)
        if st.form_submit_button("Update Outcome"):
            update_outcome(application_options[selected_application], outcome_status)
            st.success("Outcome updated.")


def render_analytics_page(show_header: bool = True) -> None:
    if show_header:
        st.header("Analytics")
    summary = fetch_analytics_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyzed Jobs", summary["total_analyzed_jobs"])
    col2.metric("Total Applications", summary["total_applications"])
    col3.metric("Interview Rate", f"{summary['interview_rate']:.1%}")
    col4.metric("Offer Rate", f"{summary['offer_rate']:.1%}")

    st.subheader("Average Fit Score By Outcome")
    outcome_scores = fetch_average_fit_score_by_outcome()
    st.dataframe(outcome_scores, width="stretch", hide_index=True)
    if not outcome_scores.empty:
        st.bar_chart(outcome_scores.set_index("outcome")["average_fit_score"])

    st.subheader("Most Common Missing Skills")
    missing_skills = fetch_most_common_missing_skills()
    st.dataframe(missing_skills, width="stretch", hide_index=True)
    if not missing_skills.empty:
        st.bar_chart(missing_skills.set_index("missing_skill")["count"])

    st.subheader("Top Scoring Companies")
    top_companies = fetch_top_scoring_companies()
    st.dataframe(top_companies, width="stretch", hide_index=True)
    if not top_companies.empty:
        st.bar_chart(top_companies.set_index("company")["average_fit_score"])


def render_job_discovery_page(show_header: bool = True, show_debug: bool = True) -> None:
    if show_header:
        st.header("Discover Jobs")
    st.caption(
        "Discover jobs from public pages, CSV, manual paste, or sample data. "
        "This page does not scrape LinkedIn, require login, or submit applications."
    )
    st.info(
        "Public URL discovery works best with public Greenhouse, Lever, Ashby, or company career pages. "
        "It does not support login-only pages, does not scrape LinkedIn, and only reads static HTML in Sprint 5."
    )

    profile = get_career_profile()
    profile_text = compact_career_profile_text(profile)
    st.subheader("Using Your Career Profile")
    if profile_text:
        render_compact_profile_strip(profile)
    else:
        st.warning("Create a Career Profile first for better job matching.")

    with st.form("job_discovery_form"):
        st.subheader("Job Search Settings")
        settings_col1, settings_col2 = st.columns(2)
        with settings_col1:
            target_roles = st.text_input(
                "Target roles",
                value=profile.get("target_roles", ""),
                placeholder="AI Engineer, GenAI Engineer, AI Application Engineer",
            )
            keywords = st.text_input(
                "Keywords",
                value="AI, GenAI, LLM, RAG, Python, FastAPI, SaaS",
            )
            pre_filter_threshold = st.number_input(
                "Pre-filter threshold",
                min_value=0,
                max_value=100,
                value=DEFAULT_PRE_FILTER_THRESHOLD,
                step=5,
            )
        with settings_col2:
            target_locations = st.text_input(
                "Target locations",
                value=profile.get("preferred_locations", ""),
                placeholder="Remote, San Francisco, New York",
            )
            excluded_keywords = st.text_input(
                "Excluded keywords",
                value="Staff, Principal, Director, VP, commission only, unpaid",
            )
            max_jobs_per_run = st.number_input(
                "Max jobs per run",
                min_value=1,
                max_value=200,
                value=DEFAULT_MAX_JOBS_PER_RUN,
                step=5,
            )
            allow_senior_roles = st.checkbox("Allow senior roles")
            force_refresh = st.checkbox("Force refresh known jobs")
        max_llm_calls_per_run = st.number_input(
            "Max LLM calls per run",
            min_value=0,
            max_value=50,
            value=DEFAULT_MAX_LLM_CALLS_PER_RUN,
            step=1,
        )

        st.subheader("Sources")
        career_page_url = st.text_input(
            "Public career page URL",
            placeholder="https://company.com/careers",
        )
        job_board_url = st.text_input(
            "Public job board/search result URL",
            placeholder="Greenhouse, Lever, Ashby, company site, etc. No LinkedIn scraping.",
        )
        job_urls = st.text_area(
            "Paste job URLs manually",
            height=120,
            placeholder="One URL per line. URL-only jobs are queued without scraping.",
        )
        raw_jds = st.text_area(
            "Paste multiple jobs manually",
            height=220,
            placeholder="Company:\nTitle:\nLocation:\nURL:\nDescription:\n---\nCompany:\nTitle:\nLocation:\nURL:\nDescription:",
        )
        csv_upload = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Required columns: company, job_title, location, job_url, jd_text",
        )
        use_sample_data = st.checkbox("Use sample dataset mode")
        submitted = st.form_submit_button("Run Discovery")

    if submitted:
        jobs = []
        source_names = []
        source_types = []
        discovery_profile = {
            **profile,
            "target_roles": target_roles or profile.get("target_roles", ""),
            "preferred_locations": target_locations or profile.get("preferred_locations", ""),
        }
        settings = {
            "target_roles": target_roles,
            "target_locations": target_locations,
            "keywords": keywords,
            "excluded_keywords": excluded_keywords,
            "pre_filter_threshold": int(pre_filter_threshold),
            "max_jobs_per_run": int(max_jobs_per_run),
            "max_llm_calls_per_run": int(max_llm_calls_per_run),
            "allow_senior_roles": bool(allow_senior_roles),
            "force_refresh": bool(force_refresh),
            "user_profile": compact_career_profile_text(discovery_profile),
            "career_profile": discovery_profile,
            "career_profile_text": compact_career_profile_text(discovery_profile),
        }

        if use_sample_data:
            try:
                sample_jobs = load_sample_discovered_jobs()
                jobs.extend(sample_jobs)
                source_names.append("Sample Dataset")
                source_types.append("sample")
                upsert_discovered_source("Sample Dataset", "data/sample_discovered_jobs.csv", "sample")
            except Exception as exc:
                st.warning(f"Could not load sample data: {exc}")

        public_urls = [
            ("Career Page", career_page_url),
            ("Job Board", job_board_url),
        ]
        for label, public_url in public_urls:
            if not public_url.strip():
                continue
            try:
                public_jobs = discover_from_public_url(public_url, settings)
                jobs.extend(public_jobs)
                source_names.append(label)
                source_types.append("public_html")
                upsert_discovered_source(label, public_url, "public_html")
                if not public_jobs:
                    st.warning(f"No jobs were found at {public_url}.")
            except Exception as exc:
                st.warning(f"Could not fetch {label} URL: {exc}")

        jobs.extend(parse_job_urls(job_urls))
        if job_urls.strip():
            source_names.append("Manual URLs")
            source_types.append("manual")

        jobs.extend(parse_manual_jobs(raw_jds))
        if raw_jds.strip():
            source_names.append("Manual Pasted Jobs")
            source_types.append("manual")

        if csv_upload:
            try:
                csv_jobs = parse_csv_jobs(csv_upload)
                jobs.extend(csv_jobs)
                source_names.append(csv_upload.name)
                source_types.append("csv")
                upsert_discovered_source(csv_upload.name, csv_upload.name, "csv")
            except Exception as exc:
                st.warning(f"Could not read CSV: {exc}")

        jobs_with_descriptions = [job for job in jobs if job.get("jd_text", "").strip()]
        has_analysis_context = bool(compact_career_profile_text(discovery_profile).strip())
        if not has_analysis_context and use_sample_data:
            settings["user_profile"] = compact_career_profile_text(discovery_profile) or (
                "Demo profile: early-career AI application engineer targeting AI Engineer, "
                "GenAI Engineer, and AI Solutions Engineer roles with Python, LLM, RAG, "
                "FastAPI, automation, SaaS, and practical business-impact projects."
            )
            has_analysis_context = True

        if not jobs:
            st.warning("No jobs found. Add a public URL, CSV, manual jobs, job URLs, or enable sample dataset mode for a working demo.")
        elif jobs_with_descriptions and not has_analysis_context:
            st.warning("Generate or complete your Career Profile before scoring jobs with descriptions.")
        else:
            if jobs_with_descriptions and not get_openai_api_key() and int(max_llm_calls_per_run) > 0:
                st.warning("OPENAI_API_KEY is missing. Running rule-based filtering and pre-scoring only.")
                settings["max_llm_calls_per_run"] = 0

            with st.spinner("Running discovery filters and LLM-limited deep analysis..."):
                try:
                    metrics = run_discovery_pipeline(jobs, settings)
                except Exception as exc:
                    st.warning(f"Could not process discovered jobs: {exc}")
                    metrics = None

            if metrics:
                st.session_state.last_discovery_metrics = metrics
                st.session_state.last_discovery_settings = settings
                metric_defaults = {
                    "already_known_jobs": 0,
                    "stale_cache_hits": 0,
                    "new_jobs": len(metrics.get("queued_jobs", [])),
                    "jobs_rejected_by_rules": 0,
                    "jobs_below_threshold": 0,
                    "jobs_filtered_rejected": metrics.get("jobs_filtered_out", 0),
                    "jobs_added_to_queue": len(metrics.get("queued_jobs", [])),
                    "actual_llm_calls_used": metrics.get("jobs_sent_to_llm", 0),
                    "llm_calls_avoided": metrics.get("skipped_llm_calls", 0),
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "profile_changed_reanalysis_recommended": 0,
                    "estimated_tokens_saved": metrics.get("estimated_token_savings", 0),
                    "token_savings_formula": "llm_calls_avoided * 750 tokens",
                    "known_jobs": [],
                    "stale_jobs": [],
                    "rejected_by_rules_jobs": [],
                    "below_threshold_jobs": [],
                    "queued_jobs": [],
                    "processed_jobs": [],
                    "duplicate_jobs": [],
                    "pre_scored_jobs": [],
                    "jobs_sent_to_llm_rows": [],
                }
                for metric_key, default_value in metric_defaults.items():
                    metrics.setdefault(metric_key, default_value)

                record_discovery_run(
                    ", ".join(source_names) or "Manual Discovery",
                    ", ".join(sorted(set(source_types))) or "manual",
                    keywords,
                    target_locations,
                    metrics,
                )
                if metrics["jobs_added_to_queue"] > 0:
                    st.success(f"Found {metrics['jobs_added_to_queue']} promising job(s) worth your attention.")
                else:
                    st.warning("Discovery finished, but nothing strong enough made it into your active opportunities list yet.")

                if metrics["profile_changed_reanalysis_recommended"] > 0:
                    st.warning(
                        "Some older matches were scored before your latest profile update. "
                        "Turn on Force refresh known jobs when you want them re-analyzed."
                    )

                if show_debug:
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    metric_col1.metric("Jobs Discovered", metrics["jobs_discovered"])
                    metric_col2.metric("Already Known Jobs", metrics["already_known_jobs"])
                    metric_col3.metric("New Jobs", metrics["new_jobs"])
                    metric_col4.metric("Duplicates Removed", metrics["duplicates_removed"])

                    metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
                    metric_col5.metric("Rejected by Rules", metrics["jobs_rejected_by_rules"])
                    metric_col6.metric("Below Threshold", metrics["jobs_below_threshold"])
                    metric_col7.metric("Sent to LLM", metrics["jobs_sent_to_llm"])
                    metric_col8.metric("Added to Main Queue", metrics["jobs_added_to_queue"])

                    metric_col9, metric_col10, metric_col11 = st.columns(3)
                    metric_col9.metric("Pre-scored", metrics["jobs_pre_scored"])
                    metric_col10.metric("Cache Hits", metrics["cache_hits"])
                    metric_col11.metric("Cache Misses", metrics["cache_misses"])

                    metric_col12, metric_col13 = st.columns(2)
                    metric_col12.metric("Stale Cache Hits", metrics["stale_cache_hits"])
                    metric_col13.metric(
                        "Profile Changed / Re-analysis Recommended",
                        metrics["profile_changed_reanalysis_recommended"],
                    )

                    st.subheader("Token Efficiency")
                    token_col1, token_col2, token_col3, token_col4, token_col5 = st.columns(5)
                    token_col1.metric("Max Jobs", int(max_jobs_per_run))
                    token_col2.metric("Max LLM Calls", int(max_llm_calls_per_run))
                    token_col3.metric("Actual LLM Calls", metrics["actual_llm_calls_used"])
                    token_col4.metric("LLM Calls Avoided", metrics["llm_calls_avoided"])
                    token_col5.metric("Tokens Saved", metrics["estimated_tokens_saved"])
                    st.caption(f"Formula: {metrics['token_savings_formula']}")

                    if metrics["jobs_sent_to_llm"] >= settings["max_llm_calls_per_run"] and settings["max_llm_calls_per_run"] > 0:
                        st.warning("Max LLM calls per run was reached. Remaining qualified jobs stayed in the queue without deep analysis.")

                    with st.expander("Rejected by Rules", expanded=True):
                        if metrics["rejected_by_rules_jobs"]:
                            st.dataframe(
                                format_decision_table(pd.DataFrame(metrics["rejected_by_rules_jobs"])),
                                width="stretch",
                                hide_index=True,
                            )
                        else:
                            st.info("No jobs were rejected by deterministic rules.")

                    with st.expander("Below Threshold", expanded=True):
                        if metrics["below_threshold_jobs"]:
                            st.dataframe(
                                format_decision_table(pd.DataFrame(metrics["below_threshold_jobs"])),
                                width="stretch",
                                hide_index=True,
                            )
                        else:
                            st.info("No jobs fell below the pre-filter threshold.")

                    with st.expander("Sent to LLM"):
                        sent_to_llm = metrics["jobs_sent_to_llm_rows"]
                        if sent_to_llm:
                            st.dataframe(
                                format_decision_table(pd.DataFrame(sent_to_llm)),
                                width="stretch",
                                hide_index=True,
                            )
                        else:
                            st.info("No jobs were sent to the LLM in this run.")

                    with st.expander("Main Queue", expanded=True):
                        if metrics["queued_jobs"]:
                            st.dataframe(
                                format_decision_table(pd.DataFrame(metrics["queued_jobs"])),
                                width="stretch",
                                hide_index=True,
                            )
                        else:
                            st.info("No jobs qualified for the main queue in this run.")

                    with st.expander("Full Raw Discovery Log"):
                        st.dataframe(metrics["processed_jobs"], width="stretch", hide_index=True)

                    with st.expander("Already Known Jobs"):
                        if metrics["known_jobs"]:
                            st.dataframe(pd.DataFrame(metrics["known_jobs"]), width="stretch", hide_index=True)
                        else:
                            st.info("No historical jobs were reused from cache.")

                    with st.expander("Stale Cache Hits"):
                        if metrics["stale_jobs"]:
                            st.dataframe(pd.DataFrame(metrics["stale_jobs"]), width="stretch", hide_index=True)
                        else:
                            st.info("No stale cached jobs detected in this run.")

                    with st.expander("Duplicates Removed"):
                        if metrics["duplicate_jobs"]:
                            st.dataframe(pd.DataFrame(metrics["duplicate_jobs"]), width="stretch", hide_index=True)
                        else:
                            st.info("No duplicates removed in this run.")

                    with st.expander("Pre-Scored Jobs", expanded=True):
                        if metrics["pre_scored_jobs"]:
                            st.dataframe(
                                format_decision_table(pd.DataFrame(metrics["pre_scored_jobs"])),
                                width="stretch",
                                hide_index=True,
                            )
                        else:
                            st.info("No jobs reached pre-scoring in this run.")

                st.download_button(
                    "Export Discovered Jobs CSV",
                    data=pd.DataFrame(metrics["imported_jobs"]).to_csv(index=False),
                    file_name="discovered_jobs.csv",
                    mime="text/csv",
                )


def render_job_queue(show_header: bool = True) -> None:
    if show_header:
        st.header("Best Matches")
    maintenance_col1, maintenance_col2 = st.columns([1, 3])
    with maintenance_col1:
        if st.button("Clean Duplicate Queue Records"):
            results = deduplicate_job_queue()
            st.success(
                f"Archived {results['duplicates_archived']} duplicate queue record(s). "
                f"{results['active_records']} active record(s) remain."
            )
            st.rerun()
    queue = fetch_queue_records()
    full_queue = fetch_queue_records(include_all=True)

    if queue.empty:
        st.info("No discovered jobs yet.")
        return

    display_columns = [
        "company",
        "job_title",
        "location",
        "work_mode",
        "job_level",
        "pre_filter_score",
        "fit_score",
        "apply_decision",
        "decision_reason",
        "status",
        "job_url",
    ]
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        company_filter = st.text_input("Company filter", key="opps_company_filter")
    with filter_col2:
        location_filter = st.text_input("Location filter", key="opps_location_filter")
    with filter_col3:
        source_filter = st.text_input("Source filter", key="opps_source_filter")
    with filter_col4:
        minimum_fit_score = st.number_input("Minimum match score", 0, 100, 0, 5, key="opps_fit_min")

    filter_col5, filter_col6, filter_col7, filter_col8 = st.columns(4)
    with filter_col5:
        post_time_filter = st.text_input("Post time", key="opps_post_time_filter")
    with filter_col6:
        job_level_filter = st.selectbox(
            "Level",
            ["All", "Internship", "Entry", "Junior", "Mid", "Senior", "Staff", "Principal", "Director", "Unknown"],
            key="opps_job_level_filter",
        )
    with filter_col7:
        work_mode_filter = st.selectbox(
            "Work mode",
            ["All", "Remote", "Hybrid", "Onsite", "Unknown"],
            key="opps_work_mode_filter",
        )
    with filter_col8:
        minimum_pre_filter_score = st.number_input("Minimum pre-filter score", 0, 100, 0, 5, key="opps_prefilter_min")

    filtered_queue = queue.copy()
    if post_time_filter:
        filtered_queue = filtered_queue[
            filtered_queue["post_time"].str.contains(post_time_filter, case=False, na=False)
        ]
    if job_level_filter != "All":
        filtered_queue = filtered_queue[filtered_queue["job_level"] == job_level_filter]
    if work_mode_filter != "All":
        filtered_queue = filtered_queue[filtered_queue["work_mode"] == work_mode_filter]
    filtered_queue = filtered_queue[
        filtered_queue["pre_filter_score"] >= minimum_pre_filter_score
    ]
    filtered_queue = filtered_queue[filtered_queue["fit_score"] >= minimum_fit_score]
    if source_filter:
        filtered_queue = filtered_queue[
            filtered_queue["source"].str.contains(source_filter, case=False, na=False)
        ]
    if company_filter:
        filtered_queue = filtered_queue[
            filtered_queue["company"].str.contains(company_filter, case=False, na=False)
        ]
    if location_filter:
        filtered_queue = filtered_queue[
            filtered_queue["location"].str.contains(location_filter, case=False, na=False)
        ]

    best_tab, apply_tab, maybe_tab, archived_tab = st.tabs(
        ["Best Matches", "Apply", "Maybe", "Skip / Archived"]
    )

    with best_tab:
        best_matches = filtered_queue.sort_values(["fit_score", "pre_filter_score"], ascending=False)
        display_best = best_matches[display_columns].rename(
            columns={
                "job_title": "Title",
                "fit_score": "Match Score",
                "apply_decision": "Decision",
                "decision_reason": "Reason",
                "job_level": "Level",
                "job_url": "Job URL",
                "company": "Company",
                "location": "Location",
                "work_mode": "Work Mode",
                "status": "Status",
                "source": "Source",
                "pre_filter_score": "Pre-Score",
            }
        )
        st.dataframe(format_decision_table(display_best), width="stretch", hide_index=True)

    with apply_tab:
        apply_rows = filtered_queue[filtered_queue["queue_category"] == "Apply"]
        if apply_rows.empty:
            st.info("No Apply opportunities match your filters right now.")
        else:
            st.dataframe(
                format_decision_table(
                    apply_rows[display_columns].rename(
                        columns={
                            "job_title": "Title",
                            "fit_score": "Match Score",
                            "apply_decision": "Decision",
                            "decision_reason": "Reason",
                            "job_level": "Level",
                            "job_url": "Job URL",
                            "company": "Company",
                            "location": "Location",
                            "work_mode": "Work Mode",
                            "status": "Status",
                            "pre_filter_score": "Pre-Score",
                        }
                    )
                ),
                width="stretch",
                hide_index=True,
            )

    with maybe_tab:
        maybe_rows = filtered_queue[filtered_queue["queue_category"] == "Maybe"]
        if maybe_rows.empty:
            st.info("No Maybe opportunities match your filters right now.")
        else:
            st.dataframe(
                format_decision_table(
                    maybe_rows[display_columns].rename(
                        columns={
                            "job_title": "Title",
                            "fit_score": "Match Score",
                            "apply_decision": "Decision",
                            "decision_reason": "Reason",
                            "job_level": "Level",
                            "job_url": "Job URL",
                            "company": "Company",
                            "location": "Location",
                            "work_mode": "Work Mode",
                            "status": "Status",
                            "pre_filter_score": "Pre-Score",
                        }
                    )
                ),
                width="stretch",
                hide_index=True,
            )

    with archived_tab:
        archived_rows = full_queue[
            full_queue["queue_category"].isin(["Filtered", "Rejected"])
            | full_queue["status"].isin(["Skipped"])
            | full_queue["apply_decision"].isin(["Skip"])
        ]
        if archived_rows.empty:
            st.info("No skipped or archived opportunities yet.")
        else:
            st.dataframe(
                format_decision_table(
                    archived_rows[
                        [
                            "company",
                            "job_title",
                            "location",
                            "work_mode",
                            "job_level",
                            "fit_score",
                            "apply_decision",
                            "decision_reason",
                            "status",
                            "job_url",
                        ]
                    ].rename(
                        columns={
                            "job_title": "Title",
                            "fit_score": "Match Score",
                            "apply_decision": "Decision",
                            "decision_reason": "Reason",
                            "job_level": "Level",
                            "job_url": "Job URL",
                            "company": "Company",
                            "location": "Location",
                            "work_mode": "Work Mode",
                            "status": "Status",
                        }
                    )
                ),
                width="stretch",
                hide_index=True,
            )

    apply_queue = queue[queue["apply_decision"] == "Apply"]
    if not apply_queue.empty:
        st.download_button(
            "Export Apply Queue CSV",
            data=apply_queue[display_columns].to_csv(index=False),
            file_name="apply_queue.csv",
            mime="text/csv",
        )

    st.subheader("Take Next Action")
    options = {
        f"{row.company} - {row.job_title or 'Untitled Job'} (#{row.id})": int(row.id)
        for row in filtered_queue.itertuples()
    }
    if not options:
        st.info("No jobs match the current filters.")
        return

    action_col1, action_col2 = st.columns([2, 3])
    with action_col1:
        selected_job_label = st.selectbox("Opportunity", list(options.keys()))
        selected_job_id = options[selected_job_label]
        st.session_state.selected_prep_job_id = selected_job_id
    with action_col2:
        action_button_col1, action_button_col2, action_button_col3, action_button_col4 = st.columns(4)
        with action_button_col1:
            generate_prep = st.button("Generate Application Pack")
        with action_button_col2:
            mark_applied = st.button("Mark Applied")
        with action_button_col3:
            skip_job = st.button("Skip")
        with action_button_col4:
            save_for_later = st.button("Save for Later")

    if mark_applied:
        update_job_queue_status(selected_job_id, "Applied")
        st.success("Marked as applied.")
        st.rerun()
    if skip_job:
        archive_queue_job(selected_job_id)
        st.success("Moved to archived opportunities.")
        st.rerun()
    if save_for_later:
        update_job_queue_status(selected_job_id, "Reviewed")
        st.success("Saved for later review.")
        st.rerun()

    apply_jobs = apply_queue
    if apply_jobs.empty:
        return

    st.subheader("Application Prep")
    prep_options = {
        f"{row.company} - {row.job_title or 'Untitled Job'} (#{row.id})": row
        for row in apply_jobs.itertuples()
    }
    prep_labels = list(prep_options.keys())
    default_index = 0
    for index, label in enumerate(prep_labels):
        if prep_options[label].id == st.session_state.get("selected_prep_job_id"):
            default_index = index
            break
    selected_prep_label = st.selectbox("Ready-to-apply job", prep_labels, index=default_index)
    selected_prep = prep_options[selected_prep_label]
    prep_profile = get_career_profile()
    prep_assets = retrieve_relevant_resume_assets(
        selected_prep.jd_text or selected_prep.short_description or "",
        prep_profile,
        limit=5,
    )
    prep_mode = st.selectbox(
        "Prep mode",
        ["asset_first", "full_llm"],
        format_func=lambda value: "Use asset-first mode" if value == "asset_first" else "Full LLM generation mode",
    )

    st.subheader("Selected Resume Assets")
    if prep_assets:
        prep_assets_df = pd.DataFrame(prep_assets)[
            [
                "asset_type",
                "title",
                "content",
                "skills",
                "projects",
                "target_roles",
                "evidence_strength",
                "match_score",
            ]
        ]
        st.dataframe(format_asset_table(prep_assets_df), width="stretch", hide_index=True)
        st.caption("Application Prep sends only the top 5 matching assets in asset-first mode.")
    else:
        st.info("No matching resume assets were found yet. Add assets on the Resume Assets page or use Full LLM generation mode.")

    st.subheader("Tailored Resume Bullets")
    if not selected_prep.resume_bullets and selected_prep.jd_text:
        st.caption("This uses an LLM call.")
        if st.button("Generate Application Prep", key=f"generate_prep_{selected_prep.id}") or generate_prep:
            if not st.session_state.user_profile.strip():
                st.warning("Add your profile/resume on the Analyze Job page before generating prep.")
            elif not get_openai_api_key():
                st.warning("OPENAI_API_KEY is missing. Add it to your .env file before generating prep.")
            else:
                with st.spinner("Generating application prep..."):
                    prep = generate_application_prep(
                        st.session_state.user_profile or compact_career_profile_text(get_career_profile()),
                        selected_prep.jd_text,
                        prep_profile,
                        compact_career_profile_text(prep_profile, selected_prep.jd_text),
                        prep_assets if prep_mode == "asset_first" else [],
                        prep_mode,
                    )
                    update_application_prep(int(selected_prep.id), prep)
                st.success("Application prep generated. Refresh this page section to view it.")

    st.markdown(selected_prep.resume_bullets or "No resume bullets generated.")

    st.subheader("Cover Letter")
    st.text_area(
        "Cover letter",
        value=selected_prep.cover_letter or "",
        height=220,
        key=f"cover_letter_{selected_prep.id}",
    )

    st.subheader("Recruiter Message")
    st.text_area(
        "Recruiter message",
        value=selected_prep.recruiter_message or "",
        height=160,
        key=f"recruiter_message_{selected_prep.id}",
    )

    st.subheader("Application Checklist")
    st.text_area(
        "Checklist",
        value=selected_prep.application_checklist or "",
        height=160,
        key=f"application_checklist_{selected_prep.id}",
    )


def render_profile_page() -> None:
    st.header("Profile")
    st.caption("This is your career memory: the profile, projects, skills, and reusable resume evidence behind every match.")
    render_profile_summary_card(get_career_profile())
    st.divider()
    render_career_profile_page(show_header=False)
    st.divider()
    render_resume_assets_page(show_header=False)


def render_opportunities_page() -> None:
    st.header("Opportunities")
    st.caption("Discover jobs, review your best matches, and move quickly on the ones worth applying to.")
    profile = get_career_profile()
    render_compact_profile_strip(profile)
    discover_tab, queue_tab, analyze_tab = st.tabs(
        ["Discover Jobs", "Best Matches", "Analyze A Specific Job"]
    )
    with discover_tab:
        render_job_discovery_page(show_header=False, show_debug=False)
    with queue_tab:
        render_job_queue(show_header=False)
    with analyze_tab:
        render_analysis_page(show_header=False)


def render_applications_page() -> None:
    st.header("Applications")
    st.caption("Track what you saved, where you applied, and what needs a follow-up next.")
    applications = get_applications()

    if applications.empty:
        st.info("No saved applications yet.")
        return

    status_map = {
        "Saved": applications["status"] == "Saved",
        "Applied": applications["status"] == "Applied",
        "Interview": applications["status"] == "Interview",
        "Offer": applications["status"] == "Offer",
        "Rejected": applications["status"] == "Rejected",
        "No Response": (
            applications["outcome_status"].fillna("").eq("No Response")
            | applications["status"].fillna("").eq("No Response")
        ),
    }
    metric_cols = st.columns(6)
    for column, (label, mask) in zip(metric_cols, status_map.items()):
        column.metric(label, int(mask.sum()))

    display = applications[
        [
            "company",
            "job_title",
            "application_date",
            "status",
            "follow_up_date",
            "next_action",
            "notes",
        ]
    ].rename(
        columns={
            "company": "Company",
            "job_title": "Title",
            "application_date": "Date Applied",
            "status": "Status",
            "follow_up_date": "Follow-up Date",
            "next_action": "Next Action",
            "notes": "Notes",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)

    st.divider()
    render_tracker_page(show_header=False)


def render_insights_page() -> None:
    st.header("Insights")
    st.caption("Use these signals to sharpen your search, highlight the right work, and close the most important gaps.")
    profile = get_career_profile()
    queue = fetch_queue_records()
    top_companies = fetch_top_scoring_companies()
    missing_skills = fetch_most_common_missing_skills()

    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        st.subheader("Best Role Matches")
        if queue.empty:
            st.info("Not enough analyzed opportunities yet. Run discovery or analyze a specific job to unlock role-level insights.")
        else:
            best_roles = (
                queue.sort_values("fit_score", ascending=False)
                .head(5)[["job_title", "company", "fit_score"]]
                .rename(columns={"job_title": "Role", "company": "Company", "fit_score": "Match Score"})
            )
            st.dataframe(best_roles, width="stretch", hide_index=True)

        st.subheader("Common Missing Skills")
        if missing_skills.empty:
            fallback = compact_text(profile.get("missing_skills", ""), "")
            if fallback:
                st.markdown(format_list_block(fallback))
            else:
                st.info("No missing-skill trends yet. Once more jobs are analyzed, this section will get sharper.")
        else:
            st.dataframe(missing_skills.head(8), width="stretch", hide_index=True)

        st.subheader("Best Projects To Highlight")
        projects = profile.get("projects", []) or []
        if not projects:
            st.info("Add projects to your profile so the app can surface stronger application evidence.")
        else:
            project_rows = pd.DataFrame(projects)[
                ["project_name", "technical_stack", "business_impact", "target_roles_supported"]
            ].rename(
                columns={
                    "project_name": "Project",
                    "technical_stack": "Technical Stack",
                    "business_impact": "Business Impact",
                    "target_roles_supported": "Supported Roles",
                }
            )
            st.dataframe(project_rows.head(5), width="stretch", hide_index=True)

    with insight_col2:
        st.subheader("Companies Worth Tracking")
        if top_companies.empty:
            st.info("No company trend data yet. Save and analyze more jobs to build this view.")
        else:
            st.dataframe(top_companies, width="stretch", hide_index=True)

        st.subheader("Skill Gaps")
        if missing_skills.empty and not str(profile.get("missing_skills", "")).strip():
            st.info("No clear skill gaps yet. The app will summarize them as your opportunity set grows.")
        else:
            skill_gap_text = profile.get("missing_skills", "")
            if str(skill_gap_text).strip():
                st.markdown(format_list_block(skill_gap_text))

        st.subheader("Suggested Next Actions")
        suggestions = []
        if queue.empty:
            suggestions.append("Run discovery with sample data or public job pages to create an opportunities list.")
        if not (profile.get("projects", []) or []):
            suggestions.append("Add project evidence to your profile so match explanations can point to proof.")
        if missing_skills.empty and not str(profile.get("missing_skills", "")).strip():
            suggestions.append("Analyze a few target roles to surface recurring skill gaps.")
        if not suggestions:
            suggestions.append(recommend_next_action(profile, queue, get_applications()))
        st.markdown("\n".join(f"- {item}" for item in suggestions))


def render_settings_page() -> None:
    st.header("Settings / Usage")
    st.caption("API setup, usage controls, and deeper pipeline diagnostics live here instead of the main job-search workflow.")

    api_col1, api_col2 = st.columns(2)
    with api_col1:
        st.subheader("API Settings")
        st.metric("OpenAI API Key", "Configured" if get_openai_api_key() else "Missing")
        st.text_input("Model", value=get_openai_model(), disabled=True)
        st.caption("Future provider settings can live here later without changing the current app flow.")
    with api_col2:
        st.subheader("Usage Limits")
        last_settings = st.session_state.get("last_discovery_settings", {}) or {}
        st.metric("Max LLM Calls Per Run", int(last_settings.get("max_llm_calls_per_run", DEFAULT_MAX_LLM_CALLS_PER_RUN)))
        st.metric("Max Jobs Per Run", int(last_settings.get("max_jobs_per_run", DEFAULT_MAX_JOBS_PER_RUN)))
        st.metric("Pre-filter Threshold", int(last_settings.get("pre_filter_threshold", DEFAULT_PRE_FILTER_THRESHOLD)))

    metrics = st.session_state.get("last_discovery_metrics", {}) or {}
    st.divider()
    st.subheader("Usage")
    if not metrics:
        st.info("Run job discovery to populate usage and cache metrics.")
    else:
        usage_col1, usage_col2, usage_col3, usage_col4 = st.columns(4)
        usage_col1.metric("LLM Calls", int(metrics.get("actual_llm_calls_used", metrics.get("jobs_sent_to_llm", 0))))
        usage_col2.metric("Cache Hits", int(metrics.get("cache_hits", 0)))
        usage_col3.metric("Stale Cache Hits", int(metrics.get("stale_cache_hits", 0)))
        usage_col4.metric("Cache Misses", int(metrics.get("cache_misses", 0)))

        usage_col5, usage_col6, usage_col7, usage_col8 = st.columns(4)
        usage_col5.metric("Estimated Tokens Saved", int(metrics.get("estimated_tokens_saved", 0)))
        usage_col6.metric("Discovery Runtime", "Session-based")
        usage_col7.metric("Jobs Sent To LLM", int(metrics.get("jobs_sent_to_llm", 0)))
        usage_col8.metric("LLM Calls Avoided", int(metrics.get("llm_calls_avoided", 0)))

    st.divider()
    st.subheader("Advanced Debug")
    if not metrics:
        st.info("No discovery debug data in this session yet.")
    else:
        with st.expander("Pipeline Metrics", expanded=True):
            st.json(
                {
                    "jobs_discovered": metrics.get("jobs_discovered", 0),
                    "already_known_jobs": metrics.get("already_known_jobs", 0),
                    "new_jobs": metrics.get("new_jobs", 0),
                    "duplicates_removed": metrics.get("duplicates_removed", 0),
                    "jobs_rejected_by_rules": metrics.get("jobs_rejected_by_rules", 0),
                    "jobs_below_threshold": metrics.get("jobs_below_threshold", 0),
                    "jobs_pre_scored": metrics.get("jobs_pre_scored", 0),
                    "jobs_sent_to_llm": metrics.get("jobs_sent_to_llm", 0),
                    "jobs_added_to_queue": metrics.get("jobs_added_to_queue", 0),
                    "estimated_tokens_saved": metrics.get("estimated_tokens_saved", 0),
                }
            )
        with st.expander("Raw Discovery Log"):
            processed_jobs = metrics.get("processed_jobs", [])
            if processed_jobs:
                st.dataframe(pd.DataFrame(processed_jobs), width="stretch", hide_index=True)
            else:
                st.info("No raw discovery log captured in this session.")
        with st.expander("Recent Model Run Logs"):
            init_db()
            with get_connection() as conn:
                model_runs = pd.read_sql_query(
                    """
                    SELECT
                        id,
                        model_name,
                        fit_score,
                        created_at,
                        substr(analysis_text, 1, 240) AS analysis_excerpt
                    FROM model_runs
                    ORDER BY id DESC
                    LIMIT 10
                    """,
                    conn,
                )
            if model_runs.empty:
                st.info("No model run logs yet.")
            else:
                st.dataframe(model_runs, width="stretch", hide_index=True)


initialize_session_state()

st.title("AI Job Search Copilot")

st.info(
    "MVP note: Data is stored locally in SQLite. User data is only sent to the LLM API for analysis. No multi-user login yet."
)

page = st.sidebar.radio(
    "Page",
    [
        "Home",
        "Profile",
        "Opportunities",
        "Applications",
        "Insights",
        "Settings / Usage",
    ],
)

if page == "Home":
    render_home_page()
elif page == "Profile":
    render_profile_page()
elif page == "Opportunities":
    render_opportunities_page()
elif page == "Applications":
    render_applications_page()
elif page == "Insights":
    render_insights_page()
else:
    render_settings_page()
