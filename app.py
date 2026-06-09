import re

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
from src.job_discovery import (
    DEFAULT_MAX_LLM_CALLS_PER_RUN,
    DEFAULT_MAX_JOBS_PER_RUN,
    DEFAULT_PRE_FILTER_THRESHOLD,
    JOB_QUEUE_STATUS_OPTIONS,
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
from src.llm import get_openai_api_key
from src.networking import generate_networking_messages
from src.profile import (
    generate_career_profile_from_resume,
    get_career_feedback_history,
    get_career_profile,
    save_career_profile,
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
    st.subheader("Career Profile Summary")
    if not any(profile.values()):
        st.info("No career profile saved yet.")
        return

    card_col1, card_col2 = st.columns(2)
    with card_col1:
        st.markdown(f"**Headline**\n\n{profile.get('headline', '') or profile.get('summary', '') or 'Not set'}")
        st.markdown(f"**Target Roles**\n\n{profile.get('target_roles', '') or 'Not set'}")
        st.markdown(f"**Top Skills**\n\n{profile.get('skills', '') or 'Not set'}")
        st.markdown(f"**Visa Status**\n\n{profile.get('visa_status', '') or 'Not set'}")
    with card_col2:
        st.markdown(f"**Acceptable Roles**\n\n{profile.get('acceptable_roles', '') or 'Not set'}")
        st.markdown(f"**Missing Skills**\n\n{profile.get('missing_skills', '') or 'Not set'}")
        st.markdown(f"**Preferred Locations**\n\n{profile.get('preferred_locations', '') or profile.get('suggested_locations', '') or 'Not set'}")
        st.markdown(f"**Years Experience**\n\n{profile.get('years_experience', '') or 'Not set'}")


def render_career_profile_page() -> None:
    st.header("Career Profile")
    st.caption("Generate and edit the structured profile that powers job discovery, fit scoring, and application prep.")

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
            except Exception as exc:
                st.error(f"Could not generate career profile: {exc}")

    saved_profile = get_career_profile()
    render_career_profile_summary(saved_profile)

    if any(saved_profile.values()):
        with st.form("career_profile_core_form"):
            st.subheader("Core Profile")
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

        st.subheader("Skill Graph")
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

        st.subheader("Project Memory")
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

        st.subheader("Feedback History")
        feedback_history = get_career_feedback_history()
        if feedback_history.empty:
            st.info("No career feedback history yet.")
        else:
            st.dataframe(feedback_history, width="stretch", hide_index=True)


def render_score_breakdown() -> None:
    scores = st.session_state.score_breakdown
    if not scores:
        return

    st.subheader("Structured Fit Score")
    st.metric("Computed Fit Score", f"{st.session_state.computed_fit_score}/100")
    score_df = score_breakdown_dataframe(scores)
    st.bar_chart(score_df.set_index("Component")["Score"])
    st.dataframe(score_df, width="stretch", hide_index=True)


def render_analysis_page() -> None:
    profile = get_career_profile()
    career_profile_text = career_profile_to_text(profile)

    if not get_openai_api_key():
        st.warning("OPENAI_API_KEY is missing. Add it to your .env file before running AI analysis.")

    st.header("User Profile / Resume")
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

    st.header("Job Description")
    st.subheader("Job Metadata")
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
                with st.spinner("Analyzing job fit..."):
                    st.session_state.analysis = analyze_job(
                        st.session_state.user_profile,
                        st.session_state.job_description,
                        career_profile_text,
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
                )
                st.session_state.analysis_user_id = data_links["user_id"]
                st.session_state.analysis_company_id = data_links["company_id"]
                st.session_state.analysis_job_id = data_links["job_id"]
                st.session_state.analysis_model_run_id = data_links["model_run_id"]
            except Exception as exc:
                st.error(f"LLM call failed: {exc}")

    if st.session_state.analysis:
        st.header("Job Fit Analysis")
        render_score_breakdown()
        if st.session_state.priority:
            st.info(f"Pipeline priority: {st.session_state.priority}")
        st.markdown(st.session_state.analysis)

    if st.session_state.resume_tailoring:
        st.header("Resume Tailoring")
        st.markdown(st.session_state.resume_tailoring)
        st.download_button(
            "Export Resume Suggestions (.txt)",
            data=st.session_state.resume_tailoring,
            file_name="resume_suggestions.txt",
            mime="text/plain",
        )

    if st.session_state.networking_messages:
        st.header("Networking Messages")
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


def render_tracker_page() -> None:
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


def render_analytics_page() -> None:
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


def render_job_discovery_page() -> None:
    st.header("Public Job Discovery")
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
    st.markdown("**Using Career Profile:**")
    if profile_text:
        st.info(profile.get("headline", "") or profile.get("summary", "") or profile_text)
    else:
        st.warning("No saved career profile yet. Use the Career Profile page for better fit scoring.")

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
                metric_defaults = {
                    "already_known_jobs": 0,
                    "new_jobs": len(metrics.get("queued_jobs", [])),
                    "jobs_rejected_by_rules": 0,
                    "jobs_below_threshold": 0,
                    "jobs_filtered_rejected": metrics.get("jobs_filtered_out", 0),
                    "jobs_added_to_queue": len(metrics.get("queued_jobs", [])),
                    "actual_llm_calls_used": metrics.get("jobs_sent_to_llm", 0),
                    "llm_calls_avoided": metrics.get("skipped_llm_calls", 0),
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "estimated_tokens_saved": metrics.get("estimated_token_savings", 0),
                    "token_savings_formula": "llm_calls_avoided * 750 tokens",
                    "known_jobs": [],
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
                st.success(f"Added {metrics['jobs_added_to_queue']} job(s) to the main queue.")
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
                        st.dataframe(metrics["rejected_by_rules_jobs"], width="stretch", hide_index=True)
                    else:
                        st.info("No jobs were rejected by deterministic rules.")

                with st.expander("Below Threshold", expanded=True):
                    if metrics["below_threshold_jobs"]:
                        st.dataframe(metrics["below_threshold_jobs"], width="stretch", hide_index=True)
                    else:
                        st.info("No jobs fell below the pre-filter threshold.")

                with st.expander("Sent to LLM"):
                    sent_to_llm = metrics["jobs_sent_to_llm_rows"]
                    if sent_to_llm:
                        st.dataframe(sent_to_llm, width="stretch", hide_index=True)
                    else:
                        st.info("No jobs were sent to the LLM in this run.")

                with st.expander("Main Queue", expanded=True):
                    if metrics["queued_jobs"]:
                        st.dataframe(metrics["queued_jobs"], width="stretch", hide_index=True)
                    else:
                        st.info("No jobs qualified for the main queue in this run.")

                with st.expander("Full Raw Discovery Log"):
                    st.dataframe(metrics["processed_jobs"], width="stretch", hide_index=True)

                with st.expander("Already Known Jobs"):
                    if metrics["known_jobs"]:
                        st.dataframe(metrics["known_jobs"], width="stretch", hide_index=True)
                    else:
                        st.info("No historical jobs were reused from cache.")

                st.download_button(
                    "Export Discovered Jobs CSV",
                    data=pd.DataFrame(metrics["imported_jobs"]).to_csv(index=False),
                    file_name="discovered_jobs.csv",
                    mime="text/csv",
                )

                with st.expander("Duplicates Removed"):
                    if metrics["duplicate_jobs"]:
                        st.dataframe(metrics["duplicate_jobs"], width="stretch", hide_index=True)
                    else:
                        st.info("No duplicates removed in this run.")

                with st.expander("Pre-Scored Jobs", expanded=True):
                    if metrics["pre_scored_jobs"]:
                        st.dataframe(metrics["pre_scored_jobs"], width="stretch", hide_index=True)
                    else:
                        st.info("No jobs reached pre-scoring in this run.")

    render_job_queue()


def render_job_queue() -> None:
    st.header("Job Queue")
    queue = fetch_job_queue()

    if queue.empty:
        st.info("No discovered jobs yet.")
        return

    queue_defaults = {
        "post_time": "Unknown",
        "job_level": "Unknown",
        "work_mode": "Unknown",
        "queue_category": "",
        "rejection_reason": "",
    }
    for column_name, default_value in queue_defaults.items():
        if column_name not in queue.columns:
            queue[column_name] = default_value
    queue["queue_category"] = queue["queue_category"].where(
        queue["queue_category"].astype(str).str.strip() != "",
        queue["apply_decision"],
    )

    display_columns = [
        "company",
        "job_title",
        "location",
        "source",
        "post_time",
        "job_level",
        "work_mode",
        "pre_filter_score",
        "fit_score",
        "apply_decision",
        "decision_reason",
        "status",
        "job_url",
    ]
    decision_col1, decision_col2 = st.columns(2)
    with decision_col1:
        apply_only = st.checkbox("Apply only")
    with decision_col2:
        maybe_only = st.checkbox("Maybe only")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        post_time_filter = st.text_input("Post time")
    with filter_col2:
        job_level_filter = st.selectbox(
            "Job level",
            ["All", "Internship", "Entry", "Junior", "Mid", "Senior", "Staff", "Principal", "Director", "Unknown"],
        )
    with filter_col3:
        work_mode_filter = st.selectbox("Work mode", ["All", "Remote", "Hybrid", "Onsite", "Unknown"])
    with filter_col4:
        source_filter = st.text_input("Source filter")

    filter_col5, filter_col6, filter_col7, filter_col8 = st.columns(4)
    with filter_col5:
        location_filter = st.text_input("Location filter")
    with filter_col6:
        company_filter = st.text_input("Company filter")
    with filter_col7:
        minimum_pre_filter_score = st.number_input("Minimum pre-filter score", 0, 100, 0, 5)
    with filter_col8:
        minimum_fit_score = st.number_input("Minimum fit score", 0, 100, 0, 5)

    filtered_queue = queue.copy()
    filtered_queue = filtered_queue[
        filtered_queue["queue_category"].isin(["Apply", "Maybe"])
    ]
    if apply_only and maybe_only:
        filtered_queue = filtered_queue[
            filtered_queue["queue_category"].isin(["Apply", "Maybe"])
        ]
    elif apply_only:
        filtered_queue = filtered_queue[filtered_queue["queue_category"] == "Apply"]
    elif maybe_only:
        filtered_queue = filtered_queue[filtered_queue["queue_category"] == "Maybe"]
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

    st.dataframe(filtered_queue[display_columns], width="stretch", hide_index=True)

    apply_queue = queue[queue["apply_decision"] == "Apply"]
    if not apply_queue.empty:
        st.download_button(
            "Export Apply Queue CSV",
            data=apply_queue[display_columns].to_csv(index=False),
            file_name="apply_queue.csv",
            mime="text/csv",
        )

    st.subheader("Update Queue Status")
    with st.form("job_queue_status_form"):
        options = {
            f"{row.company} - {row.job_title or 'Untitled Job'} (#{row.id})": int(row.id)
            for row in filtered_queue.itertuples()
        }
        if not options:
            st.info("No jobs match the current filters.")
            return
        selected_job = st.selectbox("Job", list(options.keys()))
        status = st.selectbox("Status", JOB_QUEUE_STATUS_OPTIONS)
        if st.form_submit_button("Update Status"):
            update_job_queue_status(options[selected_job], status)
            st.success("Job status updated.")

    apply_jobs = apply_queue
    if apply_jobs.empty:
        return

    st.header("Application Prep")
    prep_options = {
        f"{row.company} - {row.job_title or 'Untitled Job'} (#{row.id})": row
        for row in apply_jobs.itertuples()
    }
    selected_prep_label = st.selectbox("Ready-to-apply job", list(prep_options.keys()))
    selected_prep = prep_options[selected_prep_label]

    st.subheader("Tailored Resume Bullets")
    if not selected_prep.resume_bullets and selected_prep.jd_text:
        st.caption("This uses an LLM call.")
        if st.button("Generate Application Prep"):
            if not st.session_state.user_profile.strip():
                st.warning("Add your profile/resume on the Analyze Job page before generating prep.")
            elif not get_openai_api_key():
                st.warning("OPENAI_API_KEY is missing. Add it to your .env file before generating prep.")
            else:
                with st.spinner("Generating application prep..."):
                    prep = generate_application_prep(
                        st.session_state.user_profile or compact_career_profile_text(get_career_profile()),
                        selected_prep.jd_text,
                        compact_career_profile_text(get_career_profile(), selected_prep.jd_text),
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


initialize_session_state()

st.title("AI Job Search Copilot")

st.info(
    "MVP note: Data is stored locally in SQLite. User data is only sent to the LLM API for analysis. No multi-user login yet."
)

page = st.sidebar.radio(
    "Page",
    [
        "Analyze Job",
        "Career Profile",
        "Job Discovery",
        "Setup",
        "Job Pipeline",
        "Application Tracker",
        "Analytics",
    ],
)

if page == "Setup":
    render_setup_page()
elif page == "Career Profile":
    render_career_profile_page()
elif page == "Job Discovery":
    render_job_discovery_page()
elif page == "Job Pipeline":
    render_pipeline_page()
elif page == "Application Tracker":
    render_tracker_page()
elif page == "Analytics":
    render_analytics_page()
else:
    render_analysis_page()
