import re

import streamlit as st
from dotenv import load_dotenv

from src.file_parser import extract_text_from_uploaded_file
from src.job_analyzer import analyze_job
from src.llm import get_openai_api_key
from src.networking import generate_networking_messages
from src.profile import get_career_profile, save_career_profile
from src.resume_tailor import tailor_resume
from src.scoring import (
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
        "score_breakdown": {},
        "computed_fit_score": 0,
        "priority": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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

    render_save_application_form()


def render_save_application_form() -> None:
    st.header("Save Application")
    with st.form("save_application_form"):
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Company")
            job_title = st.text_input("Job title")
            location = st.text_input("Location")
            job_url = st.text_input("Job URL")
            recruiter_name = st.text_input("Recruiter Name")
            application_date = st.text_input("Application Date", placeholder="YYYY-MM-DD")
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
            follow_up_date = st.text_input("Follow-up Date", placeholder="YYYY-MM-DD")
            interview_stage = st.text_input("Interview Stage")
        next_action = st.text_input("Next Action")
        notes = st.text_area("Notes", height=120)

        submitted = st.form_submit_button("Save Application")
        if submitted:
            save_application(
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
            )
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


initialize_session_state()

st.title("AI Job Search Copilot")

page = st.sidebar.radio(
    "Page",
    ["Analyze Job", "Setup", "Job Pipeline", "Application Tracker"],
)

if page == "Setup":
    render_setup_page()
elif page == "Job Pipeline":
    render_pipeline_page()
elif page == "Application Tracker":
    render_tracker_page()
else:
    render_analysis_page()
