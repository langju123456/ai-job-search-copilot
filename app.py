import re

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
    DEFAULT_PRE_FILTER_THRESHOLD,
    JOB_QUEUE_STATUS_OPTIONS,
    fetch_job_queue,
    generate_application_prep,
    parse_manual_jobs,
    parse_csv_jobs,
    parse_job_urls,
    process_discovered_jobs,
    update_application_prep,
    update_job_queue_status,
)
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
    st.header("Job Discovery")
    st.caption("This page does not scrape websites or submit applications.")

    profile = get_career_profile()
    career_profile_text = career_profile_to_text(profile)

    with st.form("job_discovery_form"):
        job_urls = st.text_area(
            "Paste job URLs",
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
        settings_col1, settings_col2, settings_col3 = st.columns(3)
        with settings_col1:
            pre_filter_threshold = st.number_input(
                "Pre-filter threshold",
                min_value=0,
                max_value=100,
                value=DEFAULT_PRE_FILTER_THRESHOLD,
                step=5,
            )
        with settings_col2:
            max_jobs_per_run = st.number_input(
                "Max jobs per run",
                min_value=1,
                max_value=200,
                value=50,
                step=5,
            )
        with settings_col3:
            max_llm_calls_per_run = st.number_input(
                "Max LLM calls per run",
                min_value=0,
                max_value=50,
                value=DEFAULT_MAX_LLM_CALLS_PER_RUN,
                step=1,
            )
        submitted = st.form_submit_button("Process Jobs")

    if submitted:
        jobs = []
        jobs.extend(parse_job_urls(job_urls))
        jobs.extend(parse_manual_jobs(raw_jds))
        if csv_upload:
            try:
                jobs.extend(parse_csv_jobs(csv_upload))
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                jobs = []

        jobs_with_descriptions = [job for job in jobs if job.get("jd_text", "").strip()]

        if not jobs:
            st.warning("Add at least one job URL, raw job description, or CSV row.")
        elif jobs_with_descriptions and not st.session_state.user_profile.strip():
            st.warning("Add your profile/resume on the Analyze Job page before scoring jobs with descriptions.")
        elif jobs_with_descriptions and not get_openai_api_key():
            st.warning("OPENAI_API_KEY is missing. Add it to your .env file before processing jobs with descriptions.")
        else:
            with st.spinner("Running discovery filters and LLM-limited deep analysis..."):
                try:
                    metrics = process_discovered_jobs(
                        jobs,
                        st.session_state.user_profile,
                        profile,
                        career_profile_text,
                        int(pre_filter_threshold),
                        int(max_llm_calls_per_run),
                        int(max_jobs_per_run),
                    )
                except Exception as exc:
                    st.error(f"Could not process discovered jobs: {exc}")
                    metrics = None

            if metrics:
                st.success(f"Processed {len(metrics['processed_jobs'])} job(s).")
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                metric_col1.metric("Jobs Discovered", metrics["jobs_discovered"])
                metric_col2.metric("Filtered Out", metrics["jobs_filtered_out"])
                metric_col3.metric("Pre-scored", metrics["jobs_pre_scored"])
                metric_col4.metric("Sent to LLM", metrics["jobs_sent_to_llm"])
                metric_col5.metric("Est. Token Savings", metrics["estimated_token_savings"])

                st.subheader("Imported Jobs")
                st.dataframe(metrics["imported_jobs"], width="stretch", hide_index=True)

                st.subheader("Filtered Out Jobs")
                st.dataframe(metrics["filtered_out_jobs"], width="stretch", hide_index=True)

                st.subheader("Jobs Sent to LLM")
                sent_to_llm = metrics["jobs_sent_to_llm_rows"]
                if sent_to_llm:
                    st.dataframe(sent_to_llm, width="stretch", hide_index=True)
                else:
                    st.info("No jobs were sent to the LLM in this run.")

    render_job_queue()


def render_job_queue() -> None:
    st.header("Job Queue")
    queue = fetch_job_queue()

    if queue.empty:
        st.info("No discovered jobs yet.")
        return

    display_columns = [
        "company",
        "job_title",
        "location",
        "pre_filter_score",
        "fit_score",
        "apply_decision",
        "decision_reason",
        "job_url",
        "status",
    ]
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        decision_filter = st.selectbox("Decision", ["All", "Apply", "Maybe", "Skip"])
    with filter_col2:
        minimum_fit_score = st.number_input("Minimum fit score", 0, 100, 0, 5)
    with filter_col3:
        company_filter = st.text_input("Company filter")
    with filter_col4:
        location_filter = st.text_input("Location filter")

    filtered_queue = queue.copy()
    if decision_filter != "All":
        filtered_queue = filtered_queue[filtered_queue["apply_decision"] == decision_filter]
    filtered_queue = filtered_queue[filtered_queue["fit_score"] >= minimum_fit_score]
    if company_filter:
        filtered_queue = filtered_queue[
            filtered_queue["company"].str.contains(company_filter, case=False, na=False)
        ]
    if location_filter:
        filtered_queue = filtered_queue[
            filtered_queue["location"].str.contains(location_filter, case=False, na=False)
        ]

    st.dataframe(filtered_queue[display_columns], width="stretch", hide_index=True)

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

    apply_jobs = queue[queue["apply_decision"] == "Apply"]
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
        if st.button("Generate Application Prep"):
            if not st.session_state.user_profile.strip():
                st.warning("Add your profile/resume on the Analyze Job page before generating prep.")
            elif not get_openai_api_key():
                st.warning("OPENAI_API_KEY is missing. Add it to your .env file before generating prep.")
            else:
                with st.spinner("Generating application prep..."):
                    prep = generate_application_prep(
                        st.session_state.user_profile,
                        selected_prep.jd_text,
                        career_profile_to_text(get_career_profile()),
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
        "Job Discovery",
        "Setup",
        "Job Pipeline",
        "Application Tracker",
        "Analytics",
    ],
)

if page == "Setup":
    render_setup_page()
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
