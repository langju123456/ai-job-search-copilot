from src.database import fetch_career_profile, upsert_career_profile


def get_career_profile() -> dict:
    return fetch_career_profile()


def save_career_profile(profile: dict) -> None:
    upsert_career_profile(profile)
