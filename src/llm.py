import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
RUNTIME_OPENAI_API_KEY: Optional[str] = None


def get_openai_api_key() -> Optional[str]:
    return RUNTIME_OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")


def set_runtime_openai_api_key(api_key: str) -> None:
    global RUNTIME_OPENAI_API_KEY
    RUNTIME_OPENAI_API_KEY = (api_key or "").strip() or None


def validate_openai_api_key(api_key: str) -> tuple[bool, str]:
    cleaned = (api_key or "").strip()
    if not cleaned:
        return False, "Enter an OpenAI API key to continue."
    try:
        client = OpenAI(api_key=cleaned)
        client.models.retrieve(get_openai_model())
    except Exception:
        return False, "Invalid API key or model access issue. Double-check the key and try again."
    return True, "Valid API key"


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def call_llm(system_prompt: str, user_prompt: str) -> str:
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=get_openai_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content or ""
