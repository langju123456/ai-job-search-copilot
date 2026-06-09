import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


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
