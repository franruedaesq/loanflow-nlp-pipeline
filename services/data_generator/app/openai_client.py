import json
import os
import random
import sys
import time
from typing import Dict

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import StructuredExample

# --------------------------------------------------------------------------- #
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RETRIES = 5
TEMPERATURE = 0.9


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is missing")
    return OpenAI(api_key=api_key)


@retry(
    wait=wait_exponential(multiplier=2, max=60), stop=stop_after_attempt(MAX_RETRIES)
)
def chat_completion(prompt: str, response_schema) -> Dict:
    """
    Call OpenAI chat completion with exponential back-off.
    Returns the *parsed* response as a dict validated by pydantic.
    """
    client = _get_client()
    try:
        resp = client.chat.completions.parse(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            response_format=response_schema,
        )
        print("API Response:", resp)

        # Parse the structured response
        parsed_data = resp.choices[0].message.parsed
        print("Parsed data:", parsed_data)

        # Convert to dict
        return parsed_data.model_dump()

    except Exception as exc:
        print(f"Error in chat_completion: {exc}")
        raise exc
