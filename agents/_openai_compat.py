"""
agents/_openai_compat.py — shared async caller for any OpenAI-compatible endpoint.
Used by both Groq and OpenRouter agents.
"""
import json
import httpx
from core.logger import get_logger

log = get_logger(__name__)


async def openai_compat_call(
    endpoint: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    extra_headers: dict | None = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **(extra_headers or {}),
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {resp.status_code} from {endpoint}: {resp.text[:300]}"
            ) from e

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected response shape from {endpoint}: {e}\n{json.dumps(data)[:300]}")
