"""
agents/proposer.py — Gemini 2.5 Flash as the Proposer.
Writes the first structured draft given the task + live evidence.
Uses the REST v1beta endpoint with x-goog-api-key header (current auth standard).
"""
import json
import httpx
from core.config import Config
from core.models import EvidencePacket, Proposal
from core.logger import get_logger

log = get_logger(__name__)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_SYSTEM = """You are the Proposer in an adversarial multi-agent validation debate.
Your job: write the BEST POSSIBLE first draft answering the user's task.
- Ground your answer in the live evidence provided.
- Be thorough, structured, and cite evidence snippets where relevant.
- At the end, include a short section called "Reasoning trace:" summarising your logic.
This draft will be critiqued by two independent AI critics. Write to survive scrutiny."""


async def _call_gemini(api_key: str, model: str, system: str, user: str) -> str:
    url = _ENDPOINT.format(model=model)
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response shape: {e}\n{json.dumps(data)[:400]}")


class ProposerAgent:
    def __init__(self, config: Config):
        self._config = config

    async def propose(self, task: str, evidence: EvidencePacket) -> Proposal:
        log.info(f"[bold purple]Proposer[/bold purple] (Gemini {self._config.gemini_model}) drafting…")

        evidence_block = _format_evidence(evidence)
        user_prompt = f"""TASK:\n{task}\n\n{evidence_block}\n\nWrite your best answer now."""

        text = await _call_gemini(
            api_key=self._config.gemini_api_key,
            model=self._config.gemini_model,
            system=_SYSTEM,
            user=user_prompt,
        )

        # Split off reasoning trace if present
        reasoning = None
        if "Reasoning trace:" in text:
            parts = text.split("Reasoning trace:", 1)
            answer_body = parts[0].strip()
            reasoning = parts[1].strip()
        else:
            answer_body = text.strip()

        log.info("Proposer draft complete.")
        return Proposal(
            content=answer_body,
            model=self._config.gemini_model,
            reasoning=reasoning,
        )


def _format_evidence(evidence: EvidencePacket) -> str:
    if not evidence.snippets and not evidence.raw_answer:
        return ""
    lines = ["--- Live evidence ---"]
    if evidence.raw_answer:
        lines.append(f"Summary: {evidence.raw_answer}")
    for i, (s, u) in enumerate(zip(evidence.snippets, evidence.urls), 1):
        lines.append(f"[{i}] {u}\n{s[:350]}")
    return "\n\n".join(lines)
