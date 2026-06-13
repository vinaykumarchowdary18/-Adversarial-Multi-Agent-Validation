"""
agents/critic_a.py — Critic A: Llama 3.3-70b via Groq.
Focuses on factual accuracy and logical consistency.
"""
import json
from core.config import Config
from core.models import Proposal, EvidencePacket, CriticVerdict, CritiquePoint
from core.logger import get_logger
from agents._openai_compat import openai_compat_call

log = get_logger(__name__)

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM = """You are Critic A in an adversarial multi-agent validation debate.
Your adversarial role: CHALLENGE the proposal rigorously.
Your specific focus areas:
  1. Factual accuracy — are the claims supported by the evidence provided?
  2. Logical consistency — do the conclusions follow from the premises?
  3. Missing facts — what important information was left out?

Respond ONLY with valid JSON matching this schema exactly:
{
  "verdict": "accept" | "revise" | "reject",
  "score": <float 0.0-1.0>,
  "summary": "<one sentence overall assessment>",
  "critique_points": [
    {
      "severity": "critical" | "major" | "minor",
      "category": "factual" | "logical" | "completeness" | "clarity",
      "description": "<what is wrong>",
      "suggested_fix": "<how to fix it>"
    }
  ]
}
No markdown. No preamble. Raw JSON only."""


class CriticAgentA:
    CRITIC_ID = "critic_a"

    def __init__(self, config: Config):
        self._config = config

    async def critique(self, proposal: Proposal, evidence: EvidencePacket) -> CriticVerdict:
        log.info(f"[bold teal]Critic A[/bold teal] (Groq/{self._config.groq_model}) evaluating…")

        evidence_block = _format_evidence(evidence)
        user_prompt = (
            f"ORIGINAL TASK EVIDENCE:\n{evidence_block}\n\n"
            f"PROPOSAL TO CRITIQUE:\n{proposal.content}\n\n"
            f"Apply your critique framework now. Respond with JSON only."
        )

        raw = await openai_compat_call(
            endpoint=_ENDPOINT,
            api_key=self._config.groq_api_key,
            model=self._config.groq_model,
            system=_SYSTEM,
            user=user_prompt,
            temperature=0.1,
        )

        data = _parse_json(raw)
        points = [CritiquePoint(**p) for p in data.get("critique_points", [])]

        verdict = CriticVerdict(
            critic_id=self.CRITIC_ID,
            model=self._config.groq_model,
            verdict=data.get("verdict", "revise"),
            score=float(data.get("score", 0.5)),
            critique_points=points,
            summary=data.get("summary", ""),
        )
        log.info(f"Critic A verdict: {verdict.verdict} (score={verdict.score:.2f}, {len(points)} points)")
        return verdict


def _parse_json(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Critic A returned invalid JSON: {e}\nRaw: {raw[:300]}")


def _format_evidence(evidence: EvidencePacket) -> str:
    if not evidence.snippets:
        return "[No external evidence]"
    return "\n".join(f"[{i}] {s[:300]}" for i, s in enumerate(evidence.snippets, 1))
