"""
agents/critic_b.py — Critic B: DeepSeek v3 via OpenRouter.
Focuses on completeness, alternative perspectives, and clarity.
Intentionally uses a different model family than Critic A for genuine diversity.
"""
import json
from core.config import Config
from core.models import Proposal, EvidencePacket, CriticVerdict, CritiquePoint
from core.logger import get_logger
from agents._openai_compat import openai_compat_call

log = get_logger(__name__)

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_SYSTEM = """You are Critic B in an adversarial multi-agent validation debate.
Your adversarial role: provide a FULLY INDEPENDENT second opinion on the proposal.
Your specific focus areas:
  1. Completeness — are important angles, edge cases, or caveats missing?
  2. Alternative perspectives — what would a domain expert object to?
  3. Clarity — is the answer ambiguous, jargon-heavy, or poorly structured?

You have NOT seen Critic A's evaluation. Be genuinely independent.

Respond ONLY with valid JSON matching this schema exactly:
{
  "verdict": "accept" | "revise" | "reject",
  "score": <float 0.0-1.0>,
  "summary": "<one sentence overall assessment>",
  "critique_points": [
    {
      "severity": "critical" | "major" | "minor",
      "category": "factual" | "logical" | "completeness" | "clarity",
      "description": "<what is wrong or missing>",
      "suggested_fix": "<how to address it>"
    }
  ]
}
No markdown. No preamble. Raw JSON only."""


class CriticAgentB:
    CRITIC_ID = "critic_b"

    def __init__(self, config: Config):
        self._config = config

    async def critique(self, proposal: Proposal, evidence: EvidencePacket) -> CriticVerdict:
        log.info(f"[bold coral]Critic B[/bold coral] (OpenRouter/{self._config.openrouter_model}) evaluating…")

        evidence_block = _format_evidence(evidence)
        user_prompt = (
            f"ORIGINAL TASK EVIDENCE:\n{evidence_block}\n\n"
            f"PROPOSAL TO CRITIQUE:\n{proposal.content}\n\n"
            f"Apply your critique framework now. Respond with JSON only."
        )

        raw = await openai_compat_call(
            endpoint=_ENDPOINT,
            api_key=self._config.openrouter_api_key,
            model=self._config.openrouter_model,
            system=_SYSTEM,
            user=user_prompt,
            temperature=0.1,
            # OpenRouter recommends these headers for rate-limit transparency
            extra_headers={
                "HTTP-Referer": "https://github.com/vinay/amav",
                "X-Title": "AMAV Personal AI",
            },
        )

        data = _parse_json(raw)
        points = [CritiquePoint(**p) for p in data.get("critique_points", [])]

        verdict = CriticVerdict(
            critic_id=self.CRITIC_ID,
            model=self._config.openrouter_model,
            verdict=data.get("verdict", "revise"),
            score=float(data.get("score", 0.5)),
            critique_points=points,
            summary=data.get("summary", ""),
        )
        log.info(f"Critic B verdict: {verdict.verdict} (score={verdict.score:.2f}, {len(points)} points)")
        return verdict


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Critic B returned invalid JSON: {e}\nRaw: {raw[:300]}")


def _format_evidence(evidence: EvidencePacket) -> str:
    if not evidence.snippets:
        return "[No external evidence]"
    return "\n".join(f"[{i}] {s[:300]}" for i, s in enumerate(evidence.snippets, 1))
