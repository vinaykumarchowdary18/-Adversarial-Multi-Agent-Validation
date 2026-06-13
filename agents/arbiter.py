"""
agents/arbiter.py — Arbiter: GPT-4o-mini via GitHub Models.
Sees BOTH critic verdicts, scores consensus, decides: finalize | revise | escalate.
Uses the OpenAI-compatible GitHub Models inference endpoint.
"""
import json
from core.config import Config
from core.models import Proposal, EvidencePacket, CriticVerdict, ArbiterDecision
from core.logger import get_logger
from agents._openai_compat import openai_compat_call

log = get_logger(__name__)

_ENDPOINT = "https://models.github.ai/inference/chat/completions"

_SYSTEM = """You are the Arbiter in an adversarial multi-agent validation debate.
You receive a proposal and two fully independent critic evaluations.
Your job:
  1. Score consensus: average the critics' scores, penalise if they strongly diverge.
  2. Identify which critique points are valid (accept) vs nitpicking / wrong (reject).
  3. Issue one of three directives:
     - "finalize"  — quality is acceptable; output the proposal as-is (or lightly polished).
     - "revise"    — quality needs work; provide an improved rewrite in `revised_proposal`.
     - "escalate"  — fundamental factual or logical failure; the proposer must start over.

Consensus score rules:
  • Start with the average of critic_a_score and critic_b_score.
  • If critics diverge by > 0.3, subtract 0.1 (penalise disagreement).
  • If both verdicts are "reject", cap score at 0.4.
  • If both are "accept", floor score at 0.65.

Directive rules:
  • consensus_score >= MIN_CONSENSUS_SCORE  →  "finalize"
  • 0.4 <= consensus_score < MIN_CONSENSUS_SCORE  →  "revise" (provide revised_proposal)
  • consensus_score < 0.4  →  "escalate"

Respond ONLY with valid JSON matching this schema exactly:
{
  "consensus_score": <float 0.0-1.0>,
  "accepted_points": ["<point description>", ...],
  "rejected_points": ["<point description>", ...],
  "directive": "finalize" | "revise" | "escalate",
  "revised_proposal": "<full improved text, or null if directive != revise>",
  "reasoning": "<2-3 sentences explaining your ruling>"
}
No markdown. No preamble. Raw JSON only."""


class ArbiterAgent:
    ARBITER_ID = "arbiter"

    def __init__(self, config: Config):
        self._config = config
        self._min_score = config.min_consensus_score

    async def arbitrate(
        self,
        proposal: Proposal,
        verdict_a: CriticVerdict,
        verdict_b: CriticVerdict,
        evidence: EvidencePacket,
        round_number: int = 1,
    ) -> ArbiterDecision:
        log.info(
            f"[bold orange]Arbiter[/bold orange] (GitHub/{self._config.github_model}) "
            f"ruling on round {round_number}…"
        )

        user_prompt = _build_prompt(proposal, verdict_a, verdict_b, evidence, self._min_score)

        raw = await openai_compat_call(
            endpoint=_ENDPOINT,
            api_key=self._config.github_token,
            model=self._config.github_model,
            system=_SYSTEM,
            user=user_prompt,
            temperature=0.1,
            max_tokens=1800,
            extra_headers={
                "X-GitHub-Api-Version": "2026-03-10",
                "Accept": "application/vnd.github+json",
            },
        )

        data = _parse_json(raw)

        decision = ArbiterDecision(
            consensus_score=float(data.get("consensus_score", 0.5)),
            accepted_points=data.get("accepted_points", []),
            rejected_points=data.get("rejected_points", []),
            directive=data.get("directive", "revise"),
            revised_proposal=data.get("revised_proposal"),
            reasoning=data.get("reasoning", ""),
        )

        log.info(
            f"Arbiter ruling: {decision.directive} "
            f"(consensus={decision.consensus_score:.2f}, "
            f"accepted={len(decision.accepted_points)}, "
            f"rejected={len(decision.rejected_points)})"
        )
        return decision


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_prompt(
    proposal: Proposal,
    verdict_a: CriticVerdict,
    verdict_b: CriticVerdict,
    evidence: EvidencePacket,
    min_score: float,
) -> str:
    def fmt_verdict(v: CriticVerdict) -> str:
        points = "\n".join(
            f"  [{p.severity.upper()}] ({p.category}) {p.description}"
            + (f"\n    Fix: {p.suggested_fix}" if p.suggested_fix else "")
            for p in v.critique_points
        )
        return (
            f"Critic ID: {v.critic_id}  Model: {v.model}\n"
            f"Verdict: {v.verdict}  Score: {v.score:.2f}\n"
            f"Summary: {v.summary}\n"
            f"Points:\n{points if points else '  (none)'}"
        )

    evidence_block = (
        "[No external evidence available]"
        if not evidence.snippets
        else "\n".join(f"[{i}] {s[:200]}" for i, s in enumerate(evidence.snippets, 1))
    )

    return (
        f"MIN_CONSENSUS_SCORE (from config): {min_score}\n\n"
        f"=== PROPOSAL (by {proposal.model}) ===\n{proposal.content}\n\n"
        f"=== CRITIC A EVALUATION ===\n{fmt_verdict(verdict_a)}\n\n"
        f"=== CRITIC B EVALUATION ===\n{fmt_verdict(verdict_b)}\n\n"
        f"=== ORIGINAL EVIDENCE ===\n{evidence_block}\n\n"
        f"Issue your ruling now. Respond with JSON only."
    )


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Arbiter returned invalid JSON: {e}\nRaw: {raw[:300]}")
