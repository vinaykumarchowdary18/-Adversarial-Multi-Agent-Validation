"""
core/orchestrator.py — the debate loop engine.
Wires: Evidence → Proposer → [Critic A ‖ Critic B] → Arbiter → repeat or finalize.
Returns a fully validated FinalAnswer.
"""
import asyncio
from datetime import datetime, timezone

from core.config import Config
from core.models import (
    EvidencePacket, Proposal, CriticVerdict,
    DebateRound, ArbiterDecision, FinalAnswer,
)
from core.logger import get_logger
from tools.evidence import EvidenceTool
from agents.proposer import ProposerAgent
from agents.critic_a import CriticAgentA
from agents.critic_b import CriticAgentB
from agents.arbiter import ArbiterAgent

log = get_logger(__name__)


class Orchestrator:
    """
    Runs the full AMAV debate pipeline for a single task.

    Pipeline per round:
      1. Proposer drafts an answer (Gemini).
      2. Critic A and Critic B evaluate in parallel (Groq + OpenRouter).
      3. Arbiter reconciles, scores, decides (GitHub Models / GPT).
      4. If directive == "revise", use revised proposal and loop.
      5. If directive == "escalate", force-restart with a stronger prompt.
      6. If directive == "finalize" OR max rounds reached → emit FinalAnswer.
    """

    def __init__(self, config: Config):
        self._cfg = config
        self._evidence_tool = EvidenceTool(api_key=config.tavily_api_key)
        self._proposer = ProposerAgent(config)
        self._critic_a = CriticAgentA(config)
        self._critic_b = CriticAgentB(config)
        self._arbiter = ArbiterAgent(config)

    async def run(self, task: str) -> FinalAnswer:
        log.info(f"\n{'='*60}")
        log.info(f"[bold white]AMAV debate starting[/bold white]")
        log.info(f"Task: {task[:120]}{'…' if len(task) > 120 else ''}")
        log.info(f"Max rounds: {self._cfg.max_debate_rounds}  |  "
                 f"Min consensus: {self._cfg.min_consensus_score}")
        log.info(f"{'='*60}\n")

        # ── Step 0: gather live evidence once ────────────────────────────────
        evidence_query = task[:380] if len(task) > 380 else task
        evidence: EvidencePacket = await self._evidence_tool.fetch(evidence_query)

        rounds: list[DebateRound] = []
        current_proposal: Proposal | None = None
        last_decision: ArbiterDecision | None = None

        for round_num in range(1, self._cfg.max_debate_rounds + 1):
            log.info(f"\n[bold]── Round {round_num}/{self._cfg.max_debate_rounds} ──[/bold]")

            # ── Step 1: Propose ───────────────────────────────────────────────
            if current_proposal is None:
                # First round: fresh proposal
                current_proposal = await self._proposer.propose(task, evidence)
            elif last_decision and last_decision.directive == "revise" and last_decision.revised_proposal:
                # Arbiter supplied a rewrite — use it directly, skip Gemini call
                log.info("Using arbiter's revised proposal for next round.")
                current_proposal = Proposal(
                    content=last_decision.revised_proposal,
                    model=f"{current_proposal.model}+arbiter",
                )
            else:
                # Escalate or first round fallback: re-propose with stronger prompt
                log.info("Escalation detected — re-proposing from scratch.")
                escalation_task = (
                    f"{task}\n\n"
                    f"IMPORTANT — PREVIOUS DRAFT FAILED CRITIC REVIEW.\n"
                    f"Prior arbiter reasoning: {last_decision.reasoning if last_decision else 'N/A'}\n"
                    f"Be especially rigorous, thorough, and cite evidence explicitly."
                )
                current_proposal = await self._proposer.propose(escalation_task, evidence)

            # ── Step 2: Critique in parallel ──────────────────────────────────
            log.info("Running Critic A and Critic B in parallel…")
            verdict_a, verdict_b = await asyncio.gather(
                self._critic_a.critique(current_proposal, evidence),
                self._critic_b.critique(current_proposal, evidence),
            )

            # ── Step 3: Arbitrate ─────────────────────────────────────────────
            decision = await self._arbiter.arbitrate(
                current_proposal, verdict_a, verdict_b, evidence, round_num
            )
            last_decision = decision

            # Record round
            debate_round = DebateRound(
                round_number=round_num,
                proposal=current_proposal,
                critique_a=verdict_a,
                critique_b=verdict_b,
                arbiter_notes=decision.reasoning,
            )
            rounds.append(debate_round)

            _log_round_summary(round_num, verdict_a, verdict_b, decision)

            # ── Step 4: decide whether to loop ───────────────────────────────
            if decision.directive == "finalize":
                log.info(f"\n✅ [bold green]Consensus reached[/bold green] at round {round_num}.")
                break

            if round_num == self._cfg.max_debate_rounds:
                log.warning(
                    f"\n⚠️  Max rounds ({self._cfg.max_debate_rounds}) reached without "
                    f"consensus. Finalizing with best available answer."
                )
                break

            # Prepare next round
            if decision.directive == "revise" and decision.revised_proposal:
                # Arbiter already rewrote — loop with it
                pass
            # escalate falls through to loop naturally (current_proposal stays as-is,
            # next iteration detects escalate and calls proposer again)

        # ── Build FinalAnswer ─────────────────────────────────────────────────
        final_content = _pick_final_content(current_proposal, last_decision)
        final_score = last_decision.consensus_score if last_decision else 0.5
        dissents = _collect_dissents(rounds)

        answer = FinalAnswer(
            task=task,
            answer=final_content,
            confidence=_score_to_confidence(final_score),
            consensus_score=final_score,
            debate_rounds=len(rounds),
            sources=evidence.urls,
            dissenting_points=dissents,
            models_used={
                "proposer": self._cfg.gemini_model,
                "critic_a": self._cfg.groq_model,
                "critic_b": self._cfg.openrouter_model,
                "arbiter": self._cfg.github_model,
                "evidence": "tavily",
            },
        )

        log.info(
            f"\n[bold]Final answer ready[/bold] | "
            f"confidence={answer.confidence:.0%} | "
            f"consensus={answer.consensus_score:.2f} | "
            f"rounds={answer.debate_rounds} | "
            f"sources={len(answer.sources)}"
        )
        return answer


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick_final_content(proposal: Proposal | None, decision: ArbiterDecision | None) -> str:
    if decision and decision.directive == "revise" and decision.revised_proposal:
        return decision.revised_proposal
    if proposal:
        return proposal.content
    return "[No answer could be produced]"


def _score_to_confidence(score: float) -> float:
    """Map consensus score to a user-facing confidence value (same scale, clamped)."""
    return max(0.0, min(1.0, score))


def _collect_dissents(rounds: list[DebateRound]) -> list[str]:
    """Gather unresolved critical/major critique points from the final round."""
    if not rounds:
        return []
    last = rounds[-1]
    dissents = []
    for verdict in (last.critique_a, last.critique_b):
        for pt in verdict.critique_points:
            if pt.severity in ("critical", "major"):
                dissents.append(
                    f"[{verdict.critic_id}/{pt.severity}] {pt.description}"
                )
    return dissents


def _log_round_summary(
    round_num: int,
    verdict_a: CriticVerdict,
    verdict_b: CriticVerdict,
    decision: ArbiterDecision,
) -> None:
    log.info(
        f"\n  Round {round_num} summary:\n"
        f"    Critic A  → {verdict_a.verdict:8s}  score={verdict_a.score:.2f}\n"
        f"    Critic B  → {verdict_b.verdict:8s}  score={verdict_b.score:.2f}\n"
        f"    Arbiter   → {decision.directive:8s}  consensus={decision.consensus_score:.2f}\n"
        f"    Reasoning : {decision.reasoning[:120]}{'…' if len(decision.reasoning)>120 else ''}"
    )
