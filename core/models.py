"""
core/models.py — shared Pydantic models for the AMAV pipeline.
All agents pass these objects between stages; nothing is stringly typed.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class EvidencePacket(BaseModel):
    """Raw evidence returned by Tavily before any agent sees it."""
    query: str
    snippets: list[str]
    urls: list[str]
    raw_answer: Optional[str] = None


class Proposal(BaseModel):
    """The proposer's first-draft answer."""
    content: str
    model: str
    reasoning: Optional[str] = None  # chain-of-thought if present


class CritiquePoint(BaseModel):
    """A single issue raised by a critic."""
    severity: str          # "critical" | "major" | "minor"
    category: str          # "factual" | "logical" | "completeness" | "clarity"
    description: str
    suggested_fix: Optional[str] = None


class CriticVerdict(BaseModel):
    """One critic's full evaluation of a proposal."""
    critic_id: str         # "critic_a" | "critic_b"
    model: str
    verdict: str           # "accept" | "revise" | "reject"
    score: float = Field(ge=0.0, le=1.0)   # confidence in proposal quality
    critique_points: list[CritiquePoint]
    summary: str


class DebateRound(BaseModel):
    """One full round: proposal + both critiques."""
    round_number: int
    proposal: Proposal
    critique_a: CriticVerdict
    critique_b: CriticVerdict
    arbiter_notes: Optional[str] = None    # filled after arbiter evaluates


class ArbiterDecision(BaseModel):
    """The arbiter's ruling after seeing critiques."""
    consensus_score: float = Field(ge=0.0, le=1.0)
    accepted_points: list[str]             # critique points the arbiter agrees with
    rejected_points: list[str]             # points the arbiter dismissed
    directive: str                         # "finalize" | "revise" | "escalate"
    revised_proposal: Optional[str] = None  # if directive == "revise"
    reasoning: str


class FinalAnswer(BaseModel):
    """The fully validated output delivered to the user."""
    task: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    consensus_score: float
    debate_rounds: int
    sources: list[str]
    dissenting_points: list[str]           # unresolved minority critiques, logged for transparency
    models_used: dict[str, str]            # role → model name
