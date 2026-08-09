"""
Shared state that flows through every node in the graph.

This is intentionally a plain TypedDict (not a Pydantic model) because
LangGraph merges partial updates into this dict at every node -- keeping
it simple avoids fighting the framework. Anything that needs strict
validation (the triage output, the guardrail verdict) uses a Pydantic
model *inside* one of these fields instead.
"""
from __future__ import annotations
from typing import TypedDict, Optional, Literal, List
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured sub-objects (validated with Pydantic, stored inside the state)
# ---------------------------------------------------------------------------

IssueCategory = Literal[
    "card_not_delivered",
    "redemption_failed",
    "balance_inquiry",
    "refund_request",
    "fraud_or_stolen",
    "account_access",
    "general_faq",
    "other",
]


class TriageResult(BaseModel):
    category: IssueCategory
    urgency: Literal["low", "medium", "high"]
    sentiment: Literal["neutral", "frustrated", "angry"]
    short_summary: str = Field(..., description="One-sentence summary for a human reviewer.")
    requires_pii_or_money_action: bool = Field(
        ..., description="True if resolving this requires touching balance, refunds, or account data."
    )


class GuardrailVerdict(BaseModel):
    allow_auto_reply: bool
    reason: str
    triggered_rules: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class TicketState(TypedDict, total=False):
    # inputs
    ticket_id: str
    customer_id: str
    customer_message: str
    conversation_history: List[str]

    # produced by nodes, in pipeline order
    triage: TriageResult
    kb_context: List[str]
    draft_reply: str
    proposed_action: Optional[str]        # e.g. "reissue_card", "check_balance" -- None if just a reply
    action_result: Optional[str]
    guardrail: GuardrailVerdict

    # routing / outcome
    next_step: Literal["auto_reply", "escalate"]
    final_reply: Optional[str]
    escalation_summary: Optional[str]
