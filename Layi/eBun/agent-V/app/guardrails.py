"""
Deterministic guardrails.

Deliberately NOT model calls. The whole point of a guardrail is that it
holds even when the LLM is wrong, hallucinating, or was prompt-injected
by something in a retrieved document or a customer message. Every rule
here is a plain Python check over structured fields, not a "please be
careful" instruction to a model.

Tune REFUND_AUTO_APPROVE_CEILING and FRAUD_CATEGORIES for your policy.
"""
from __future__ import annotations
import re
from app.state import TriageResult, GuardrailVerdict

REFUND_AUTO_APPROVE_CEILING = 50.00

NEVER_AUTO_RESOLVE_CATEGORIES = {"fraud_or_stolen", "account_access"}

ESCALATE_ON_SENTIMENT = {"angry"}

# A real customer reply is a sentence or two of prose. Model outputs that
# are suspiciously short, or that look like a classifier verdict rather
# than a reply (e.g. "User Safety: safe" -- seen in production from a
# free-tier auto-router landing on a moderation model instead of an
# assistant model), must never reach a customer unreviewed.
MIN_REPLY_LENGTH = 25
SUSPICIOUS_REPLY_PATTERNS = [
    r"^\s*(user\s+)?safety\s*:\s*(safe|unsafe)\s*$",
    r"^\s*(flag|label|category|verdict|classification)\s*:",
    r"^\s*(safe|unsafe|blocked|refused)\s*$",
]


def _reply_sanity_check(draft_reply: str | None) -> str | None:
    """Returns a failure reason string if the draft fails, else None."""
    if not draft_reply or len(draft_reply.strip()) < MIN_REPLY_LENGTH:
        return f"draft reply is missing or suspiciously short (<{MIN_REPLY_LENGTH} chars)"
    for pattern in SUSPICIOUS_REPLY_PATTERNS:
        if re.match(pattern, draft_reply.strip(), re.IGNORECASE):
            return f"draft reply looks like a classifier verdict, not a customer reply: {draft_reply!r}"
    return None


def evaluate(
    triage: TriageResult,
    proposed_action: str | None,
    proposed_amount: float | None = None,
    draft_reply: str | None = None,
) -> GuardrailVerdict:
    """
    Returns a verdict on whether the pipeline is allowed to auto-reply,
    or must escalate to a human. This is the single choke point every
    ticket passes through before anything is sent to a customer.
    """
    triggered: list[str] = []

    if triage.category in NEVER_AUTO_RESOLVE_CATEGORIES:
        triggered.append(f"category '{triage.category}' is never auto-resolved")

    if triage.sentiment in ESCALATE_ON_SENTIMENT:
        triggered.append("customer sentiment is angry")

    if triage.urgency == "high":
        triggered.append("triage marked this high urgency")

    if proposed_action == "issue_refund":
        if proposed_amount is None:
            triggered.append("refund action proposed with no amount -- cannot verify against ceiling")
        elif proposed_amount >= REFUND_AUTO_APPROVE_CEILING:
            triggered.append(
                f"refund amount ${proposed_amount:.2f} meets or exceeds "
                f"auto-approve ceiling of ${REFUND_AUTO_APPROVE_CEILING:.2f}"
            )

    if proposed_action == "reissue_card" and triage.category == "fraud_or_stolen":
        triggered.append("reissue on a fraud-flagged card requires human verification first")

    reply_issue = _reply_sanity_check(draft_reply)
    if reply_issue:
        triggered.append(reply_issue)

    allow = len(triggered) == 0
    reason = "No policy rules triggered; safe to auto-reply." if allow else "; ".join(triggered)
    return GuardrailVerdict(allow_auto_reply=allow, reason=reason, triggered_rules=triggered)
