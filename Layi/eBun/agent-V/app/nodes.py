"""
LangGraph node functions. Each takes the TicketState and returns a
partial dict of updates -- this is standard LangGraph node shape.
"""
from __future__ import annotations
from app.state import TicketState, TriageResult, GuardrailVerdict
from app.llm import LLMClient
from app.tools import giftcard_tools, kb_tools, ticketing_tools
from app import guardrails

llm = LLMClient()

TRIAGE_SYSTEM_PROMPT = """You are a triage classifier for a gift card app's customer support system.
Classify the customer's message into exactly one category, assess urgency and sentiment,
and flag whether resolving this will require touching money or PII."""

DRAFT_SYSTEM_PROMPT = """You are a customer support agent for a gift card app.
Write a short, warm, direct reply to the customer using the provided policy context and
tool results. Do not invent balances, amounts, or policies not present in the context."""


def triage_node(state: TicketState) -> dict:
    triage = llm.structured(
        system=TRIAGE_SYSTEM_PROMPT,
        user=state["customer_message"],
        schema=TriageResult,
    )
    return {"triage": triage}


def retrieve_node(state: TicketState) -> dict:
    docs = kb_tools.search_kb(state["customer_message"])
    return {"kb_context": [f"{d['title']}: {d['text']}" for d in docs]}


def resolve_node(state: TicketState) -> dict:
    """
    Maps the triage category to a concrete tool call where one applies,
    then drafts a customer-facing reply grounded in the KB context and
    the tool result. For categories with no direct action (general FAQ),
    it just drafts a reply from KB context.

    NOTE: this uses a simple category->tool mapping rather than a full
    tool-calling agent loop for clarity and determinism. To make this a
    true agent that chooses its own tools, bind giftcard_tools functions
    to the LLM client via function calling and let it decide -- trade-off
    is less predictability, so most teams start deterministic like this
    and loosen it once they trust the model's tool-selection accuracy.
    """
    category = state["triage"].category
    customer_id = state["customer_id"]
    proposed_action = None
    action_result = None

    if category == "balance_inquiry":
        card_code = _extract_card_code(state["customer_message"]) or "GC-1001"
        action_result = giftcard_tools.check_balance(card_code)
        proposed_action = "check_balance"

    elif category == "redemption_failed":
        card_code = _extract_card_code(state["customer_message"]) or "GC-1002"
        action_result = giftcard_tools.check_redemption_status(card_code)
        proposed_action = "check_redemption_status"

    elif category == "card_not_delivered":
        card_code = _extract_card_code(state["customer_message"]) or "GC-1001"
        proposed_action = "reissue_card"
        action_result = giftcard_tools.reissue_card(card_code, reason="reported not delivered")

    elif category == "refund_request":
        proposed_action = "issue_refund"
        # amount would normally come from transaction lookup / the ticket itself
        amount = _extract_refund_amount(state["customer_message"]) or 25.00
        action_result = {"proposed_amount": amount}

    context = "\n".join(state.get("kb_context", []))
    tool_summary = f"Tool result: {action_result}" if action_result else "No tool action taken."
    draft = llm.complete(
        system=DRAFT_SYSTEM_PROMPT,
        user=(
            f"Customer message: {state['customer_message']}\n\n"
            f"Policy/FAQ context:\n{context}\n\n"
            f"{tool_summary}"
        ),
    )

    return {
        "proposed_action": proposed_action,
        "action_result": str(action_result) if action_result else None,
        "draft_reply": draft,
    }


def guardrail_node(state: TicketState) -> dict:
    proposed_amount = None
    if state.get("proposed_action") == "issue_refund" and isinstance(state.get("action_result"), str):
        proposed_amount = _extract_refund_amount(state["action_result"])

    verdict = guardrails.evaluate(
        triage=state["triage"],
        proposed_action=state.get("proposed_action"),
        proposed_amount=proposed_amount,
        draft_reply=state.get("draft_reply"),
    )
    return {
        "guardrail": verdict,
        "next_step": "auto_reply" if verdict.allow_auto_reply else "escalate",
    }


def auto_reply_node(state: TicketState) -> dict:
    ticketing_tools.add_reply(state["ticket_id"], state["draft_reply"])
    ticketing_tools.close_ticket(state["ticket_id"])
    return {"final_reply": state["draft_reply"]}


def escalate_node(state: TicketState) -> dict:
    summary = (
        f"[ESCALATED] category={state['triage'].category} "
        f"urgency={state['triage'].urgency} sentiment={state['triage'].sentiment}\n"
        f"Reason: {state['guardrail'].reason}\n"
        f"Draft reply for reference: {state['draft_reply']}"
    )
    ticketing_tools.add_internal_note(state["ticket_id"], summary)
    ticketing_tools.assign_to_human(state["ticket_id"], team_or_agent_id="customer_care_team")
    return {"escalation_summary": summary}


def route_after_guardrail(state: TicketState) -> str:
    return state["next_step"]


# --- tiny helpers, replace with real parsing / structured extraction -------

def _extract_card_code(text: str) -> str | None:
    import re
    m = re.search(r"GC-\d{4}", text.upper())
    return m.group(0) if m else None


def _extract_refund_amount(text: str) -> float | None:
    import re
    m = re.search(r"\$?(\d+(?:\.\d{1,2})?)", text)
    return float(m.group(1)) if m else None
