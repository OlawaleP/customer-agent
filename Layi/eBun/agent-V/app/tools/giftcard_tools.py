"""
Gift card backend operations.

This is a MOCK in-memory implementation so the whole system is runnable
today. Replace the body of each function with a real call to your gift
card platform's API -- the function signatures, docstrings, and return
shapes are the actual contract the agent reasons over, so keep those
stable even after you wire in the real backend.

Each function here is also exposed as an MCP tool in mcp_servers/giftcard_server.py
so any MCP-compatible framework (not just this LangGraph app) can use it.
"""
from __future__ import annotations
from datetime import datetime, timedelta
import random

# --- mock "database" -------------------------------------------------------
_MOCK_CARDS = {
    "GC-1001": {"customer_id": "cust_001", "balance": 25.00, "status": "active",
                "issued_at": "2026-06-01", "redeemed_amount": 0.0},
    "GC-1002": {"customer_id": "cust_002", "balance": 0.00, "status": "fully_redeemed",
                "issued_at": "2026-05-15", "redeemed_amount": 50.00},
    "GC-1003": {"customer_id": "cust_003", "balance": 100.00, "status": "active",
                "issued_at": "2026-06-20", "redeemed_amount": 0.0},
}
_MOCK_TRANSACTIONS = {
    "cust_001": [{"date": "2026-06-01", "type": "purchase", "amount": 25.00, "card": "GC-1001"}],
    "cust_002": [
        {"date": "2026-05-15", "type": "purchase", "amount": 50.00, "card": "GC-1002"},
        {"date": "2026-05-16", "type": "redemption", "amount": -50.00, "card": "GC-1002"},
    ],
}


def check_balance(card_code: str) -> dict:
    """Look up the current balance and status of a gift card by its code."""
    card = _MOCK_CARDS.get(card_code)
    if not card:
        return {"found": False, "error": f"No card found with code {card_code}"}
    return {"found": True, "card_code": card_code, **card}


def check_redemption_status(card_code: str) -> dict:
    """Check whether a gift card has been successfully redeemed/activated."""
    card = _MOCK_CARDS.get(card_code)
    if not card:
        return {"found": False, "error": f"No card found with code {card_code}"}
    return {
        "found": True,
        "card_code": card_code,
        "status": card["status"],
        "redeemed_amount": card["redeemed_amount"],
    }


def reissue_card(card_code: str, reason: str) -> dict:
    """
    Reissue a gift card (e.g. lost/not delivered/redemption failed).
    Generates a new card code preserving the original balance; the old
    code is invalidated. `reason` is logged for audit purposes.
    """
    card = _MOCK_CARDS.get(card_code)
    if not card:
        return {"success": False, "error": f"No card found with code {card_code}"}
    new_code = f"GC-{random.randint(9000, 9999)}"
    _MOCK_CARDS[new_code] = {**card, "status": "active"}
    card["status"] = "invalidated"
    return {
        "success": True,
        "old_card_code": card_code,
        "new_card_code": new_code,
        "balance_transferred": card["balance"],
        "reason_logged": reason,
    }


def check_transaction_history(customer_id: str) -> dict:
    """Return recent transaction history for a customer (purchases, redemptions, refunds)."""
    return {"customer_id": customer_id, "transactions": _MOCK_TRANSACTIONS.get(customer_id, [])}


def issue_refund(customer_id: str, amount: float, reason: str) -> dict:
    """
    Issue a refund to a customer's original payment method.
    NOTE: production callers must pass this through the guardrail's
    refund-amount-ceiling check BEFORE calling this -- this function
    does not enforce policy limits itself, it only executes the action.
    """
    return {
        "success": True,
        "customer_id": customer_id,
        "amount_refunded": amount,
        "reason": reason,
        "refund_id": f"RF-{random.randint(100000, 999999)}",
        "processed_at": datetime.utcnow().isoformat(),
    }
