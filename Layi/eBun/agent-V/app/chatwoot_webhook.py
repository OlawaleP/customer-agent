"""
Adapts real Chatwoot webhook events into the internal shape our LangGraph
pipeline expects. This is the piece that makes WhatsApp, Email, and the
web widget all "just work" the same way -- Chatwoot normalizes every
channel into the same message_created event before it ever reaches us,
so nothing in app/nodes.py or app/guardrails.py needs to know or care
which channel a ticket came from.

Two things this module exists specifically to get right:

1. LOOP PREVENTION. Chatwoot fires message_created for every message in
   a conversation, including the reply our own auto_reply_node posts.
   Without filtering on message_type == "incoming", the agent would
   receive a webhook for its own reply and could attempt to process it
   again. Every path through this module drops anything that isn't a
   genuine incoming customer message before it reaches the graph.

2. SIGNATURE VERIFICATION. Chatwoot signs webhook payloads so a receiver
   can confirm a request actually came from Chatwoot and wasn't replayed
   or forged. Verification is skipped (with a loud warning) if no secret
   is configured -- fine for local testing, not for anything reachable
   from the internet.
"""
from __future__ import annotations
import os
import hmac
import hashlib
import time
from typing import Optional

WEBHOOK_SECRET = os.getenv("CHATWOOT_WEBHOOK_SECRET")
# Reject webhook requests whose timestamp is older than this many seconds,
# to blunt replay attacks even if a signature is somehow captured.
MAX_TIMESTAMP_AGE_SECONDS = 300


def verify_signature(raw_body: bytes, signature_header: Optional[str], timestamp_header: Optional[str]) -> tuple[bool, str]:
    """
    Verifies X-Chatwoot-Signature against HMAC-SHA256 of
    "{timestamp}.{raw_body}" using CHATWOOT_WEBHOOK_SECRET.

    Returns (is_valid, reason). If no secret is configured, verification
    is skipped and this returns (True, "skipped -- no secret configured")
    so local testing isn't blocked -- but that's a real gap to close
    before this is reachable from the public internet.
    """
    if not WEBHOOK_SECRET:
        return True, "skipped -- CHATWOOT_WEBHOOK_SECRET not set (fine for local dev only)"

    if not signature_header or not timestamp_header:
        return False, "missing X-Chatwoot-Signature or X-Chatwoot-Timestamp header"

    try:
        ts = int(timestamp_header)
    except ValueError:
        return False, "malformed timestamp header"

    if abs(time.time() - ts) > MAX_TIMESTAMP_AGE_SECONDS:
        return False, f"timestamp outside allowed window ({MAX_TIMESTAMP_AGE_SECONDS}s) -- possible replay"

    signed_payload = f"{timestamp_header}.".encode() + raw_body
    expected = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), signed_payload, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, signature_header):
        return False, "signature mismatch"

    return True, "verified"


def extract_channel(payload: dict) -> str:
    """
    Best-effort channel label for logging/analytics. Chatwoot's payload
    shape for this has shifted across versions and event types (see
    chatwoot/chatwoot#13993), so this deliberately checks several
    possible locations rather than assuming one fixed schema, and
    falls back to "unknown" rather than raising.
    """
    conv = payload.get("conversation", {}) or {}
    for candidate in (
        conv.get("channel"),
        payload.get("channel"),
        (conv.get("meta", {}) or {}).get("channel"),
    ):
        if candidate:
            return candidate
    return "unknown"


def parse_incoming_ticket(payload: dict) -> Optional[dict]:
    """
    Returns a dict matching our internal {ticket_id, customer_id,
    customer_message} shape if this payload is a genuine new incoming
    customer message worth processing, or None if it should be ignored
    (wrong event type, outgoing/template message, missing data).
    """
    if payload.get("event") != "message_created":
        return None

    if payload.get("message_type") != "incoming":
        # This is the loop-prevention check. Outgoing = us (or a human
        # agent) replying; template = a WhatsApp template send. Neither
        # should re-enter the pipeline.
        return None

    content = (payload.get("content") or "").strip()
    if not content:
        # Non-text messages (attachments only, reactions, etc.) have no
        # text for triage to work with -- skip rather than crash.
        return None

    conversation = payload.get("conversation", {}) or {}
    conversation_id = conversation.get("id") or payload.get("conversation_id")
    if conversation_id is None:
        return None

    contact = payload.get("contact") or conversation.get("contact") or {}
    contact_id = contact.get("id")

    return {
        "ticket_id": str(conversation_id),
        "customer_id": str(contact_id) if contact_id is not None else f"unknown-{conversation_id}",
        "customer_message": content,
        "channel": extract_channel(payload),
    }
