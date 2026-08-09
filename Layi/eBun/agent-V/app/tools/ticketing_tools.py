"""
Ticketing / helpdesk operations, wired for Chatwoot (open source).

Set these env vars to talk to a real Chatwoot instance:
  CHATWOOT_BASE_URL   e.g. https://your-chatwoot-host
  CHATWOOT_API_TOKEN  agent bot API access token
  CHATWOOT_ACCOUNT_ID your account id

If they're unset, every function falls back to printing what it WOULD
have done and returning a mock success -- this keeps local dev/demo
working without a live Chatwoot instance.

Swapping to Zammad/Freshdesk/Intercom later: only this file changes.
The graph and the rest of the tools don't know or care which helpdesk
is behind these functions -- that's the point of isolating I/O here.
"""
from __future__ import annotations
import os
import requests

BASE_URL = os.getenv("CHATWOOT_BASE_URL")
API_TOKEN = os.getenv("CHATWOOT_API_TOKEN")
ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")

_LIVE = bool(BASE_URL and API_TOKEN and ACCOUNT_ID)


def _headers():
    return {"api_access_token": API_TOKEN, "Content-Type": "application/json"}


def get_ticket(conversation_id: str) -> dict:
    """Fetch a ticket/conversation's details and message history."""
    if not _LIVE:
        return {"mock": True, "conversation_id": conversation_id, "status": "open",
                "messages": ["[mock] customer message would appear here"]}
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}"
    r = requests.get(url, headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def add_reply(conversation_id: str, message: str) -> dict:
    """Send a customer-facing reply on a ticket."""
    if not _LIVE:
        print(f"[MOCK add_reply] conversation={conversation_id} message={message!r}")
        return {"mock": True, "success": True}
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/messages"
    r = requests.post(url, headers=_headers(), json={"content": message, "message_type": "outgoing"}, timeout=10)
    r.raise_for_status()
    return r.json()


def add_internal_note(conversation_id: str, note: str) -> dict:
    """Add an internal note (visible only to human agents), e.g. an escalation summary."""
    if not _LIVE:
        print(f"[MOCK add_internal_note] conversation={conversation_id} note={note!r}")
        return {"mock": True, "success": True}
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/messages"
    r = requests.post(url, headers=_headers(), json={"content": note, "message_type": "outgoing", "private": True}, timeout=10)
    r.raise_for_status()
    return r.json()


def assign_to_human(conversation_id: str, team_or_agent_id: str) -> dict:
    """Assign a ticket to a human agent or team queue for escalation."""
    if not _LIVE:
        print(f"[MOCK assign_to_human] conversation={conversation_id} -> {team_or_agent_id}")
        return {"mock": True, "success": True}
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/assignments"
    r = requests.post(url, headers=_headers(), json={"assignee_id": team_or_agent_id}, timeout=10)
    r.raise_for_status()
    return r.json()


def close_ticket(conversation_id: str) -> dict:
    """Mark a ticket resolved/closed after auto-reply."""
    if not _LIVE:
        print(f"[MOCK close_ticket] conversation={conversation_id}")
        return {"mock": True, "success": True}
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/toggle_status"
    r = requests.post(url, headers=_headers(), json={"status": "resolved"}, timeout=10)
    r.raise_for_status()
    return r.json()
