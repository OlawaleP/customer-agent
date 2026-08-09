"""
Webhook entrypoint plus read/status endpoints.

Two ways tickets get in:
  POST /webhook/ticket      simple manual payload, for direct testing (Postman, curl)
  POST /webhooks/chatwoot   REAL Chatwoot webhook target -- point WhatsApp, Email,
                            and web-widget conversations here via Settings ->
                            Integrations -> Webhooks, subscribed to "message_created".
                            Channel-agnostic: Chatwoot normalizes every channel into
                            the same event shape before it reaches this endpoint.

Run locally:
    uvicorn app.main:app --reload --port 8080

Endpoints:
    POST /webhook/ticket              submit a ticket directly (mode: "live" or "shadow")
    POST /webhooks/chatwoot           real Chatwoot webhook target (see above)
    POST /tickets/{id}/approve        resume a shadow-mode ticket pending approval
    GET  /tickets/{id}                latest state of a ticket
    GET  /tickets/{id}/history        full checkpoint history for a ticket
    GET  /tickets                     list recent tickets (?limit=&offset=)
    GET  /stats/summary               aggregate outcome/category counts
    GET  /guardrail-rules             current guardrail thresholds
    POST /uploads/moderate-image      check a platform image upload BEFORE it's
                                       accepted -- unrelated to the ticket pipeline
                                       above; see app/tools/moderation_tools.py
    GET  /healthz                     liveness check

NOTE ON AUTH: none of these endpoints currently require authentication.
That's a deliberate short-term choice for local development, not a
recommendation -- /tickets/{id} and /tickets both return real customer
messages and drafted replies. Add an API key check before exposing this
outside your own machine. /webhooks/chatwoot has its own separate
protection via HMAC signature verification (CHATWOOT_WEBHOOK_SECRET) --
set that before pointing a real Chatwoot instance at a public URL.
"""
from __future__ import annotations
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal, Optional
from dotenv import load_dotenv
import random
from .llm import groq_client # Import the Groq instance

# Load .env from the project root explicitly (not just cwd), so this works
# regardless of the directory you run `uvicorn` from.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from app.graph import build_graph, build_shadow_graph, close_checkpointer_pool
from app import ticket_queries
from app import guardrails as guardrails_module
from app import chatwoot_webhook
from app.tools import moderation_tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_checkpointer_pool()  # cleanly stop pool threads instead of hanging on exit


app = FastAPI(title="Gift Card Customer Care Agent", lifespan=lifespan)
graph = build_graph()
shadow_graph = build_shadow_graph()  # same checkpointer/pool, pauses before auto_reply/escalate


def _startup_diagnostics():
    """Prints exactly which subsystems are live vs mocked -- this is the
    single most common source of confusion (things silently fall back to
    mock with zero error if a credential didn't load), so make it loud."""
    from app.llm import CODE_VERSION
    lines = ["", "=" * 60, f"Gift Card Agent -- startup diagnostics (code version: {CODE_VERSION})", "=" * 60]

    llm_base = os.getenv("LLM_BASE_URL")
    lines.append(f"LLM:        {'LIVE -> ' + llm_base if llm_base else 'MOCK (MockLLM) -- LLM_BASE_URL not set'}")

    chatwoot = os.getenv("CHATWOOT_BASE_URL") and os.getenv("CHATWOOT_API_TOKEN") and os.getenv("CHATWOOT_ACCOUNT_ID")
    lines.append(f"Ticketing:  {'LIVE -> ' + os.getenv('CHATWOOT_BASE_URL') if chatwoot else 'MOCK -- CHATWOOT_* env vars not fully set'}")

    qdrant = os.getenv("QDRANT_URL")
    lines.append(f"Knowledge base: {'LIVE -> ' + qdrant if qdrant else 'MOCK (4 sample docs) -- QDRANT_URL not set'}")

    db = os.getenv("DATABASE_URL")
    lines.append(f"Checkpointer:   {'Postgres (durable)' if db else 'In-memory (lost on restart) -- DATABASE_URL not set'}")
    if not db:
        lines.append("                NOTE: /tickets, /tickets/{id}, /stats/summary require")
        lines.append("                Postgres -- they will return an empty/note response on MemorySaver.")

    webhook_secret = os.getenv("CHATWOOT_WEBHOOK_SECRET")
    lines.append(f"Webhook auth:   {'Signature verification ON' if webhook_secret else 'OFF -- CHATWOOT_WEBHOOK_SECRET not set (fine locally, NOT for a public URL)'}")

    openai_key = os.getenv("OPENAI_API_KEY")
    lines.append(f"Moderation:     {'LIVE (OpenAI omni-moderation-latest)' if openai_key else 'MOCK -- OPENAI_API_KEY not set (this is NOT your Groq key)'}")
    mod_conv = os.getenv("CHATWOOT_MODERATION_CONVERSATION_ID")
    lines.append(f"Mod escalation: {'LIVE -> conversation ' + mod_conv if (chatwoot and mod_conv) else 'MOCK -- CHATWOOT_MODERATION_CONVERSATION_ID not set'}")

    lines.append("=" * 60)
    lines.append("If something above says MOCK and shouldn't: check .env is in the")
    lines.append("project root, and that you ran `pip install -r requirements.txt`")
    lines.append("after pulling the latest code (python-dotenv must be installed).")
    lines.append("No auth on any endpoint yet -- see module docstring.")
    lines.append("=" * 60 + "\n")
    print("\n".join(lines))


_startup_diagnostics()


class TicketWebhookPayload(BaseModel):
    ticket_id: str
    customer_id: str
    customer_message: str
    mode: Literal["live", "shadow"] = "live"


def _format_result(ticket_id: str, result: dict, pending: bool = False) -> dict:
    triage = result.get("triage")
    return {
        "ticket_id": ticket_id,
        "status": "pending_approval" if pending else "completed",
        "outcome": result.get("next_step"),
        "category": triage.category if triage else None,
        "final_reply": result.get("final_reply"),
        "escalation_summary": result.get("escalation_summary"),
    }


def _run_ticket(ticket_id: str, customer_id: str, customer_message: str, mode: str = "live") -> dict:
    """Shared by both /webhook/ticket (manual/testing, simple payload) and
    /webhooks/chatwoot (real Chatwoot events, adapted first). Nothing about
    the pipeline itself changes based on which endpoint called it or which
    channel the ticket came from -- that's deliberate."""
    config = {"configurable": {"thread_id": ticket_id}}
    active_graph = shadow_graph if mode == "shadow" else graph

    result = active_graph.invoke(
        {"ticket_id": ticket_id, "customer_id": customer_id, "customer_message": customer_message},
        config=config,
    )

    if mode == "shadow":
        snapshot = active_graph.get_state(config)
        if snapshot.next:
            return _format_result(ticket_id, result, pending=True)

    return _format_result(ticket_id, result)


@app.post("/webhook/ticket")
async def handle_new_ticket(payload: TicketWebhookPayload):
    return _run_ticket(payload.ticket_id, payload.customer_id, payload.customer_message, payload.mode)


@app.post("/webhooks/chatwoot")
async def handle_chatwoot_webhook(request: Request):
    """
    Real Chatwoot webhook endpoint -- point Settings -> Integrations ->
    Webhooks at this URL, subscribed to at least "message_created".
    This is what actually receives WhatsApp, Email, and web-widget
    tickets alike, since Chatwoot normalizes all of them into the same
    event shape before they reach us.

    Always returns 200 quickly, including for events we intentionally
    ignore (outgoing messages, non-message events) -- Chatwoot expects a
    fast ack and doesn't need to know we skipped something.
    """
    raw_body = await request.body()
    signature = request.headers.get("X-Chatwoot-Signature")
    timestamp = request.headers.get("X-Chatwoot-Timestamp")

    is_valid, reason = chatwoot_webhook.verify_signature(raw_body, signature, timestamp)
    if not is_valid:
        print(f"[chatwoot_webhook] REJECTED: {reason}")
        raise HTTPException(status_code=401, detail=f"Webhook signature verification failed: {reason}")

    payload = await request.json()
    parsed = chatwoot_webhook.parse_incoming_ticket(payload)

    if parsed is None:
        # Not a genuine new incoming customer message -- could be our own
        # reply, an agent's reply, a non-text message, or an event type
        # we don't act on. Ack it and move on; this is expected traffic,
        # not an error.
        return {"status": "ignored", "event": payload.get("event"), "message_type": payload.get("message_type")}

    print(f"[chatwoot_webhook] Processing ticket {parsed['ticket_id']} from channel={parsed['channel']}")
    result = _run_ticket(parsed["ticket_id"], parsed["customer_id"], parsed["customer_message"])
    return result


@app.post("/tickets/{ticket_id}/approve")
async def approve_ticket(ticket_id: str):
    """Resume a shadow-mode ticket that's paused awaiting human approval.
    Only meaningful for tickets originally submitted with mode="shadow" --
    a "live" ticket has nothing pending and this will 400."""
    config = {"configurable": {"thread_id": ticket_id}}
    snapshot = shadow_graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"No ticket found with id {ticket_id}")
    if not snapshot.next:
        raise HTTPException(
            status_code=400,
            detail=f"Ticket {ticket_id} has no pending approval (already completed, or was "
                   f"never submitted in shadow mode).",
        )

    result = shadow_graph.invoke(None, config=config)  # resumes from the interrupt
    return _format_result(ticket_id, result)


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    config = {"configurable": {"thread_id": ticket_id}}
    snapshot = graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"No ticket found with id {ticket_id}")

    triage = snapshot.values.get("triage")
    guardrail = snapshot.values.get("guardrail")
    return {
        "ticket_id": ticket_id,
        "customer_id": snapshot.values.get("customer_id"),
        "customer_message": snapshot.values.get("customer_message"),
        "triage": triage.model_dump() if triage else None,
        "guardrail": guardrail.model_dump() if guardrail else None,
        "outcome": snapshot.values.get("next_step"),
        "final_reply": snapshot.values.get("final_reply"),
        "escalation_summary": snapshot.values.get("escalation_summary"),
        "pending_approval": bool(snapshot.next),
        "next_step_in_graph": list(snapshot.next) if snapshot.next else None,
    }


@app.get("/tickets/{ticket_id}/history")
async def get_ticket_history(ticket_id: str):
    config = {"configurable": {"thread_id": ticket_id}}
    history = []
    for snapshot in graph.get_state_history(config):
        history.append({
            "step": snapshot.metadata.get("step") if snapshot.metadata else None,
            "created_at": snapshot.created_at,
            "next_node": list(snapshot.next) if snapshot.next else None,
            "state_at_this_point": {
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in snapshot.values.items()
            },
        })

    if not history:
        raise HTTPException(status_code=404, detail=f"No ticket found with id {ticket_id}")

    return {"ticket_id": ticket_id, "checkpoint_count": len(history), "history": history}


@app.get("/tickets")
async def list_tickets(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return ticket_queries.list_tickets(limit=limit, offset=offset)


@app.get("/stats/summary")
async def stats_summary():
    return ticket_queries.get_stats_summary()


@app.get("/guardrail-rules")
async def guardrail_rules():
    """Read-only view of current policy thresholds -- for transparency,
    not just for us; useful to show anyone asking 'what are the actual rules'."""
    return {
        "refund_auto_approve_ceiling": guardrails_module.REFUND_AUTO_APPROVE_CEILING,
        "never_auto_resolve_categories": sorted(guardrails_module.NEVER_AUTO_RESOLVE_CATEGORIES),
        "escalate_on_sentiment": sorted(guardrails_module.ESCALATE_ON_SENTIMENT),
        "min_reply_length": guardrails_module.MIN_REPLY_LENGTH,
    }


@app.post("/uploads/moderate-image")
async def moderate_image(background_tasks: BackgroundTasks, uploader_id: str, image: UploadFile = File(...)):
    """
    Call this BEFORE accepting a user's image upload on the platform --
    unrelated to the ticket pipeline above. Synchronous by design: the
    caller is a user waiting to know if their upload is accepted, so this
    holds the response open through the moderation check itself. Only the
    escalation side-effect (notifying a human on a flag) is backgrounded,
    since the caller doesn't need to wait on that part.

    Deliberately fail-closed: if moderation_tools.check_image() raises
    (e.g. OPENAI_API_KEY missing), that propagates as a 500 rather than
    being caught and treated as "approved" -- a moderation gate that fails
    open on a config error is worse than one that fails loudly.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    content_type = image.content_type or "image/jpeg"

    result = moderation_tools.check_image(image_bytes, content_type)

    if not result.flagged:
        return {"approved": True, "status": "approved"}

    background_tasks.add_task(
        moderation_tools.escalate_flagged_upload,
        image_bytes, image.filename or "upload.jpg", content_type, uploader_id, result,
    )
    return {
        "approved": False,
        "status": "pending_review",
        "message": "This content needs to be reviewed before being uploaded.",
    }

    # Define the expected JSON payload
class GenerateMessageRequest(BaseModel):
    category: str
    tone: Optional[str] = None

@app.post("/api/v1/messages/generate")
async def generate_random_message(request: GenerateMessageRequest):
    # Pool of tones used when the client does not send a tone
    tones = ["Warm", "Short", "Funny", "Happy", "Lovely", "Heartfelt", "Sweet", "Appreciative"]
    
    # Use the tone from the payload if provided, otherwise pick randomly
    selected_tone = request.tone if request.tone else random.choice(tones)
    
    try:
        system_prompt = (
            f"You are an expert card writer. Generate a single, completely random short message "
            f"for a '{request.category}' gift card. The tone of the message must be {selected_tone}. "
            "Keep it strictly under 2 sentences. Do not include quotes, placeholders, or introductory text. "
            "Return only the final message text."
        )
        
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=60
        )
        
        generated_text = chat_completion.choices[0].message.content.strip()
        
        return {
            "category": request.category,
            "applied_tone": selected_tone,
            "message": generated_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PolishRequest(BaseModel):
    original_message: str
    target_tone: str
    recipient_name: str = ""

@app.post("/api/v1/messages/polish")
async def polish_message(request: PolishRequest):
    try:
        # Define strict instructions using the system role
        system_prompt = (
            f"You are an expert copywriter. Rewrite the following message to reflect a '{request.target_tone}' tone. "
            f"If a recipient name ({request.recipient_name}) is provided, naturally incorporate it into the text. "
            "Return ONLY the rewritten message text. Do not include quotes, conversational filler, or introductory phrases."
        )
        
        # Call the Groq chat completions API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.original_message}
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.7,
            max_tokens=150
        )
        
        # Extract the polished text from the model's response
        polished_text = chat_completion.choices[0].message.content.strip()
        
        return {"polished_message": polished_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
