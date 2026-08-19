# Gift Card Customer Care Agent

A LangGraph pipeline that triages support tickets, retrieves policy context,
resolves low-risk issues automatically, and escalates everything else to a
human -- built entirely on open-source components.

```
new ticket -> triage -> retrieve (KB) -> resolve -> guardrail -> auto_reply | escalate
```

**Status: runs end-to-end right now with zero infrastructure.** Every
external dependency (LLM, ticketing system, knowledge base) has a mock
fallback, so you can run and evaluate the whole pipeline before connecting
anything real. Swap each one in independently as it becomes available.

## Quickstart (zero infra, mock mode)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m app.eval.run_eval app/eval/golden_set_template.csv
```

You should see 8/8 tickets classified and routed correctly against the
sample golden set. This is the same command you'll run against your real
historical tickets before going live.

## Run the API locally

```bash
uvicorn app.main:app --reload --port 8080

curl -X POST http://localhost:8080/webhook/ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "T-100", "customer_id": "cust_001", "customer_message": "My card GC-1001 never arrived"}'
```

## Project structure

```
app/
  state.py          # TicketState schema shared across all graph nodes
  llm.py            # model client -- MockLLM by default, or any OpenAI-compatible endpoint
  guardrails.py      # deterministic policy rules (NOT model calls)
  nodes.py           # the 6 graph nodes: triage, retrieve, resolve, guardrail, auto_reply, escalate
  graph.py            # wires the nodes into the LangGraph StateGraph
  main.py            # FastAPI webhook entrypoint
  tools/
    giftcard_tools.py    # gift card ops -- MOCK in-memory, replace with your real API calls
    ticketing_tools.py   # Chatwoot wrapper -- mock fallback if env vars unset
    kb_tools.py           # knowledge base search -- Qdrant or in-memory fallback
  eval/
    run_eval.py                # accuracy + "dangerous false auto-resolve" checker
    golden_set_template.csv    # replace with your real historical tickets
mcp_servers/
  giftcard_server.py        # exposes giftcard_tools.py over MCP
  knowledge_base_server.py  # exposes kb search + policy docs over MCP
docker-compose.yml   # Postgres (checkpointing) + Qdrant (knowledge base)
```

## What's real vs. mocked right now

| Component | Status | To make it real |
|---|---|---|
| LangGraph pipeline & routing | **Real, tested** | Nothing -- this is the actual logic |
| Guardrail rules | **Real, tested** | Tune the constants in `guardrails.py` to your policy |
| Eval harness | **Real, tested** | Replace `golden_set_template.csv` with your real tickets |
| Triage / drafting | Mock (`MockLLM`) | Set `LLM_BASE_URL` in `.env` (vLLM, Ollama, or OpenRouter) |
| Gift card ops | Mock (in-memory) | Replace function bodies in `giftcard_tools.py` with real API calls |
| Ticketing | Mock (prints actions) | Set `CHATWOOT_*` env vars once your Chatwoot instance exists |
| Knowledge base | Mock (4 sample docs) | Set `QDRANT_URL` and index your real policy docs |
| Checkpointing | In-memory (`MemorySaver`) | Swap to `PostgresSaver` in `graph.py` before production |

Nothing about the pipeline's structure changes as you swap these in --
that's deliberate. The function signatures in `tools/` are the contract;
only their internals change.

## Wiring in a real model (open source path)

```bash
# Option A: self-hosted, needs a GPU
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest --model Qwen/Qwen2.5-72B-Instruct

# Option B: no GPU needed, hosted open-weight model
# get a key from openrouter.ai, then in .env:
#   LLM_BASE_URL=https://openrouter.ai/api/v1
#   LLM_API_KEY=sk-or-...
#   LLM_MODEL=qwen/qwen-2.5-72b-instruct
```

## Adding channels: WhatsApp, Email, Calls (via Chatwoot)

Chatwoot normalizes every channel into the same `message_created` webhook event, so `app/nodes.py`, `app/guardrails.py`, and the rest of the pipeline don't change at all for this -- the only new code is `app/chatwoot_webhook.py`, an adapter that parses Chatwoot's real event shape and filters out anything that isn't a genuine new incoming customer message (critically, this includes **our own replies**, which Chatwoot also fires webhooks for -- without this filter the agent would try to process its own responses).

### 1. Point Chatwoot at the real webhook endpoint
In Chatwoot: **Settings -> Integrations -> Webhooks -> Add new webhook**
- URL: `https://your-host/webhooks/chatwoot`
- Subscribe to at least `message_created`
- Copy the secret shown once at creation into `.env`:
  ```dotenv
  CHATWOOT_WEBHOOK_SECRET=the-secret-shown-when-you-created-the-webhook
  ```
  Without this set, signature verification is skipped (fine for local testing, not for a public URL) -- the startup banner will tell you plainly whether it's on.

### 2. WhatsApp
**Settings -> Inboxes -> Add Inbox -> WhatsApp.** Two options:
- **Embedded Signup** (recommended): log in with Facebook, pick or create a WhatsApp Business Account, add your number -- Chatwoot handles webhook/token config automatically. Fastest path, no manual Meta Developer Console work.
- **Manual Setup**: only needed if you're a tech provider onboarding your own number, or already have infra in the Meta Developer Console. Requires generating tokens, creating a system user, and configuring the webhook by hand.

Once connected, WhatsApp messages arrive as ordinary conversations and flow through `/webhooks/chatwoot` exactly like any other channel. One WhatsApp-specific thing worth knowing: outbound messages sent more than 24 hours after the customer's last message require an approved message **template**, not free-form text -- if `auto_reply_node` tries to reply to an old WhatsApp conversation, that reply may fail or need to go through a template instead. Not currently handled specially in the code; flagging as a known gap if you see WhatsApp auto-replies silently fail on older conversations.

### 3. Email
**Settings -> Inboxes -> Add Inbox -> Email.** Either use a Chatwoot-provided address or connect your own domain via IMAP/SMTP forwarding. No code changes needed beyond what's already built -- email tickets flow through the identical pipeline.

### 4. Calls -- read this before promising it to anyone
Chatwoot's current voice support is specifically **WhatsApp Calling**: a live call that rings in a human agent's browser (Settings -> Inboxes -> Add Inbox -> WhatsApp Call, requires the number to be enrolled in Meta's WhatsApp Business Calling API). This is fundamentally different from everything else in this list: **there is no async text message for the pipeline to triage while a call is ringing**, so the AI agent cannot answer or triage a live call the way it handles WhatsApp/Email/web tickets. Calls should be expected to route straight to a human, same as today, with no AI involvement in the call itself. If Chatwoot later exposes a post-call transcript or summary as a normal conversation message, that could flow through the existing pipeline unchanged -- but that's not confirmed to exist today, so don't scope it as done until we've verified it against a real call.

### Testing the new endpoint before touching real channels
```bash
uvicorn app.main:app --reload --port 8080
```
Send a simulated Chatwoot payload directly, without needing WhatsApp/Email connected yet:
```bash
curl -X POST http://localhost:8080/webhooks/chatwoot \
  -H "Content-Type: application/json" \
  -d '{
    "event": "message_created",
    "content": "My card GC-1001 never arrived",
    "message_type": "incoming",
    "conversation": {"id": "TEST-1", "channel": "Channel::Whatsapp", "contact": {"id": 42}},
    "contact": {"id": 42},
    "account": {"id": 1}
  }'
```
If `CHATWOOT_WEBHOOK_SECRET` is set, this direct curl won't have a valid signature and will correctly 401 -- that's the verification working, not a bug. Temporarily unset it for this kind of manual testing, or compute a matching signature (see `app/chatwoot_webhook.py::verify_signature` for the exact HMAC construction).

## Before this touches a real customer

1. Export your historical tickets into the `golden_set_template.csv` format,
   with the category and the outcome (auto_reply/escalate) a human would have
   chosen.
2. Run `python -m app.eval.run_eval your_real_tickets.csv` and drive
   **"dangerous false auto-resolves" to zero first** -- that's the failure
   mode that actually costs you money and trust. Category accuracy matters
   less than outcome accuracy.
3. Tune `guardrails.py`'s `REFUND_AUTO_APPROVE_CEILING` and
   `NEVER_AUTO_RESOLVE_CATEGORIES` to your actual policy.
4. Launch in shadow mode: change `auto_reply_node` to post as an internal
   note instead of a customer-facing reply, and require human approval
   before it sends, for at least a couple of weeks.
5. Only then let `auto_reply` fire live, starting with the lowest-risk
   category (`balance_inquiry`) and expanding one category at a time.

## Production checkpointing (Postgres)

```bash
docker compose up -d postgres
pip install langgraph-checkpoint-postgres "psycopg[binary]"
```

```python
# in graph.py
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
graph = build_graph(checkpointer=checkpointer)
```

This is what makes an escalated ticket's state survive a server restart
while it's sitting in a human's queue -- without it, `MemorySaver` loses
everything on restart.
