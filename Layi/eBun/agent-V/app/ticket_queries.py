"""
Reads LangGraph's checkpoint tables directly for ticket listing and stats,
rather than adding a dedicated app-owned table.

Trade-off, made deliberately: this is faster to ship but couples us to
LangGraph's internal schema, which is NOT a public API contract and can
change on a LangGraph upgrade. If that ever bites (a migration changes
column names, etc.), the fix is to add a dedicated `tickets` table that
auto_reply_node/escalate_node write to directly -- decoupled from
checkpoint internals entirely. Flagging this now so it's a known,
intentional trade-off rather than a surprise later.

Schema this relies on (confirmed against langgraph-checkpoint-postgres
via BasePostgresSaver.MIGRATIONS at build time):
  checkpoints(thread_id, checkpoint_ns, checkpoint_id, checkpoint JSONB, metadata JSONB)
    -- checkpoint->'channel_values' holds PRIMITIVE state fields inline
    -- (str/int/float/bool/None), e.g. customer_id, next_step, final_reply.
  checkpoint_blobs(thread_id, checkpoint_ns, channel, version, type, blob BYTEA)
    -- non-primitive fields (triage, guardrail, kb_context, conversation_history)
    -- live here instead, serialized -- must go through the checkpointer's
    -- own serde to decode, not read as plain JSON.
"""
from __future__ import annotations
from typing import Optional
from app.graph import get_checkpoint_pool


def _serde():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return JsonPlusSerializer()


def list_tickets(limit: int = 20, offset: int = 0) -> dict:
    """
    Latest state per ticket (thread), most recently updated first.
    Only reads primitive fields -- fast, no blob decoding.
    """
    pool = get_checkpoint_pool()
    if pool is None:
        return {"tickets": [], "note": "No DATABASE_URL configured -- using MemorySaver, "
                                        "which cannot be listed/queried this way."}

    query = """
        SELECT thread_id, ts, customer_id, next_step, final_reply, escalation_summary
        FROM (
            SELECT DISTINCT ON (thread_id)
                thread_id,
                checkpoint->>'ts' AS ts,
                checkpoint->'channel_values'->>'customer_id' AS customer_id,
                checkpoint->'channel_values'->>'next_step' AS next_step,
                checkpoint->'channel_values'->>'final_reply' AS final_reply,
                checkpoint->'channel_values'->>'escalation_summary' AS escalation_summary
            FROM checkpoints
            WHERE checkpoint_ns = ''
            ORDER BY thread_id, checkpoint->>'ts' DESC
        ) latest
        ORDER BY ts DESC
        LIMIT %s OFFSET %s
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit, offset))
            rows = cur.fetchall()

    tickets = [
        {
            "ticket_id": r["thread_id"],
            "updated_at": r["ts"],
            "customer_id": r["customer_id"],
            "outcome": r["next_step"],  # None if still mid-pipeline / never completed
            "final_reply": r["final_reply"],
            "escalation_summary": r["escalation_summary"],
        }
        for r in rows
    ]
    return {"tickets": tickets, "limit": limit, "offset": offset}


def get_stats_summary() -> dict:
    """
    Aggregate counts across all tickets: outcome breakdown (cheap, primitive
    field) and category breakdown (requires decoding the `triage` blob,
    since TriageResult is a non-primitive value).
    """
    pool = get_checkpoint_pool()
    if pool is None:
        return {"note": "No DATABASE_URL configured -- using MemorySaver, cannot aggregate stats."}

    # --- outcome counts: cheap, primitive field, pure SQL ---
    outcome_query = """
        SELECT next_step, COUNT(*) AS n
        FROM (
            SELECT DISTINCT ON (thread_id)
                thread_id,
                checkpoint->'channel_values'->>'next_step' AS next_step
            FROM checkpoints
            WHERE checkpoint_ns = ''
            ORDER BY thread_id, checkpoint->>'ts' DESC
        ) latest
        GROUP BY next_step
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(outcome_query)
            outcome_rows = cur.fetchall()

    outcome_counts = {(r["next_step"] or "in_progress"): r["n"] for r in outcome_rows}

    # --- category counts: needs the triage blob decoded per-thread ---
    blob_query = """
        SELECT DISTINCT ON (thread_id) thread_id, type, blob
        FROM checkpoint_blobs
        WHERE checkpoint_ns = '' AND channel = 'triage' AND blob IS NOT NULL
        ORDER BY thread_id, version DESC
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(blob_query)
            blob_rows = cur.fetchall()

    serde = _serde()
    category_counts: dict[str, int] = {}
    decode_failures = 0
    for r in blob_rows:
        try:
            triage_obj = serde.loads_typed((r["type"], bytes(r["blob"])))
            category = getattr(triage_obj, "category", None) or (
                triage_obj.get("category") if isinstance(triage_obj, dict) else None
            )
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
        except Exception:
            decode_failures += 1

    result = {
        "total_tickets": sum(outcome_counts.values()),
        "by_outcome": outcome_counts,
        "by_category": category_counts,
    }
    if decode_failures:
        result["category_decode_failures"] = decode_failures
    return result
