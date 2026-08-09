"""
Assembles the pipeline shown in the architecture diagram:

  triage -> retrieve -> resolve -> guardrail -> [auto_reply | escalate]

Checkpointing: uses MemorySaver by default (fine for local dev/demo).
For production, swap to the Postgres checkpointer so an escalated
ticket's state survives a process restart while waiting on a human:

    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])

interrupt_before=["escalate"] pauses the graph right before escalation
executes, so a human-approval step could sit in front of it if you want
one even for the escalation itself (e.g. a lead reviews before it's
assigned). Remove it if you want escalation to fire immediately.
"""
from __future__ import annotations
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import TicketState
from app import nodes

# Held at module level so the pool survives for the app's lifetime -- letting
# this get garbage collected (as a previous version of this file did with a
# raw single connection) silently closes the underlying connection.
_pg_pool = None

# Memoized so multiple build_graph() calls (e.g. the live graph and the
# shadow-mode graph) share the SAME checkpointer/storage. Without this,
# _default_checkpointer() under MemorySaver would hand out a fresh, empty
# in-memory store on every call -- graph and shadow_graph would silently
# have two disconnected stores, and a ticket created via one would 404
# when looked up via the other. Postgres mode doesn't hit this (the pool
# is already a singleton), but it's memoized here too for consistency.
_checkpointer_singleton = None


def _default_checkpointer():
    """
    Uses Postgres if DATABASE_URL is set (so escalated tickets survive a
    restart), otherwise falls back to in-memory (fine for local dev/eval,
    state is lost on restart).

    Uses a connection pool rather than a single raw connection: a lone
    connection that drops (idle timeout, network blip, Postgres restart)
    has no way to recover and every subsequent request fails with
    "the connection is closed". A pool transparently reconnects.
    """
    global _pg_pool, _checkpointer_singleton
    if _checkpointer_singleton is not None:
        return _checkpointer_singleton

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        _checkpointer_singleton = MemorySaver()
        return _checkpointer_singleton
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool
        from psycopg.rows import dict_row

        if _pg_pool is None:
            _pg_pool = ConnectionPool(
                conninfo=database_url,
                max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                open=True,
            )
        _checkpointer_singleton = PostgresSaver(_pg_pool)
        _checkpointer_singleton.setup()  # creates checkpoint tables on first run, no-op after
        return _checkpointer_singleton
    except ImportError:
        print("DATABASE_URL is set but langgraph-checkpoint-postgres / psycopg / psycopg_pool "
              "aren't installed. Run: pip install langgraph-checkpoint-postgres 'psycopg[binary,pool]'. "
              "Falling back to MemorySaver for now.")
        _checkpointer_singleton = MemorySaver()
        return _checkpointer_singleton


def build_graph(checkpointer=None, interrupt_before=None):
    graph = StateGraph(TicketState)

    graph.add_node("triage", nodes.triage_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("resolve", nodes.resolve_node)
    graph.add_node("guardrail", nodes.guardrail_node)
    graph.add_node("auto_reply", nodes.auto_reply_node)
    graph.add_node("escalate", nodes.escalate_node)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "retrieve")
    graph.add_edge("retrieve", "resolve")
    graph.add_edge("resolve", "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        nodes.route_after_guardrail,
        {"auto_reply": "auto_reply", "escalate": "escalate"},
    )
    graph.add_edge("auto_reply", END)
    graph.add_edge("escalate", END)

    return graph.compile(
        checkpointer=checkpointer or _default_checkpointer(),
        interrupt_before=interrupt_before,
    )


def build_shadow_graph(checkpointer=None):
    """
    Shadow-mode graph: identical pipeline, but pauses right before the
    action that actually sends anything (auto_reply or escalate) so a
    human can approve first. Same checkpointer/pool as the live graph --
    this is a separate compiled graph object, not a separate database.
    """
    return build_graph(checkpointer=checkpointer, interrupt_before=["auto_reply", "escalate"])


def get_checkpoint_pool():
    """Expose the shared pool for read-only queries against checkpoint
    internals (ticket listing, stats). Returns None if running on
    MemorySaver (no DATABASE_URL set) -- callers must handle that."""
    global _pg_pool
    if _pg_pool is None:
        _default_checkpointer()  # ensures the pool is created if DATABASE_URL is set
    return _pg_pool


def close_checkpointer_pool():
    """Call on app shutdown so psycopg_pool's background threads exit cleanly
    instead of the 'couldn't stop thread' warning on process exit."""
    global _pg_pool, _checkpointer_singleton
    if _pg_pool is not None:
        _pg_pool.close()
        _pg_pool = None
    _checkpointer_singleton = None
