"""
Evaluation harness. Run this against a CSV built from your real
historical tickets BEFORE letting the agent touch live customers.

CSV format (see golden_set_template.csv):
    ticket_id, customer_id, customer_message, expected_category, expected_outcome

expected_outcome must be "auto_reply" or "escalate".

Usage:
    python -m app.eval.run_eval app/eval/golden_set_template.csv
"""
from __future__ import annotations
import sys
import csv
import os

from dotenv import load_dotenv
load_dotenv()

# The golden set uses synthetic ticket IDs (T-001, T-002, ...) that don't exist
# in a real Chatwoot instance. If real Chatwoot credentials are present in the
# environment, force ticketing_tools into mock mode for this eval run so we
# don't send real API calls against nonexistent conversations. This must
# happen BEFORE `from app.graph import build_graph`, since ticketing_tools
# decides live-vs-mock at import time.
for _var in ("CHATWOOT_BASE_URL", "CHATWOOT_API_TOKEN", "CHATWOOT_ACCOUNT_ID"):
    os.environ.pop(_var, None)

from langgraph.checkpoint.memory import MemorySaver
from app.graph import build_graph


def run(csv_path: str):
    # Eval runs use synthetic thread IDs and don't need durable state across
    # restarts -- force in-memory so we never open a Postgres pool here at
    # all, regardless of whether DATABASE_URL is set in .env.
    graph = build_graph(checkpointer=MemorySaver())
    rows = list(csv.DictReader(open(csv_path)))

    correct_category = 0
    correct_outcome = 0
    dangerous_false_auto_resolves = []  # the failure mode that actually matters

    for row in rows:
        config = {"configurable": {"thread_id": f"eval-{row['ticket_id']}"}}
        result = graph.invoke(
            {
                "ticket_id": row["ticket_id"],
                "customer_id": row["customer_id"],
                "customer_message": row["customer_message"],
            },
            config=config,
        )

        got_category = result["triage"].category
        got_outcome = result["next_step"]

        if got_category == row["expected_category"]:
            correct_category += 1
        if got_outcome == row["expected_outcome"]:
            correct_outcome += 1

        if got_outcome == "auto_reply" and row["expected_outcome"] == "escalate":
            dangerous_false_auto_resolves.append(row["ticket_id"])

        print(f"{row['ticket_id']}: category={got_category} (expected {row['expected_category']}) "
              f"outcome={got_outcome} (expected {row['expected_outcome']})")

    n = len(rows)
    print("\n--- Summary ---")
    print(f"Category accuracy: {correct_category}/{n} ({100*correct_category/n:.1f}%)")
    print(f"Outcome accuracy:  {correct_outcome}/{n} ({100*correct_outcome/n:.1f}%)")
    print(f"DANGEROUS false auto-resolves (should have escalated): {len(dangerous_false_auto_resolves)}")
    if dangerous_false_auto_resolves:
        print(f"  -> {dangerous_false_auto_resolves}")
        print("  These are the ones to fix first -- a wrongly auto-resolved fraud/refund")
        print("  ticket is far more costly than an unnecessary escalation.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "app/eval/golden_set_template.csv")
