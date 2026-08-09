
"""
Knowledge base retrieval (the Agentic RAG piece).

Set QDRANT_URL to point at a real Qdrant instance with your policy/FAQ
docs embedded and indexed. Until then, falls back to a tiny in-memory
keyword-matched doc set so the pipeline is fully runnable today.

Swap-in path for real Qdrant:
  1. `pip install qdrant-client sentence-transformers`
  2. Embed your policy docs / FAQs into a collection, e.g. "kb_policies"
  3. Replace the body of search_kb() with an embed-query + qdrant search call
The function signature and return shape below is the contract the rest
of the system depends on -- keep it stable.

NOTE: The live Qdrant + SentenceTransformer path is currently DISABLED
to avoid downloading large models from Hugging Face Hub on startup.
Uncomment the relevant blocks below when you are ready to use real embeddings.
"""
from __future__ import annotations
import os

QDRANT_URL = os.getenv("QDRANT_URL")

_MOCK_DOCS = [
    {"id": "policy_refunds", "title": "Refund policy",
     "text": "Refunds under $50 can be auto-approved. Refunds of $50 or more, "
             "or any refund tied to a fraud claim, require human review before issuing."},
    {"id": "policy_reissue", "title": "Card reissue policy",
     "text": "A gift card reported as not delivered within 7 days of purchase can be "
             "reissued automatically, preserving the original balance."},
    {"id": "faq_redemption", "title": "Redemption FAQ",
     "text": "To redeem a gift card, enter the 16-digit code at checkout. Codes are "
             "case-insensitive. Redemption failures are usually due to an expired or "
             "already-used code."},
    {"id": "policy_fraud", "title": "Fraud handling policy",
     "text": "Any report of a stolen, hacked, or fraudulently used card must be escalated "
             "to a human agent immediately. Do not attempt automated resolution."},
]


def search_kb(query: str, top_k: int = 3) -> list[dict]:
    """
    Search internal policy documents and FAQs relevant to a customer query.
    Returns the top_k most relevant documents with their text.
    """
    # --- DISABLED: live Qdrant path (triggers Hugging Face model download) ---
    # if QDRANT_URL:
    #     try:
    #         return _search_qdrant(query, top_k)
    #     except Exception as e:
    #         # Common cause: the "kb_policies" collection hasn't been created/
    #         # indexed yet -- run scripts/ingest_kb.py first. Don't take down
    #         # the whole ticket pipeline over a KB outage; degrade to the
    #         # small built-in doc set and keep going.
    #         print(f"[kb_tools] Qdrant search failed ({e}); falling back to in-memory "
    #               f"KB docs. If this is unexpected, run: python scripts/ingest_kb.py")
    #         return _search_mock(query, top_k)

    return _search_mock(query, top_k)


def _search_mock(query: str, top_k: int) -> list[dict]:
    q = query.lower()
    scored = []
    for doc in _MOCK_DOCS:
        score = sum(1 for word in q.split() if word in doc["text"].lower())
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return [doc for _, doc in scored[:top_k]] or _MOCK_DOCS[:1]


def _search_qdrant(query: str, top_k: int) -> list[dict]:  # pragma: no cover
    # --- DISABLED: this block downloads the embedding model from Hugging Face ---
    # from qdrant_client import QdrantClient
    # from sentence_transformers import SentenceTransformer
    #
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # client = QdrantClient(url=QDRANT_URL)
    # vector = model.encode(query).tolist()
    # # Note: QdrantClient.search() was removed in newer qdrant-client versions
    # # (confirmed gone as of 1.18) in favor of query_points(), which wraps
    # # results in a `.points` attribute rather than returning a bare list.
    # result = client.query_points(collection_name="kb_policies", query=vector, limit=top_k)
    # return [{"id": h.id, "title": h.payload.get("title"), "text": h.payload.get("text")} for h in result.points]

    # Temporary fallback so the function still exists if anything calls it
    return _search_mock(query, top_k)
