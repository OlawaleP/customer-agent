"""
Creates the "kb_policies" Qdrant collection and indexes your policy/FAQ
documents into it. Run this once before real (non-mock) KB retrieval will
work -- kb_tools.py's search_kb() queries this exact collection name.

Usage:
    # Index the built-in sample docs (good for confirming the pipeline works):
    python scripts/ingest_kb.py

    # Index your real policy docs from a folder of .md/.txt files, one
    # document per file (filename becomes the title):
    python scripts/ingest_kb.py --docs-dir path/to/your/policy_docs
"""
from __future__ import annotations
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = "kb_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim, matches VECTOR_SIZE below
VECTOR_SIZE = 384


def load_docs_from_dir(docs_dir: str) -> list[dict]:
    import pathlib
    docs = []
    for path in pathlib.Path(docs_dir).glob("*"):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        docs.append({
            "id": path.stem,
            "title": path.stem.replace("_", " ").replace("-", " ").title(),
            "text": path.read_text(encoding="utf-8"),
        })
    return docs


def load_builtin_sample_docs() -> list[dict]:
    from app.tools.kb_tools import _MOCK_DOCS
    return _MOCK_DOCS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default=None,
                         help="Folder of .md/.txt policy docs to index. "
                              "Omit to index the built-in sample docs instead.")
    parser.add_argument("--recreate", action="store_true",
                         help="Drop and recreate the collection instead of upserting into it.")
    args = parser.parse_args()

    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        print("QDRANT_URL is not set in .env -- nothing to index against. Set it first.")
        sys.exit(1)

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer

    docs = load_docs_from_dir(args.docs_dir) if args.docs_dir else load_builtin_sample_docs()
    if not docs:
        print(f"No documents found{' in ' + args.docs_dir if args.docs_dir else ''}. Nothing to index.")
        sys.exit(1)

    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url)

    exists = client.collection_exists(COLLECTION_NAME)
    if args.recreate or not exists:
        print(f"{'Recreating' if exists else 'Creating'} collection '{COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    print(f"Embedding and indexing {len(docs)} document(s)...")
    points = []
    for i, doc in enumerate(docs):
        vector = model.encode(doc["text"]).tolist()
        points.append(PointStruct(
            id=i,
            vector=vector,
            payload={"doc_id": doc["id"], "title": doc["title"], "text": doc["text"]},
        ))
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Done. Indexed {len(points)} document(s) into '{COLLECTION_NAME}'.")
    print("Test it with: python -m app.eval.run_eval app/eval/golden_set_template.csv")


if __name__ == "__main__":
    main()
