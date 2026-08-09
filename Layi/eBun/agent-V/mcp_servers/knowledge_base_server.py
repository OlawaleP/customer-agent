"""
Exposes knowledge base search as an MCP tool, and individual policy
docs as MCP resources (read-only context the host app can attach
directly, distinct from a tool the model actively decides to call).

Run standalone for testing:
    python mcp_servers/knowledge_base_server.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from app.tools.kb_tools import search_kb, _MOCK_DOCS

mcp = FastMCP("Knowledge Base")

mcp.tool()(search_kb)


@mcp.resource("policy://{doc_id}")
def get_policy_doc(doc_id: str) -> str:
    """Return the full text of a policy/FAQ document by id."""
    for doc in _MOCK_DOCS:
        if doc["id"] == doc_id:
            return f"{doc['title']}\n\n{doc['text']}"
    return f"No document found with id {doc_id}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
