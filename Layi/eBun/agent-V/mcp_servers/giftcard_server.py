"""
Exposes app/tools/giftcard_tools.py as an MCP server, so any
MCP-compatible client (Claude, OpenAI Agents SDK, LangGraph via
langchain-mcp-adapters, a CrewAI sub-crew, etc.) can call these
same tools -- not just this LangGraph app.

Run standalone for testing:
    python mcp_servers/giftcard_server.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from app.tools import giftcard_tools

mcp = FastMCP("Gift Card Operations")

mcp.tool()(giftcard_tools.check_balance)
mcp.tool()(giftcard_tools.check_redemption_status)
mcp.tool()(giftcard_tools.reissue_card)
mcp.tool()(giftcard_tools.check_transaction_history)
mcp.tool()(giftcard_tools.issue_refund)

if __name__ == "__main__":
    mcp.run(transport="stdio")
