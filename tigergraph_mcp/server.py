"""
tigergraph_mcp/server.py — Model Context Protocol (MCP) Server for TigerGraph Fraud Investigation.

Exposes graph queries and GraphRAG operations as standard MCP tools for AI agents.
Conforms to TigerGraph MCP specification (https://github.com/tigergraph/tigergraph-mcp).
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer
from agent.tools import (
    graph_card_window,
    graph_txn_subgraph,
    graph_customer_history,
    graph_card_testing_check,
    graph_region_history,
    graph_device_neighbors,
    graph_similar_closed_cases,
    retrieve_policy_context,
    _get_conn,
)

# Initialize MCP Server
app = MCPServer("TigerGraph-Fraud-Investigation-MCP")


@app.tool()
def get_card_window(card_id: str, hours: int = 48) -> str:
    """Retrieve all transactions made on a card in a given time window (default 48 hours)."""
    res = graph_card_window.invoke({"card_id": card_id, "hours": hours})
    return json.dumps(res)


@app.tool()
def get_transaction_subgraph(txn_id: str) -> str:
    """Retrieve 1-hop subgraph around a transaction: card, customer, device, email, billing region."""
    res = graph_txn_subgraph.invoke({"txn_id": txn_id})
    return json.dumps(res)


@app.tool()
def get_customer_history(customer_id: str) -> str:
    """Retrieve customer profile, all owned cards, and transaction history across all cards."""
    res = graph_customer_history.invoke({"customer_id": customer_id})
    return json.dumps(res)


@app.tool()
def check_card_testing(card_id: str, amount_threshold: float = 5.0, count_threshold: int = 3) -> str:
    """Check if a card exhibits card testing: 3+ small online transactions (<$5) followed by a larger purchase."""
    res = graph_card_testing_check.invoke({
        "card_id": card_id,
        "amount_threshold": amount_threshold,
        "count_threshold": count_threshold,
    })
    return json.dumps(res)


@app.tool()
def get_region_history(card_id: str) -> str:
    """Analyze billing region (addr1/addr2) distribution for a card to detect out-of-region use."""
    res = graph_region_history.invoke({"card_id": card_id})
    return json.dumps(res)


@app.tool()
def get_device_neighbors(txn_id: str, days: int = 90) -> str:
    """Find other cards and customers that transacted from the same device profile within 90 days."""
    res = graph_device_neighbors.invoke({"txn_id": txn_id, "days": days})
    return json.dumps(res)


@app.tool()
def get_similar_closed_cases(pattern: str, top_k: int = 3) -> str:
    """Retrieve historical closed fraud cases matching a fraud pattern to inform decision-making."""
    res = graph_similar_closed_cases.invoke({"pattern": pattern, "top_k": top_k})
    return json.dumps(res)


@app.tool()
def query_policy_knowledge(query: str, top_k: int = 2) -> str:
    """GraphRAG: retrieve relevant sections of bank fraud policy rules and approval matrices."""
    res = retrieve_policy_context.invoke({"query": query, "top_k": top_k})
    return json.dumps(res)


@app.tool()
def upsert_fraud_case(
    case_id: str,
    source_case_pack_id: str,
    customer_id: str,
    card_id: str,
    opened_at: str,
    status: str,
    verdict: str,
    fraud_probability: float,
    pattern: str,
    exposure_usd: float,
    summary: str,
) -> str:
    """Write an investigated fraud case back into TigerGraph case memory."""
    conn = _get_conn()
    conn.upsertVertex("FraudCase", case_id, {
        "case_id": case_id,
        "source_case_pack_id": source_case_pack_id,
        "customer_id": customer_id,
        "card_id": card_id,
        "opened_at": opened_at,
        "status": status,
        "verdict": verdict,
        "fraud_probability": fraud_probability,
        "pattern": pattern,
        "exposure_usd": exposure_usd,
        "summary": summary,
    })
    return json.dumps({"status": "success", "case_id": case_id})


if __name__ == "__main__":
    import asyncio
    print("🚀 Starting TigerGraph Fraud Investigation MCP Server on stdio...")
    asyncio.run(app.run_stdio_async())
