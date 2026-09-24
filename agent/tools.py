from __future__ import annotations
import os
import json
import hashlib
import time
from typing import Any
from functools import lru_cache

import pyTigerGraph as tg
from google import genai as google_genai
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────
# Clients (lazy init)
# ─────────────────────────────────────────────────────────
_conn: tg.TigerGraphConnection | None = None
_genai_client = None

def _get_conn() -> tg.TigerGraphConnection:
    global _conn
    if _conn is None:
        host = os.getenv("TIGERGRAPH_HOST", "localhost")
        ssl  = os.getenv("TIGERGRAPH_USE_SSL", "false").lower() == "true"
        _conn = tg.TigerGraphConnection(
            host=f"https://{host}" if ssl else f"http://{host}",
            graphname=os.getenv("TIGERGRAPH_GRAPH", "fraud_investigation"),
            username=os.getenv("TIGERGRAPH_USERNAME", "tigergraph"),
            password=os.getenv("TIGERGRAPH_PASSWORD", "tigergraph"),
            restppPort=os.getenv("TIGERGRAPH_PORT", "9000"),
            useCert=ssl,
        )
        _conn.getToken(_conn.createSecret())
    return _conn

def _get_genai():
    global _genai_client
    if _genai_client is None:
        _genai_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _genai_client

def _embed(text: str) -> list[float]:
    """Generate embedding via Google Generative AI text-embedding-004 (free tier)."""
    client = _get_genai()
    result = client.models.embed_content(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
        contents=text[:8192],
    )
    return result.embeddings[0].values


# ─────────────────────────────────────────────────────────
# 1. Graph Investigation Tools
# ─────────────────────────────────────────────────────────

@tool
def graph_card_window(card_id: str, hours: int = 48) -> dict:
    """
    Retrieve all transactions for a card in the past N hours.
    Used to detect velocity anomalies and card testing sequences.
    """
    conn = _get_conn()
    edges = conn.getEdges("Card", card_id, "MADE")
    txn_ids = [e.get("to_id") for e in edges[:100]]
    txns = conn.getVerticesById("Transaction", txn_ids) if txn_ids else []
    return {"query": "card_window", "card_id": card_id, "hours": hours,
            "txn_count": len(txns), "transactions": txns}


@tool
def graph_txn_subgraph(txn_id: str) -> dict:
    """
    Get the neighborhood of a transaction: card, device, billing region.
    """
    conn = _get_conn()
    txn = conn.getVerticesById("Transaction", txn_id)
    if not txn:
        return {"error": f"Transaction {txn_id} not found"}
    card_id = txn[0].get("attributes", {}).get("card_id", "")
    device_edges = conn.getEdges("Transaction", txn_id, "FROM_DEVICE")
    region_edges = conn.getEdges("Transaction", txn_id, "BILLED_IN")
    device_ids = [e.get("to_id") for e in device_edges]
    return {"txn_id": txn_id, "transaction": txn, "card_id": card_id,
            "device_ids": device_ids, "region_edges": region_edges}


@tool
def graph_device_neighbors(txn_id: str, days: int = 90) -> dict:
    """
    Find all cards that used the same device as a given transaction.
    Key for detecting shared-device fraud rings.
    """
    conn = _get_conn()
    device_edges = conn.getEdges("Transaction", txn_id, "FROM_DEVICE")
    if not device_edges:
        return {"txn_id": txn_id, "devices": [], "other_txn_ids": []}
    device_ids = [e.get("to_id") for e in device_edges]
    other_txn_ids = []
    for dev_id in device_ids:
        edges = conn.getEdges("DeviceProfile", dev_id, "DEVICE_USED_IN")
        other_txn_ids.extend([e.get("to_id") for e in edges if e.get("to_id") != txn_id])
    return {"txn_id": txn_id, "device_ids": device_ids, "other_txn_ids": other_txn_ids[:20]}


@tool
def graph_velocity_check(card_id: str, hours: int = 24) -> dict:
    """
    Count and sum transactions for a card in a time window.
    Returns txn count and total amount to detect rapid spending.
    """
    conn = _get_conn()
    edges = conn.getEdges("Card", card_id, "MADE")
    txn_ids = [e.get("to_id") for e in edges[:100]]
    txns = conn.getVerticesById("Transaction", txn_ids) if txn_ids else []
    total = sum(t.get("attributes", {}).get("amount", 0) for t in txns)
    return {"card_id": card_id, "txn_count": len(txns),
            "total_amount": round(total, 2),
            "txn_ids": [t["v_id"] for t in txns]}


@tool
def graph_customer_history(customer_id: str) -> dict:
    """
    Get all cards and recent transactions for a customer.
    """
    conn = _get_conn()
    card_edges = conn.getEdges("Customer", customer_id, "OWNS")
    card_ids = [e.get("to_id") for e in card_edges]
    cards = conn.getVerticesById("Card", card_ids) if card_ids else []
    all_txns = []
    for cid in card_ids[:5]:
        edges = conn.getEdges("Card", cid, "MADE")
        txn_ids = [e.get("to_id") for e in edges[:20]]
        if txn_ids:
            all_txns.extend(conn.getVerticesById("Transaction", txn_ids))
    return {"customer_id": customer_id, "card_ids": card_ids,
            "cards": cards, "recent_txns": all_txns[:30]}


@tool
def graph_region_history(card_id: str) -> dict:
    """
    Get billing regions a card has been used in.
    Used for out-of-region fraud detection.
    """
    conn = _get_conn()
    edges = conn.getEdges("Card", card_id, "MADE")
    txn_ids = [e.get("to_id") for e in edges[:50]]
    txns = conn.getVerticesById("Transaction", txn_ids) if txn_ids else []
    regions: dict = {}
    for t in txns:
        region = str(t.get("attributes", {}).get("addr1", "unknown"))
        regions[region] = regions.get(region, 0) + 1
    home_count = regions.get("87.0", 0) + regions.get("87", 0)
    foreign = {k: v for k, v in regions.items() if "87" not in k}
    return {"card_id": card_id, "home_count": home_count,
            "foreign_regions": foreign, "total_txns": len(txns)}


@tool
def graph_similar_closed_cases(pattern: str, top_k: int = 5) -> dict:
    """
    Retrieve closed fraud cases matching the same pattern.
    Used as case memory retrieval for similar prior cases.
    """
    conn = _get_conn()
    try:
        cases = conn.getVertices(
            "ClosedCase",
            select="case_id,pattern,outcome,exposure_usd,actions_taken",
            filter=f'outcome=="confirmed_fraud" AND pattern=="{pattern}"',
            limit=top_k,
        )
    except Exception:
        cases = []
    return {"pattern": pattern, "similar_cases": cases}


@tool
def graph_card_testing_check(card_id: str, hours: int = 24, threshold: float = 10.0) -> dict:
    """
    Check for card testing: many small transactions in a short window.
    Returns count of small txns vs total txns.
    """
    conn = _get_conn()
    edges = conn.getEdges("Card", card_id, "MADE")
    txn_ids = [e.get("to_id") for e in edges[:100]]
    txns = conn.getVerticesById("Transaction", txn_ids) if txn_ids else []
    small = [t for t in txns if t.get("attributes", {}).get("amount", 999) <= threshold]
    return {"card_id": card_id, "total_txns": len(txns), "small_txns": len(small),
            "threshold": threshold,
            "is_card_testing_pattern": len(small) >= 3 and len(txns) >= 4}


# ─────────────────────────────────────────────────────────
# 2. GraphRAG Retrieval
# ─────────────────────────────────────────────────────────

@tool
def retrieve_policy_context(query: str) -> dict:
    """
    Retrieve relevant fraud policy rules and fraud pattern descriptions.
    Use this to check what policy rules apply to the current situation.
    """
    conn = _get_conn()
    try:
        policies = conn.getVertices("PolicyDocument", limit=10)
        patterns = conn.getVertices("FraudPattern", limit=10)
    except Exception:
        policies, patterns = [], []
    return {"query": query, "policy_docs": policies, "fraud_patterns": patterns}


@tool
def retrieve_similar_case_narratives(situation_description: str, top_k: int = 3) -> dict:
    """
    Search closed case analyst notes for similar past cases.
    Returns the most relevant past case narratives as investigation memory.
    """
    conn = _get_conn()
    try:
        cases = conn.getVertices("ClosedCase",
                                  select="case_id,pattern,outcome,analyst_notes,exposure_usd",
                                  filter='outcome=="confirmed_fraud"',
                                  limit=top_k)
    except Exception:
        cases = []
    return {"situation": situation_description, "similar_cases": cases}


# ─────────────────────────────────────────────────────────
# 3. Case Management
# ─────────────────────────────────────────────────────────

@tool
def case_create(
    case_pack_id: str, customer_id: str, card_id: str,
    status: str, verdict: str, fraud_probability: float,
    pattern: str, pattern_description: str,
    first_suspicious_txn_id: str, exposure_usd: float,
    summary: str, stop_reason: str,
) -> dict:
    """
    Write a FraudCase vertex to TigerGraph. Call this when fraud_probability
    reaches 0.30 or when requesting evidence. Action: CREATE_CASE (auto).
    """
    conn = _get_conn()
    case_id = f"CASE-{case_pack_id}-{int(time.time())}"
    conn.upsertVertex("FraudCase", case_id, {
        "source_case_pack_id": case_pack_id,
        "customer_id": customer_id,
        "card_id": card_id,
        "opened_at": "2026-01-01 00:00:00",
        "status": status,
        "verdict": verdict,
        "fraud_probability": fraud_probability,
        "pattern": pattern,
        "pattern_description": pattern_description,
        "first_suspicious_txn_id": first_suspicious_txn_id,
        "exposure_usd": exposure_usd,
        "summary": summary,
        "stop_reason": stop_reason,
        "written_to_graph": True,
    })
    return {"action": "CREATE_CASE", "route": "auto", "case_id": case_id, "written": True}


# ─────────────────────────────────────────────────────────
# 4. Evidence Gathering (stubs)
# ─────────────────────────────────────────────────────────

@tool
def request_customer_validation(card_id: str, txn_id: str, message: str) -> dict:
    """
    (STUB) Ask the cardholder whether they made a specific transaction.
    Policy: VERIFY_WITH_CUSTOMER (auto). Rule R1.
    """
    return {"action": "VERIFY_WITH_CUSTOMER", "route": "auto", "card_id": card_id,
            "txn_id": txn_id, "message_sent": message, "status": "sent",
            "note": "Response not yet received. Agent must simulate and state assumption."}


@tool
def request_step_up_auth(card_id: str, reason: str) -> dict:
    """
    (STUB) Require a one-time passcode or app confirmation.
    Policy: STEP_UP_AUTH (auto). Rule R1, R5.
    """
    return {"action": "STEP_UP_AUTH", "route": "auto", "card_id": card_id,
            "reason": reason, "status": "requested",
            "note": "Step-up auth requested. Agent must simulate response."}


@tool
def request_analyst_info(case_id: str, question: str) -> dict:
    """
    (STUB) Request additional information from a fraud analyst.
    Policy: ESCALATE_TO_ANALYST (auto).
    """
    return {"action": "ESCALATE_TO_ANALYST", "route": "auto",
            "case_id": case_id, "question": question, "status": "requested"}


# ─────────────────────────────────────────────────────────
# 5. Action Recommendation
# ─────────────────────────────────────────────────────────

APPROVAL_ROUTES: dict[str, str] = {
    "ALLOW_TRANSACTION":        "auto",
    "DECLINE_TRANSACTION":      "L1",
    "MONITOR_CARD":             "auto",
    "MONITOR_CONNECTED_CARDS":  "auto",
    "WARN_CUSTOMER":            "auto",
    "VERIFY_WITH_CUSTOMER":     "auto",
    "STEP_UP_AUTH":             "auto",
    "BLOCK_CARD":               "L1",
    "BLOCK_ALL_CARDS":          "L2",
    "GENERATE_REPORT":          "auto",
    "CREATE_CASE":              "auto",
    "FILE_REPORT":              "L2",
    "ESCALATE_TO_ANALYST":      "auto",
    "CLOSE_NO_FRAUD":           "auto",
}


def get_approval_route(action: str, exposure_usd: float = 0.0) -> str:
    if action == "BLOCK_CARD" and exposure_usd > 2500:
        return "L2"
    return APPROVAL_ROUTES.get(action, "L2")


@tool
def recommend_action(action: str, reason: str, exposure_usd: float = 0.0) -> dict:
    """
    Recommend a policy action with the correct approval route.
    Use this to build the next_best_actions list.
    """
    route = get_approval_route(action, exposure_usd)
    return {"action": action, "route": route, "reason": reason, "exposure_usd": exposure_usd}


# ─────────────────────────────────────────────────────────
# Tool list for the agent
# ─────────────────────────────────────────────────────────
INVESTIGATION_TOOLS = [
    graph_card_window,
    graph_txn_subgraph,
    graph_device_neighbors,
    graph_velocity_check,
    graph_customer_history,
    graph_region_history,
    graph_similar_closed_cases,
    graph_card_testing_check,
    retrieve_policy_context,
    retrieve_similar_case_narratives,
    request_customer_validation,
    request_step_up_auth,
    request_analyst_info,
    case_create,
    recommend_action,
]

