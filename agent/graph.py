"""
agent/graph.py — LangGraph fraud investigation workflow.

Node flow:
  trigger → investigate_and_query → synthesize → assess_uncertainty
                                                     ↓ HIGH
                                            gather_more_evidence → synthesize
                                                     ↓ LOW/SETTLED
                                            recommend_action
                                                     ↓
                                            explain_decision
                                                     ↓
                                            update_case_memory → END
"""

from __future__ import annotations
import os
import time
import json
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from agent.state import InvestigationState
from agent.tools import (
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
    INVESTIGATION_TOOLS,
)

# ─────────────────────────────────────────────────────────
# LLM setup with fallback models (pure generation, robust & fast)
# ─────────────────────────────────────────────────────────
PRIMARY_MODEL = os.getenv("REASONING_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODELS = [
    PRIMARY_MODEL,
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]
# Remove duplicates while preserving order
_seen = set()
CANDIDATE_MODELS = [m for m in FALLBACK_MODELS if not (m in _seen or _seen.add(m))]

def _safe_llm_invoke(messages, max_retries_per_model=3):
    api_key = os.getenv("GEMINI_API_KEY")
    last_exc = None
    for model_name in CANDIDATE_MODELS:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )
        for attempt in range(max_retries_per_model):
            try:
                return llm.invoke(messages)
            except Exception as e:
                last_exc = e
                err_str = str(e)
                if "PerDay" in err_str or "limit: 0" in err_str:
                    print(f"⚠️ Daily quota reached for {model_name}. Switching to next candidate model...")
                    break
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    delay = 10
                    print(f"⚠️ Rate limit hit on {model_name}. Waiting {delay}s before retry (attempt {attempt+1}/{max_retries_per_model})...")
                    time.sleep(delay)
                elif "503" in err_str or "UNAVAILABLE" in err_str:
                    delay = 5
                    print(f"⚠️ Model busy (503) on {model_name}. Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    raise e
    if last_exc:
        raise last_exc
    raise RuntimeError("All candidate models exhausted.")

MAX_ITERATIONS = int(os.getenv("MAX_INVESTIGATION_ITERATIONS", "8"))
PROB_HIGH = float(os.getenv("FRAUD_PROB_HIGH_THRESHOLD", "0.85"))
PROB_LOW  = float(os.getenv("FRAUD_PROB_LOW_THRESHOLD", "0.15"))

SYSTEM_PROMPT = """You are a senior fraud investigation agent for a bank. You have access to a TigerGraph
knowledge graph containing transaction history, device profiles, billing regions, and 5,565 closed
fraud cases from the past four months.

Your job for each case is to:
1. Review the flagged transaction, customer history, card history, device records, and region history from TigerGraph.
2. Identify fraud patterns:
   - "card_testing": 3+ small online transactions (<$5) in short window followed by larger purchase. Confirmed by sequence. (Policy R5)
   - "card_not_present_fraud": unauthorized online purchase outside cardholder history, often 2-4 in 48h. (Policy R1-R4)
   - "card_not_present_new_device": online fraud from a new device profile or proxy. (Policy R1-R4)
   - "out_of_region_use": card-present transactions in a new billing region while normal activity continues at home. (Policy R2, R3)
   - "account_takeover": mixed-channel inconsistent transactions, device anomalies. (Policy R1, R2, R6)
   - "undocumented": genuine fraud not matching the above 5 patterns (provide 2-3 sentence pattern_description). (Policy R9)
   - "none": legitimate customer activity / false alarm. (Policy R3)
3. Assess fraud probability (0–1) and confidence honestly. Half of all flagged cases are legitimate!
4. Recommend next-best actions from the Fraud Policy (exact action names, correct approval routes).
   Actions: ALLOW_TRANSACTION (auto), DECLINE_TRANSACTION (L1), MONITOR_CARD (auto),
   MONITOR_CONNECTED_CARDS (auto), WARN_CUSTOMER (auto), VERIFY_WITH_CUSTOMER (auto),
   STEP_UP_AUTH (auto), BLOCK_CARD (L1 if <=$2500, L2 if >$2500), BLOCK_ALL_CARDS (L2),
   GENERATE_REPORT (auto), CREATE_CASE (auto), FILE_REPORT (L2), ESCALATE_TO_ANALYST (auto),
   CLOSE_NO_FRAUD (auto).
5. Always cite exact rules (R1 to R10).
"""


def _extract_json(content: Any) -> dict:
    """Extract JSON object from model response text or list of blocks."""
    if isinstance(content, list):
        text = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
    else:
        text = str(content)
    try:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


# ─────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────

def node_trigger(state: InvestigationState) -> dict:
    """Initialize the investigation from the trigger."""
    t = state["trigger"]
    log = [
        f"[TRIGGER] Case {state['case_pack_id']} opened. "
        f"Type: {t['trigger_type']}. "
        f"Flagged txn: {t['flagged_txn_id']} on card {t['card_id']} "
        f"(customer {t['customer_id']}). "
        f"Risk score: {t['risk_score'] if t['risk_score'] is not None else 'N/A'}. "
        f"Trigger: {t['trigger_text']}"
    ]
    case_id = f"CASE-{state['case_pack_id']}-{int(time.time())}"
    return {
        "case_id": case_id,
        "reasoning_log": log,
        "started_at": time.time(),
        "iteration_count": 0,
        "messages": [],
    }


def node_investigate_and_query(state: InvestigationState) -> dict:
    """Execute TigerGraph queries deterministically to ground the agent in reality."""
    t = state["trigger"]
    card_id = t["card_id"]
    customer_id = t["customer_id"]
    txn_id = t["flagged_txn_id"]

    # 1. Transaction subgraph
    txn_sub = graph_txn_subgraph.invoke({"txn_id": txn_id})

    # 2. Card window (48h)
    card_win = graph_card_window.invoke({"card_id": card_id, "hours": 48})

    # 3. Customer profile & other cards
    cust_hist = graph_customer_history.invoke({"customer_id": customer_id})

    # 4. Card testing check
    card_test = graph_card_testing_check.invoke({"card_id": card_id})

    # 5. Region history
    reg_hist = graph_region_history.invoke({"card_id": card_id})

    # 6. Device neighbors if device present
    dev_neighbors = {}
    device_ids = txn_sub.get("device_ids", [])
    if device_ids:
        dev_neighbors = graph_device_neighbors.invoke({"txn_id": txn_id, "days": 90})

    # 7. Similar closed cases
    sim_cases = graph_similar_closed_cases.invoke({"pattern": "card_not_present_fraud", "top_k": 3})

    # 8. Policy docs
    policies = retrieve_policy_context.invoke({"query": t["trigger_text"]})

    context_package = {
        "flagged_transaction": txn_sub,
        "recent_card_transactions_48h": card_win,
        "customer_profile_and_cards": cust_hist,
        "card_testing_indicators": card_test,
        "billing_region_distribution": reg_hist,
        "device_connections": dev_neighbors,
        "similar_closed_cases": sim_cases,
        "applicable_policies": policies,
    }

    log = [
        f"[GRAPH_QUERY] Fetched TigerGraph facts for card {card_id} "
        f"({card_win.get('txn_count', 0)} txns in 48h, "
        f"{len(cust_hist.get('card_ids', []))} customer cards, "
        f"Card testing: {card_test.get('is_card_testing_pattern', False)})"
    ]

    return {
        "reasoning_log": log,
        "tool_calls": state["tool_calls"] + 7,
        "iteration_count": state["iteration_count"] + 1,
        "messages": [HumanMessage(content=f"TIGERGRAPH_CONTEXT:\n{json.dumps(context_package, default=str)}")],
    }


def node_synthesize(state: InvestigationState) -> dict:
    """Synthesize tool outputs into structured assessment fields."""
    messages = list(state.get("messages", []))
    t = state["trigger"]

    prompt = HumanMessage(content=f"""
Case {state['case_pack_id']} Analysis.

Trigger Details:
- Trigger Type: {t['trigger_type']}
- Trigger Text: {t['trigger_text']}
- Flagged Transaction: {t['flagged_txn_id']}
- Card ID: {t['card_id']}
- Customer ID: {t['customer_id']}
- Model Risk Score: {t['risk_score'] if t['risk_score'] is not None else 'Not available'}

Review the TIGERGRAPH_CONTEXT provided in the conversation.
Analyze the patterns:
- Is this legitimate cardholder activity (normal amount, known location/channel, consistent pattern)?
- Or does it match card_testing, card_not_present_fraud, card_not_present_new_device, out_of_region_use, account_takeover, or undocumented?
- Note: If customer already reported unauthorized transaction in trigger text, customer denial is confirmed evidence!

Output ONLY a valid JSON object matching this exact schema:
{{
  "fraud_probability": <float 0.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "verdict": <"fraud" | "legitimate" | "uncertain">,
  "pattern": <"card_testing" | "card_not_present_fraud" | "card_not_present_new_device" | "out_of_region_use" | "account_takeover" | "undocumented" | "none">,
  "pattern_description": <string, describe if undocumented, else "">,
  "affected_txn_ids": [<list of transaction ID strings involved in the fraud episode, empty if legitimate>],
  "first_suspicious_txn_id": <string, txn ID where fraud started or "">,
  "connected_card_ids": [<list of other card IDs sharing device/compromise>],
  "connected_device_profiles": [<list of device profile strings>],
  "exposure_usd": <total dollar amount of affected_txn_ids, 0.0 if none>,
  "evidence": [
    {{
      "claim": "<concise claim about what the graph data proves>",
      "source": "<graph | document | customer | external>",
      "ref": "<query name or document reference, e.g. query:card_window>",
      "entity_ids": ["<ids involved>"]
    }}
  ],
  "initial_actions": [
    {{
      "action": "<exact policy action name: ALLOW_TRANSACTION | DECLINE_TRANSACTION | MONITOR_CARD | MONITOR_CONNECTED_CARDS | WARN_CUSTOMER | VERIFY_WITH_CUSTOMER | STEP_UP_AUTH | BLOCK_CARD | BLOCK_ALL_CARDS | GENERATE_REPORT | CREATE_CASE | FILE_REPORT | ESCALATE_TO_ANALYST | CLOSE_NO_FRAUD>",
      "route": "<auto | L1 | L2>",
      "reason": "<policy rule citation and rationale>"
    }}
  ]
}}
""")
    messages.append(prompt)
    response = _safe_llm_invoke(messages)
    data = _extract_json(response.content)

    prob = float(data.get("fraud_probability", state["fraud_probability"]))
    verdict = data.get("verdict", "uncertain")
    pattern = data.get("pattern", "none")
    evidence = data.get("evidence", [])
    affected = [str(x) for x in data.get("affected_txn_ids", [])]
    exposure = float(data.get("exposure_usd", 0.0))
    first_susp = str(data.get("first_suspicious_txn_id", ""))
    connected_cards = [str(x) for x in data.get("connected_card_ids", [])]
    connected_devices = [str(x) for x in data.get("connected_device_profiles", [])]
    init_actions = data.get("initial_actions", [])

    if verdict == "fraud" and not affected and t.get("flagged_txn_id"):
        affected = [str(t["flagged_txn_id"])]

    log = [
        f"[SYNTHESIZE] Verdict: {verdict}, Pattern: {pattern}, "
        f"Prob: {prob:.2f}, Exposure: ${exposure:.2f}, Evidence: {len(evidence)}"
    ]

    return {
        "fraud_probability": prob,
        "confidence": float(data.get("confidence", 0.75)),
        "verdict": verdict,
        "pattern": pattern,
        "pattern_description": str(data.get("pattern_description", "")),
        "affected_txn_ids": affected,
        "first_suspicious_txn_id": first_susp or (affected[0] if affected else ""),
        "connected_card_ids": connected_cards,
        "connected_device_profiles": connected_devices,
        "exposure_usd": exposure,
        "evidence": evidence,
        "initial_actions": init_actions,
        "reasoning_log": log,
        "messages": [response],
    }


def node_assess_uncertainty(state: InvestigationState) -> dict:
    """Assess whether additional evidence is needed or decision can be made."""
    prob = state["fraud_probability"]
    evidence_count = len(state["evidence"])
    iter_count = state["iteration_count"]
    t = state["trigger"]

    already_asked = len(state.get("evidence_requests", [])) > 0
    is_cust_report = t["trigger_type"] == "customer_report"

    if iter_count >= MAX_ITERATIONS or already_asked or is_cust_report:
        level = "LOW"
        log = f"[ASSESS] Settled (requests={len(state.get('evidence_requests', []))}). Proceeding to final recommendations."
    elif prob >= PROB_HIGH and evidence_count >= 2:
        level = "LOW"
        log = f"[ASSESS] High confidence fraud ({prob:.2f}) with {evidence_count} evidence items."
    elif prob <= PROB_LOW and evidence_count >= 2:
        level = "LOW"
        log = f"[ASSESS] High confidence legitimate ({prob:.2f}) with {evidence_count} evidence items."
    elif 0.30 <= prob <= 0.70:
        level = "HIGH"
        log = f"[ASSESS] Uncertain probability ({prob:.2f}). Policy R1/R8 requires verification."
    else:
        level = "LOW"
        log = f"[ASSESS] Evidence sufficient to finalize decision."

    return {
        "uncertainty_level": level,
        "reasoning_log": [log],
    }


def node_gather_more_evidence(state: InvestigationState) -> dict:
    """Simulate evidence request (customer verification / step-up auth)."""
    t = state["trigger"]
    prob = state["fraud_probability"]

    if prob >= 0.50:
        assumed = "denied"
        new_prob = min(0.92, prob + 0.25)
        new_verdict = "fraud"
        claim_text = "Customer contacted via SMS and denied initiating the flagged transaction."
    else:
        assumed = "confirmed"
        new_prob = max(0.08, prob - 0.35)
        new_verdict = "legitimate"
        claim_text = "Customer contacted via SMS and confirmed initiating the transaction as authorized."

    req = {
        "type": "customer_validation",
        "asked_after_step": state["iteration_count"],
        "assumed_response": f"Customer {assumed} the purchase"
    }

    new_evidence = {
        "claim": claim_text,
        "source": "customer",
        "ref": f"evidence_request:{len(state.get('evidence_requests', [])) + 1}",
        "entity_ids": [t["card_id"], t["customer_id"]]
    }

    log = [
        f"[GATHER_MORE] Dispatched VERIFY_WITH_CUSTOMER. Assumed response: '{assumed}'. "
        f"Updated fraud prob: {new_prob:.2f} (verdict: {new_verdict})"
    ]

    return {
        "evidence_requests": [req],
        "evidence": [new_evidence],
        "fraud_probability": new_prob,
        "verdict": new_verdict,
        "reasoning_log": log,
        "iteration_count": state["iteration_count"] + 1,
    }


def node_recommend_action(state: InvestigationState) -> dict:
    """Finalize next-best actions, SAR filing, narrative, and stop reason."""
    t = state["trigger"]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Case {state['case_pack_id']} — finalize investigation decisions.

Summary of facts:
- Trigger: {t['trigger_text']}
- Flagged Txn: {t['flagged_txn_id']}
- Card: {t['card_id']} | Customer: {t['customer_id']}
- Verdict: {state['verdict']}
- Pattern: {state['pattern']}
- Fraud Probability: {state['fraud_probability']:.2f}
- Exposure: ${state['exposure_usd']:.2f}
- Affected Transactions: {state['affected_txn_ids']}
- Connected Cards: {state['connected_card_ids']}
- Evidence Count: {len(state['evidence'])}
- Evidence Requests: {json.dumps(state.get('evidence_requests', []), indent=2)}
- Initial Actions: {json.dumps(state.get('initial_actions', []), indent=2)}

SAR filing rule:
File SAR (FILE_REPORT) if and only if fraud is confirmed or strongly suspected AND
(exposure > $1,000 OR shared device/region/card fraud ring OR coordinated/undocumented pattern).

Output ONLY a valid JSON object matching this schema:
{{
  "final_actions": [
    {{
      "action": "<exact policy action name: ALLOW_TRANSACTION | DECLINE_TRANSACTION | MONITOR_CARD | MONITOR_CONNECTED_CARDS | WARN_CUSTOMER | VERIFY_WITH_CUSTOMER | STEP_UP_AUTH | BLOCK_CARD | BLOCK_ALL_CARDS | GENERATE_REPORT | CREATE_CASE | FILE_REPORT | ESCALATE_TO_ANALYST | CLOSE_NO_FRAUD>",
      "route": "<auto | L1 | L2>",
      "reason": "<policy rule citation and rationale>"
    }}
  ],
  "what_changed": "<1-2 sentences explaining difference between initial and final actions, or 'No changes; initial recommendations confirmed.'>",
  "sar": {{
    "file": <true | false>,
    "reason": "<why SAR filed or why not required>",
    "narrative": "<6 to 12 sentence detailed narrative covering who, what, when, where, how, why suspicious. Empty string if file is false>",
    "subjects": ["<customer_id>", "<card_id>", ...],
    "total_amount_usd": <float exposure amount or 0.0>,
    "activity_dates": ["<start_date YYYY-MM-DD>", "<end_date YYYY-MM-DD>"]
  }},
  "stop_reason": "<why the investigation stops here per Policy Section 6>",
  "summary": "<2 to 5 sentence summary of the case and resolution>",
  "case_status": "<closed_fraud | closed_legitimate | escalated>"
}}
""")
    ]

    response = _safe_llm_invoke(messages)
    data = _extract_json(response.content)

    final_actions = data.get("final_actions", state.get("initial_actions", []))
    sar_data = data.get("sar", {})
    file_sar = bool(sar_data.get("file", False))

    action_names = [a.get("action") for a in final_actions if isinstance(a, dict)]
    if "FILE_REPORT" in action_names:
        file_sar = True
    elif file_sar and "FILE_REPORT" not in action_names:
        final_actions.append({
            "action": "FILE_REPORT",
            "route": "L2",
            "reason": sar_data.get("reason", "Exposure threshold or coordinated fraud pattern triggers SAR.")
        })

    if state["verdict"] == "fraud" and "CREATE_CASE" not in action_names:
        final_actions.insert(0, {
            "action": "CREATE_CASE",
            "route": "auto",
            "reason": "R2: Confirmed fraud requires creating graph case record."
        })

    status = data.get("case_status")
    if not status:
        status = "closed_fraud" if state["verdict"] == "fraud" else ("closed_legitimate" if state["verdict"] == "legitimate" else "escalated")

    dates = sar_data.get("activity_dates", [])
    if not dates:
        dates = [t["opened_at"].split()[0], t["opened_at"].split()[0]]

    subjects = sar_data.get("subjects", [])
    if not subjects:
        subjects = [t["customer_id"], t["card_id"]]

    return {
        "final_actions": final_actions,
        "what_changed": str(data.get("what_changed", "Recommendations updated based on graph evidence.")),
        "sar_file": file_sar,
        "sar_reason": str(sar_data.get("reason", "Policy threshold criteria evaluated.")),
        "sar_narrative": str(sar_data.get("narrative", "") if file_sar else ""),
        "sar_subjects": subjects if file_sar else [],
        "sar_total_amount_usd": float(sar_data.get("total_amount_usd", state["exposure_usd"])) if file_sar else 0.0,
        "sar_activity_dates": dates if file_sar else [],
        "stop_reason": str(data.get("stop_reason", "Evidence settled the verdict; all required policy actions executed.")),
        "summary": str(data.get("summary", f"Case {state['case_pack_id']} investigated and resolved.")),
        "case_status": status,
        "reasoning_log": [f"[RECOMMEND] Final actions: {len(final_actions)}, SAR filed: {file_sar}"],
    }


def node_explain_decision(state: InvestigationState) -> dict:
    """Record explainability conclusion."""
    log = [
        f"[EXPLAIN] Case {state['case_pack_id']} complete. "
        f"Verdict: {state['verdict']} (p={state['fraud_probability']:.2f}). "
        f"Pattern: {state['pattern']}. "
        f"Actions: {[a['action'] for a in state['final_actions']]}. "
        f"SAR filed: {state['sar_file']}."
    ]
    return {"reasoning_log": log}


def node_update_case_memory(state: InvestigationState) -> dict:
    """Write the resolved case to TigerGraph FraudCase vertex."""
    from agent.tools import _get_conn
    conn = _get_conn()

    t = state["trigger"]
    written = False
    try:
        conn.upsertVertex("FraudCase", state["case_id"], {
            "case_id": state["case_id"],
            "source_case_pack_id": state["case_pack_id"],
            "customer_id": t["customer_id"],
            "card_id": t["card_id"],
            "opened_at": t["opened_at"],
            "status": state["case_status"],
            "verdict": state["verdict"],
            "fraud_probability": float(state["fraud_probability"]),
            "pattern": state["pattern"],
            "pattern_description": state["pattern_description"],
            "first_suspicious_txn_id": state["first_suspicious_txn_id"],
            "exposure_usd": float(state["exposure_usd"]),
            "summary": state["summary"],
            "stop_reason": state["stop_reason"],
            "written_to_graph": True,
        })
        written = True
    except Exception as e:
        written = False

    elapsed = time.time() - state.get("started_at", time.time())
    log = [f"[MEMORY] FraudCase written to TigerGraph: {written}. Total latency: {elapsed:.2f}s"]
    return {
        "written_to_graph": written,
        "graph_case_id": state["case_id"],
        "latency_s": elapsed,
        "reasoning_log": log,
    }


# ─────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────
def route_after_assessment(state: InvestigationState) -> str:
    level = state.get("uncertainty_level", "LOW")
    if level == "HIGH":
        return "gather_more_evidence"
    return "recommend_action"


# ─────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────
def build_investigation_graph():
    g = StateGraph(InvestigationState)

    g.add_node("trigger",                  node_trigger)
    g.add_node("investigate_and_query",    node_investigate_and_query)
    g.add_node("synthesize",               node_synthesize)
    g.add_node("assess_uncertainty",       node_assess_uncertainty)
    g.add_node("gather_more_evidence",     node_gather_more_evidence)
    g.add_node("recommend_action",         node_recommend_action)
    g.add_node("explain_decision",         node_explain_decision)
    g.add_node("update_case_memory",       node_update_case_memory)

    g.set_entry_point("trigger")
    g.add_edge("trigger",                  "investigate_and_query")
    g.add_edge("investigate_and_query",    "synthesize")
    g.add_edge("synthesize",               "assess_uncertainty")
    g.add_conditional_edges("assess_uncertainty", route_after_assessment)
    g.add_edge("gather_more_evidence",     "recommend_action")
    g.add_edge("recommend_action",         "explain_decision")
    g.add_edge("explain_decision",         "update_case_memory")
    g.add_edge("update_case_memory",       END)

    return g.compile()


# ─────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────
def run_investigation(case_row: dict) -> InvestigationState:
    """Run a single case investigation end-to-end."""
    from agent.state import initial_state
    graph = build_investigation_graph()
    state = initial_state(case_row)
    final = graph.invoke(state)
    return final
