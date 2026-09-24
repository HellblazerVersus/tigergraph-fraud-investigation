"""
agent/state.py — InvestigationState and related types for LangGraph.
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Literal
from datetime import datetime
import operator
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────────────────
# Enums / literals
# ─────────────────────────────────────────────────────────
FraudPattern = Literal[
    "card_testing",
    "card_not_present_fraud",
    "card_not_present_new_device",
    "out_of_region_use",
    "account_takeover",
    "undocumented",
    "none",
]

CaseStatus = Literal["open", "closed_fraud", "closed_legitimate", "escalated"]
Verdict     = Literal["fraud", "legitimate", "uncertain"]
ApprovalRoute = Literal["auto", "L1", "L2"]
EvidenceSource = Literal["graph", "document", "customer", "external"]
EvidenceRequestType = Literal["customer_validation", "step_up_auth", "analyst_info"]


# ─────────────────────────────────────────────────────────
# Sub-models (plain dicts for JSON-serializable state)
# ─────────────────────────────────────────────────────────
class Evidence(TypedDict):
    claim: str
    source: EvidenceSource
    ref: str                      # query name, doc section, or request id
    entity_ids: list[str]


class NextAction(TypedDict):
    action: str                   # exact policy action name
    route: ApprovalRoute
    reason: str                   # cite policy rule


class EvidenceRequest(TypedDict):
    type: EvidenceRequestType
    asked_after_step: int
    assumed_response: str


class TriggerEvent(TypedDict):
    trigger_type: Literal["risk_score", "customer_report", "analyst_request"]
    trigger_text: str
    flagged_txn_id: str
    card_id: str
    customer_id: str
    risk_score: float | None      # None for non-model triggers
    opened_at: str


# ─────────────────────────────────────────────────────────
# Main state
# ─────────────────────────────────────────────────────────
class InvestigationState(TypedDict):
    # Chat messages for tool interaction
    messages: Annotated[list, add_messages]

    # Trigger
    case_pack_id: str                        # e.g. "HHG-001"
    trigger: TriggerEvent

    # Case identity (populated by agent)
    case_id: str                             # e.g. "CASE-2016-HHG001"

    # Accumulated evidence (list is merged across nodes)
    evidence: Annotated[list[Evidence], operator.add]

    # Graph findings
    affected_txn_ids: list[str]
    first_suspicious_txn_id: str
    connected_card_ids: list[str]
    connected_device_profiles: list[str]
    exposure_usd: float

    # Assessment
    fraud_probability: float                 # 0.0 – 1.0
    confidence: float                        # 0.0 – 1.0 in the assessment
    uncertainty_level: Literal["LOW", "MEDIUM", "HIGH"]
    verdict: Verdict
    pattern: FraudPattern
    pattern_description: str                 # required when pattern == "undocumented"

    # Case memory
    similar_prior_cases: list[str]           # closed case IDs

    # Evidence requests
    evidence_requests: Annotated[list[EvidenceRequest], operator.add]

    # Actions
    initial_actions: list[NextAction]        # before any extra evidence
    final_actions: list[NextAction]          # after assumed responses
    what_changed: str

    # Case progression
    case_status: CaseStatus
    summary: str
    stop_reason: str
    written_to_graph: bool
    graph_case_id: str

    # SAR
    sar_file: bool
    sar_reason: str
    sar_narrative: str
    sar_subjects: list[str]
    sar_total_amount_usd: float
    sar_activity_dates: list[str]

    # Reasoning log (explainability trail)
    reasoning_log: Annotated[list[str], operator.add]

    # Telemetry
    iteration_count: int
    tool_calls: int
    tokens_used: int
    latency_s: float
    started_at: float                        # time.time()


def initial_state(case_row: dict) -> InvestigationState:
    """Build the initial state from a case_pack.csv row."""
    return InvestigationState(
        messages=[],
        case_pack_id=case_row["case_id"],
        trigger=TriggerEvent(
            trigger_type=case_row["trigger_type"],
            trigger_text=case_row["trigger_text"],
            flagged_txn_id=str(case_row["flagged_txn_id"]),
            card_id=str(case_row["card_id"]),
            customer_id=str(case_row["customer_id"]),
            risk_score=float(case_row["risk_score"]) if case_row.get("risk_score") else None,
            opened_at=str(case_row["opened_at"]),
        ),
        case_id="",
        evidence=[],
        affected_txn_ids=[],
        first_suspicious_txn_id="",
        connected_card_ids=[],
        connected_device_profiles=[],
        exposure_usd=0.0,
        fraud_probability=float(case_row["risk_score"]) if case_row.get("risk_score") else 0.5,
        confidence=0.3,
        uncertainty_level="HIGH",
        verdict="uncertain",
        pattern="none",
        pattern_description="",
        similar_prior_cases=[],
        evidence_requests=[],
        initial_actions=[],
        final_actions=[],
        what_changed="nothing",
        case_status="open",
        summary="",
        stop_reason="",
        written_to_graph=False,
        graph_case_id="",
        sar_file=False,
        sar_reason="",
        sar_narrative="",
        sar_subjects=[],
        sar_total_amount_usd=0.0,
        sar_activity_dates=[],
        reasoning_log=[],
        iteration_count=0,
        tool_calls=0,
        tokens_used=0,
        latency_s=0.0,
        started_at=0.0,
    )
