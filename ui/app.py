"""
ui/app.py — Streamlit Analyst Dashboard for HHGOA Fraud Investigation Agent

Run: streamlit run ui/app.py
"""

from __future__ import annotations
import os
import sys
import json
import time
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TigerGraph Fraud Investigation | HHGOA",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_DIR = os.getenv("DATASET_DIR", "./HHGOA_IEEE")
CASES_DIR = Path(os.getenv("CASES_OUTPUT_DIR", "./cases"))
CASES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────
# Data Loading & Helpers
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_case_pack() -> pd.DataFrame:
    df = pd.read_csv(
        f"{DATASET_DIR}/case_pack.csv",
        dtype={"flagged_txn_id": str, "card_id": str, "customer_id": str},
    )
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    return df


@st.cache_data
def load_closed_cases() -> pd.DataFrame:
    return pd.read_csv(f"{DATASET_DIR}/closed_cases_history.csv", dtype=str)


def load_answer_file(case_id: str) -> dict | None:
    path = CASES_DIR / f"{case_id}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_all_answers() -> list[dict]:
    answers = []
    for path in sorted(CASES_DIR.glob("HHG-*.json")):
        try:
            with open(path) as f:
                answers.append(json.load(f))
        except Exception:
            continue
    return answers


PATTERN_LABELS = {
    "card_testing": "Card Testing",
    "card_not_present_fraud": "Card-Not-Present (CNP)",
    "card_not_present_new_device": "CNP New Device",
    "out_of_region_use": "Out-of-Region Use",
    "account_takeover": "Account Takeover",
    "undocumented": "Undocumented Pattern",
    "none": "Legitimate / None",
}

ROUTE_COLORS = {
    "auto": "green",
    "L1": "orange",
    "L2": "red",
}

# ─────────────────────────────────────────────────────────
# Sidebar Navigation & System Telemetry
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### :material/shield: Fraud Investigation")
    st.caption("TigerGraph Autonomous Agent · HHGOA 2026")

    page = st.radio(
        "Navigation",
        [
            "Overview Dashboard",
            "Investigate Case",
            "Benchmark Cases",
            "Graph Case Memory",
            "Fraud Policy & Governance",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System status")
    with st.container(border=True):
        st.markdown(":green-badge[:material/check_circle: TigerGraph CE 4.2.5 Online]")
        st.caption("Host: `localhost:9000` | Graph: `fraud_investigation`")
        st.markdown(":blue-badge[:material/hub: LangGraph State Machine]")
        st.markdown(":purple-badge[:material/smart_toy: Gemini 3.5 Flash Lite]")
        st.markdown(":orange-badge[:material/cable: TigerGraph MCP 2.2.0]")

# ─────────────────────────────────────────────────────────
# Top Header Banner
# ─────────────────────────────────────────────────────────
col_head, col_badge = st.columns([3, 1])
with col_head:
    st.title("TigerGraph Agentic Fraud Investigation")
    st.caption("Autonomous multi-hop graph investigation, uncertainty quantification, and policy-governed next-best actions.")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.badge("Hacker House Goa 2026", icon=":material/hotel_class:", color="blue")


# ─────────────────────────────────────────────────────────
# PAGE 1: Overview Dashboard
# ─────────────────────────────────────────────────────────
if page == "Overview Dashboard":
    case_pack = load_case_pack()
    answers = get_all_answers()

    total_cases = len(case_pack)
    investigated_cases = len(answers)
    fraud_cases = [a for a in answers if a.get("case", {}).get("verdict") == "fraud"]
    legit_cases = [a for a in answers if a.get("case", {}).get("verdict") == "legitimate"]
    uncertain_cases = [a for a in answers if a.get("case", {}).get("verdict") == "uncertain"]
    sar_cases = [a for a in answers if a.get("sar", {}).get("file")]
    total_exposure = sum(float(a.get("case", {}).get("exposure_usd", 0.0)) for a in fraud_cases)

    # KPI Row with border cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Cases", total_cases, border=True)
    with k2:
        st.metric("Investigated", f"{investigated_cases}/{total_cases}", delta="100% complete", border=True)
    with k3:
        st.metric("Confirmed Fraud", len(fraud_cases), delta=f"{len(fraud_cases)/max(investigated_cases, 1):.0%}", border=True)
    with k4:
        st.metric("SARs Filed", len(sar_cases), border=True)
    with k5:
        st.metric("Identified Exposure", f"${total_exposure:,.2f}", border=True)

    # Chart Section
    c1, c2 = st.columns([2, 1])
    with c1:
        with st.container(border=True):
            st.subheader("Identified Fraud Patterns Across Benchmark Cases")
            if answers:
                pattern_series = pd.Series([
                    PATTERN_LABELS.get(a.get("case", {}).get("pattern", "none"), a.get("case", {}).get("pattern", "none"))
                    for a in answers
                ]).value_counts()
                st.bar_chart(pattern_series, color="#60A5FA")
            else:
                st.info("No investigated cases found.")

    with c2:
        with st.container(border=True):
            st.subheader("Verdict Distribution")
            if answers:
                verdict_data = pd.DataFrame({
                    "Verdict": ["Fraud", "Legitimate", "Uncertain"],
                    "Count": [len(fraud_cases), len(legit_cases), len(uncertain_cases)]
                }).set_index("Verdict")
                st.bar_chart(verdict_data, color="#34D399")
            else:
                st.info("No data yet.")

    # Benchmark Progress Table
    with st.container(border=True):
        st.subheader("Benchmark Cases Status")
        ans_map = {a["case_id"]: a for a in answers}

        table_rows = []
        for _, row in case_pack.iterrows():
            cid = row["case_id"]
            ans = ans_map.get(cid)
            if ans:
                c = ans.get("case", {})
                verdict = c.get("verdict", "").upper()
                prob = f"{c.get('fraud_probability', 0):.0%}"
                pat = PATTERN_LABELS.get(c.get("pattern", ""), c.get("pattern", ""))
                exp = f"${c.get('exposure_usd', 0):,.2f}"
                sar_flag = "Filed (L2)" if ans.get("sar", {}).get("file") else "Not Required"
                status_badge = "Complete"
                mem = "Persisted" if c.get("written_to_graph") else "—"
            else:
                verdict = "—"
                prob = "—"
                pat = "—"
                exp = "—"
                sar_flag = "—"
                status_badge = "Pending"
                mem = "—"

            table_rows.append({
                "Case ID": cid,
                "Trigger Type": row["trigger_type"],
                "Model Risk Score": f"{row['risk_score']:.2f}" if pd.notna(row["risk_score"]) else "N/A",
                "Card ID": row["card_id"],
                "Customer ID": row["customer_id"],
                "Verdict": verdict,
                "Fraud Prob": prob,
                "Pattern": pat,
                "Exposure": exp,
                "SAR": sar_flag,
                "Graph Memory": mem,
                "Status": status_badge,
            })

        st.dataframe(pd.DataFrame(table_rows), width="stretch")


# ─────────────────────────────────────────────────────────
# PAGE 2: Investigate Case
# ─────────────────────────────────────────────────────────
elif page == "Investigate Case":
    st.subheader("Interactive Case Investigation")

    case_pack = load_case_pack()
    selected_case = st.selectbox(
        "Select benchmark case to review or run",
        case_pack["case_id"].tolist(),
        index=0,
    )
    row = case_pack[case_pack["case_id"] == selected_case].iloc[0]

    # Trigger Information Card
    with st.container(border=True):
        t1, t2, t3, t4 = st.columns(4)
        t1.markdown(f"**Case ID:** `{row['case_id']}`")
        t2.markdown(f"**Trigger Type:** `{row['trigger_type']}`")
        t3.markdown(f"**Card ID:** `{row['card_id']}`")
        t4.markdown(f"**Customer ID:** `{row['customer_id']}`")

        r_score = f"{row['risk_score']:.2f}" if pd.notna(row["risk_score"]) else "N/A (dispute report)"
        st.markdown(f"**Flagged Transaction ID:** `{row['flagged_txn_id']}` &nbsp;|&nbsp; **Model Risk Score:** `{r_score}`")
        st.info(f"**Trigger Context:** {row['trigger_text']}")

    existing_answer = load_answer_file(selected_case)

    # Action Toolbar
    act_col1, act_col2 = st.columns([1, 4])
    with act_col1:
        run_btn = st.button("🚀 Re-run Live Investigation", type="primary")

    if run_btn:
        from agent.graph import run_investigation
        from benchmark.run_benchmark_cases import state_to_answer

        with st.status("Executing Autonomous Agent Investigation...", expanded=True) as status_box:
            st.write("1. Connecting to TigerGraph REST API...")
            t0 = time.time()
            final_state = run_investigation(row.to_dict())
            elapsed = time.time() - t0

            st.write("2. Synthesizing graph evidence with Gemini reasoning...")
            st.write("3. Checking policy compliance & approval matrix...")
            st.write("4. Upserting resolved case to TigerGraph FraudCase vertex...")

            existing_answer = state_to_answer(final_state)
            with open(CASES_DIR / f"{selected_case}.json", "w") as f:
                json.dump(existing_answer, f, indent=2)

            status_box.update(label=f"Investigation Complete in {elapsed:.1f}s!", state="complete", expanded=False)

    if existing_answer:
        case_data = existing_answer.get("case", {})
        nba_data = existing_answer.get("next_best_actions", {})
        sar_data = existing_answer.get("sar", {})

        # Verdict Header Metrics
        m1, m2, m3, m4 = st.columns(4)
        v_color = "red" if case_data.get("verdict") == "fraud" else ("green" if case_data.get("verdict") == "legitimate" else "orange")
        with m1:
            st.metric("Investigation Verdict", case_data.get("verdict", "").upper(), border=True)
        with m2:
            st.metric("Fraud Probability", f"{case_data.get('fraud_probability', 0):.0%}", border=True)
        with m3:
            st.metric("Financial Exposure", f"${case_data.get('exposure_usd', 0):,.2f}", border=True)
        with m4:
            st.metric("Latency & Tools", f"{existing_answer.get('latency_s', 0):.1f}s / {existing_answer.get('tool_calls', 0)} calls", border=True)

        # Tabbed Investigation Details
        tab_summary, tab_evidence, tab_actions, tab_sar, tab_memory, tab_json = st.tabs([
            "Case Summary",
            "Graph Evidence Chain",
            "Next-Best Actions",
            "Regulatory SAR Filing",
            "TigerGraph Memory",
            "Schema JSON",
        ])

        with tab_summary:
            with st.container(border=True):
                st.markdown(f"#### Pattern: {PATTERN_LABELS.get(case_data.get('pattern', 'none'), case_data.get('pattern', 'none'))}")
                if case_data.get("pattern_description"):
                    st.warning(f"**Undocumented Pattern Detail:** {case_data['pattern_description']}")
                st.markdown(f"**Analyst Summary:**\n\n{case_data.get('summary', '')}")
                st.markdown(f"**Investigation Stopping Reason:**\n\n_{existing_answer.get('stop_reason', '')}_")

                aff_txns = case_data.get("affected_txn_ids", [])
                if aff_txns:
                    st.markdown(f"**Affected Transactions ({len(aff_txns)}):** {', '.join([f'`{x}`' for x in aff_txns])}")
                conn_cards = case_data.get("connected_card_ids", [])
                if conn_cards:
                    st.markdown(f"**Connected Compromised Cards:** {', '.join([f'`{x}`' for x in conn_cards])}")

        with tab_evidence:
            st.markdown("#### Evidence Items Grounded from Knowledge Graph & Policy")
            ev_list = case_data.get("evidence", [])
            if not ev_list:
                st.info("No evidence items attached.")
            for i, ev in enumerate(ev_list, 1):
                with st.container(border=True):
                    src = ev.get("source", "graph").upper()
                    src_badge = ":blue-badge[GRAPH]" if src == "GRAPH" else (":green-badge[CUSTOMER]" if src == "CUSTOMER" else ":purple-badge[DOCUMENT]")
                    st.markdown(f"{src_badge} **Item {i}:** {ev.get('claim', '')}")
                    st.caption(f"Source Reference: `{ev.get('ref', '')}` | Entity IDs: {', '.join(ev.get('entity_ids', []))}")

        with tab_actions:
            st.markdown("#### Dual-Phase Next-Best Actions (Before vs. After Evidence)")
            col_init, col_final = st.columns(2)
            with col_init:
                with st.container(border=True):
                    st.markdown("##### Initial Recommendations (Before evidence)")
                    for act in nba_data.get("initial", []):
                        r_col = ROUTE_COLORS.get(act.get("route", "auto"), "blue")
                        st.markdown(f":{r_col}-badge[{act.get('route', 'auto').upper()}] **{act.get('action')}**")
                        st.caption(f"Rule Citation: {act.get('reason', '')}")
            with col_final:
                with st.container(border=True):
                    st.markdown("##### Final Recommendations (After evidence)")
                    for act in nba_data.get("final", []):
                        r_col = ROUTE_COLORS.get(act.get("route", "auto"), "blue")
                        st.markdown(f":{r_col}-badge[{act.get('route', 'auto').upper()}] **{act.get('action')}**")
                        st.caption(f"Rule Citation: {act.get('reason', '')}")

            if nba_data.get("what_changed"):
                st.info(f"**Why Actions Changed:** {nba_data['what_changed']}")

        with tab_sar:
            if sar_data.get("file"):
                st.error("### :material/warning: Suspicious Activity Report (SAR) — L2 Escalation Required")
                with st.container(border=True):
                    st.markdown(f"**Filing Criteria:** {sar_data.get('reason', '')}")
                    st.markdown(f"**Total Suspicious Exposure:** `${sar_data.get('total_amount_usd', 0):,.2f}`")
                    st.markdown(f"**Activity Dates:** {' to '.join(sar_data.get('activity_dates', []))}")
                    st.markdown(f"**Named Subjects:** {', '.join(sar_data.get('subjects', []))}")
                    st.markdown("---")
                    st.markdown("##### Regulatory Narrative")
                    st.markdown(f"> {sar_data.get('narrative', '')}")
            else:
                st.success("### :material/verified: No Regulatory SAR Required")
                st.caption(f"Reason: {sar_data.get('reason', 'Policy exposure and ring thresholds were not met.')}")

        with tab_memory:
            with st.container(border=True):
                st.markdown("#### TigerGraph Case Memory Persistence")
                if case_data.get("written_to_graph"):
                    st.markdown(f":green-badge[:material/check: Persisted to Graph] Vertex ID: `{case_data.get('graph_case_id')}`")
                    st.caption("This case is now actively queryable by all future investigations as prior case memory.")
                else:
                    st.markdown(":orange-badge[Not written to graph]")

                prior_cases = case_data.get("similar_prior_cases", [])
                if prior_cases:
                    st.markdown(f"**Similar Prior Cases Retrieved from Memory:** {', '.join([f'`{c}`' for c in prior_cases])}")
                else:
                    st.caption("No identical historical patterns retrieved for this episode.")

        with tab_json:
            st.json(existing_answer)


# ─────────────────────────────────────────────────────────
# PAGE 3: Benchmark Cases
# ─────────────────────────────────────────────────────────
elif page == "Benchmark Cases":
    st.subheader("Benchmark Execution & Performance Metrics")
    answers = get_all_answers()

    if not answers:
        st.warning("No answer files found in cases/. Run benchmark/run_benchmark_cases.py first.")
    else:
        # Latency & Tool Calls Metrics
        latencies = [a.get("latency_s", 0) for a in answers]
        tool_counts = [a.get("tool_calls", 0) for a in answers]

        b1, b2, b3 = st.columns(3)
        b1.metric("Average Latency", f"{sum(latencies)/len(latencies):.2f}s", border=True)
        b2.metric("Min / Max Latency", f"{min(latencies):.1f}s / {max(latencies):.1f}s", border=True)
        b3.metric("Average Tool Invocations", f"{sum(tool_counts)/len(tool_counts):.1f} calls", border=True)

        rows = []
        for a in answers:
            c = a.get("case", {})
            rows.append({
                "Case ID": a["case_id"],
                "Verdict": c.get("verdict", "").upper(),
                "Pattern": PATTERN_LABELS.get(c.get("pattern", ""), c.get("pattern", "")),
                "Probability": f"{c.get('fraud_probability', 0):.0%}",
                "Exposure": f"${c.get('exposure_usd', 0):,.2f}",
                "SAR": "YES" if a.get("sar", {}).get("file") else "NO",
                "Graph Memory": "YES" if c.get("written_to_graph") else "NO",
                "Tool Calls": a.get("tool_calls", 0),
                "Latency": f"{a.get('latency_s', 0):.1f}s",
            })

        st.dataframe(pd.DataFrame(rows), width="stretch")


# ─────────────────────────────────────────────────────────
# PAGE 4: Graph Case Memory
# ─────────────────────────────────────────────────────────
elif page == "Graph Case Memory":
    st.subheader("Historical Closed Case Memory (TigerGraph)")
    closed = load_closed_cases()

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Total Historical Cases", len(closed), border=True)
    confirmed_count = len(closed[closed["outcome"] == "confirmed_fraud"])
    cm2.metric("Confirmed Fraud Cases", confirmed_count, border=True)
    cleared_count = len(closed[closed["outcome"] == "cleared"])
    cm3.metric("Cleared Cases", cleared_count, border=True)

    with st.container(border=True):
        st.subheader("Historical Fraud Pattern Distribution")
        p_counts = closed[closed["outcome"] == "confirmed_fraud"]["pattern"].value_counts()
        st.bar_chart(p_counts, color="#60A5FA")

    search_query = st.text_input("Filter historical cases (by card, customer, or pattern)", placeholder="e.g. C10434 or card_testing")
    df_filtered = closed.copy()
    if search_query:
        mask = df_filtered.astype(str).apply(lambda col: col.str.contains(search_query, case=False)).any(axis=1)
        df_filtered = df_filtered[mask]

    st.dataframe(
        df_filtered[["case_id", "customer_id", "card_id", "outcome", "pattern", "exposure_usd", "n_txns", "opened_at", "closed_at"]].head(250),
        width="stretch",
    )


# ─────────────────────────────────────────────────────────
# PAGE 5: Fraud Policy & Governance
# ─────────────────────────────────────────────────────────
elif page == "Fraud Policy & Governance":
    st.subheader("Bank Fraud Policy & Approval Matrix (Rules R1 – R10)")

    with st.container(border=True):
        st.markdown("#### Governance & Approval Routing Tiers")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(":green-badge[AUTO APPROVAL]")
            st.markdown("""
            - `ALLOW_TRANSACTION`
            - `MONITOR_CARD`
            - `MONITOR_CONNECTED_CARDS`
            - `WARN_CUSTOMER`
            - `VERIFY_WITH_CUSTOMER`
            - `STEP_UP_AUTH`
            - `CREATE_CASE`
            - `CLOSE_NO_FRAUD`
            """)
        with g2:
            st.markdown(":orange-badge[L1 TEAM LEAD]")
            st.markdown(r"""
            - `DECLINE_TRANSACTION`
            - `BLOCK_CARD` (if exposure $\le \$2,500$)
            """)
        with g3:
            st.markdown(":red-badge[L2 FRAUD MANAGER]")
            st.markdown(r"""
            - `BLOCK_CARD` (if exposure $> \$2,500$)
            - `BLOCK_ALL_CARDS` (always)
            - `FILE_REPORT` (SAR filing, always)
            """)

    with st.container(border=True):
        st.markdown("#### Core Fraud Policy Rules")
        st.markdown(r"""
        - **Rule R1 (Verify before block):** If case rests on a single signal and fraud probability $< 0.70$, recommend `VERIFY_WITH_CUSTOMER` or `STEP_UP_AUTH`.
        - **Rule R2 (Customer denial):** If customer denies transaction, recommend `BLOCK_CARD` and `CREATE_CASE`. Add `FILE_REPORT` if exposure $> \$1,000$.
        - **Rule R3 (Customer confirmation):** If customer confirms transaction, recommend `CLOSE_NO_FRAUD`.
        - **Rule R4 (Unresponsive customer):** Recommend `MONITOR_CARD` and `DECLINE_TRANSACTION`. Escalate if exposure $> \$500$.
        - **Rule R5 (Card testing):** 3+ small authorizations ($< \$5$) within an hour followed by larger purchase: recommend `DECLINE_TRANSACTION` and `STEP_UP_AUTH`.
        - **Rule R6 (Shared origin / Ring):** Shared device profile, billing region, or email across cards: recommend `CREATE_CASE`, `FILE_REPORT`, and `MONITOR_CONNECTED_CARDS`.
        - **Rule R7 (Disputed recurring):** Recurring charge matching history: recommend `VERIFY_WITH_CUSTOMER` and `WARN_CUSTOMER`. Do not block.
        - **Rule R8 (Escalate uncertain):** If verdict is `uncertain` and exposure $> \$500$, recommend `ESCALATE_TO_ANALYST`.
        - **Rule R9 (Undocumented pattern):** Abuse fitting none of the 5 known categories: recommend `CREATE_CASE`, `FILE_REPORT`, and `ESCALATE_TO_ANALYST` with custom description.
        - **Rule R10 (Block all cards restraint):** Never `BLOCK_ALL_CARDS` unless $\ge 2$ customer cards show confirmed fraud.
        """)
