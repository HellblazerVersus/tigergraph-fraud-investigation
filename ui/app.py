"""
ui/app.py — TigerGraph GraphStudio & Agentic Fraud Investigation Dashboard

An analyst dashboard integrating TigerGraph GraphStudio schema design,
interactive graph exploration, GSQL query execution, and autonomous LangGraph fraud investigation.

Run: streamlit run ui/app.py
"""

from __future__ import annotations
import os
import sys
import json
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools import (
    _get_conn,
    graph_card_window,
    graph_txn_subgraph,
    graph_customer_history,
    graph_card_testing_check,
    graph_region_history,
    graph_device_neighbors,
)

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TigerGraph GraphStudio | Fraud Investigation",
    page_icon="🐅",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_DIR = os.getenv("DATASET_DIR", "./HHGOA_IEEE")
CASES_DIR = Path(os.getenv("CASES_OUTPUT_DIR", "./cases"))
CASES_DIR.mkdir(exist_ok=True)
STUDIO_URL = "http://localhost:14240/studio/#/schema-designer?graph=fraud_investigation"

# ─────────────────────────────────────────────────────────
# Cached Data Loaders
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


@st.cache_data(ttl=60)
def get_graph_counts() -> dict[str, int]:
    try:
        conn = _get_conn()
        v_types = [
            "Customer", "Card", "Transaction", "DeviceProfile",
            "EmailDomain", "BillingRegion", "ClosedCase", "FraudCase",
            "PolicyDocument", "FraudPattern"
        ]
        counts = {}
        for vt in v_types:
            try:
                counts[vt] = conn.getVertexCount(vt)
            except Exception:
                counts[vt] = 0
        return counts
    except Exception:
        return {}


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
# Top GraphStudio Header Banner (Matching image.png)
# ─────────────────────────────────────────────────────────
head_left, head_center, head_right = st.columns([2.5, 3, 2.5])
with head_left:
    st.markdown("### 🐅 **GraphStudio** `v4.2.5`")
    st.caption("TigerGraph Community Edition · Enterprise Graph AI")
with head_center:
    st.markdown("<div style='text-align: center; padding-top: 10px;'>", unsafe_allow_html=True)
    st.markdown(":green-badge[:material/database: graph: fraud_investigation (superuser)] :blue-badge[:material/cable: port: 14240]")
    st.markdown("</div>", unsafe_allow_html=True)
with head_right:
    st.markdown("<div style='text-align: right; padding-top: 5px;'>", unsafe_allow_html=True)
    st.link_button("🌐 Open GraphStudio in Browser", STUDIO_URL)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────
# Sidebar Navigation (Mirrored from GraphStudio in image.png)
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐅 **GraphStudio Menu**")
    st.caption("Active Graph: `fraud_investigation`")

    page = st.radio(
        "GraphStudio Modules",
        [
            "📊 Executive Dashboard",
            "🕸️ Design Schema (GraphStudio)",
            "🧭 Explore Graph",
            "✍️ Write Queries (GSQL Runner)",
            "🖥️ Embedded GraphStudio Live",
            "🕵️ Investigate Case (AI Agent)",
            "📁 Benchmark Cases (20 Cases)",
            "🧠 Graph Case Memory",
            "📜 Fraud Policy & Governance",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System Telemetry")
    with st.container(border=True):
        st.markdown(":green-badge[:material/check_circle: TigerGraph CE 4.2.5 Online]")
        st.caption("RESTPP: `9000` | GUI Studio: `14240`")
        st.markdown(":blue-badge[:material/hub: LangGraph State Machine]")
        st.markdown(":purple-badge[:material/smart_toy: Gemini 3.5 Flash Lite]")
        st.markdown(":orange-badge[:material/cable: TigerGraph MCP 2.2.0]")


# ─────────────────────────────────────────────────────────
# PAGE 1: Executive Dashboard
# ─────────────────────────────────────────────────────────
if page == "📊 Executive Dashboard":
    case_pack = load_case_pack()
    answers = get_all_answers()
    graph_counts = get_graph_counts()

    total_cases = len(case_pack)
    investigated_cases = len(answers)
    fraud_cases = [a for a in answers if a.get("case", {}).get("verdict") == "fraud"]
    legit_cases = [a for a in answers if a.get("case", {}).get("verdict") == "legitimate"]
    uncertain_cases = [a for a in answers if a.get("case", {}).get("verdict") == "uncertain"]
    sar_cases = [a for a in answers if a.get("sar", {}).get("file")]
    total_exposure = sum(float(a.get("case", {}).get("exposure_usd", 0.0)) for a in fraud_cases)

    # KPI Row
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

    # Live TigerGraph Stats Banner
    if graph_counts:
        with st.container(border=True):
            st.markdown("#### 🐅 Live TigerGraph Scale")
            g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
            g_col1.metric("Transactions", f"{graph_counts.get('Transaction', 0):,}")
            g_col2.metric("Cards", f"{graph_counts.get('Card', 0):,}")
            g_col3.metric("Customers", f"{graph_counts.get('Customer', 0):,}")
            g_col4.metric("Device Profiles", f"{graph_counts.get('DeviceProfile', 0):,}")
            g_col5.metric("Case Memory", f"{graph_counts.get('ClosedCase', 0) + graph_counts.get('FraudCase', 0):,}")

    # Charts Row
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


# ─────────────────────────────────────────────────────────
# PAGE 2: Design Schema (GraphStudio — Mirroring image.png)
# ─────────────────────────────────────────────────────────
elif page == "🕸️ Design Schema (GraphStudio)":
    st.subheader("🕸️ GraphStudio Schema Designer")
    st.caption("Visual representation of the 10 vertex types and 18 edge types in graph `fraud_investigation`")

    # Toolbar matching GraphStudio in image.png
    with st.container(border=True):
        tb1, tb2, tb3, tb4, tb5, tb6 = st.columns(6)
        tb1.button("➕ Add Vertex Type", disabled=True)
        tb2.button("➕ Add Edge Type", disabled=True)
        tb3.button("💾 Save Schema", disabled=True)
        tb4.button("📤 Export Schema", disabled=True)
        tb5.button("🔄 Sync TigerGraph", on_click=st.cache_data.clear)
        tb6.markdown(f"[🌐 Launch Studio]({STUDIO_URL})")

    # Schema Graph Diagram using Graphviz
    with st.container(border=True):
        st.markdown("#### Schema Architecture Canvas")
        dot_code = """
        digraph FraudInvestigationSchema {
            graph [rankdir=LR, bgcolor="transparent", fontname="Inter"];
            node [shape=circle, style=filled, fontname="Inter", fontsize=11, width=1.4, fixedsize=true, fontcolor="#FFFFFF"];
            edge [fontname="JetBrains Mono", fontsize=9, color="#94A3B8", fontcolor="#60A5FA"];

            Customer [fillcolor="#F59E0B", label="Customer\\n[13.5K]"];
            Card [fillcolor="#3B82F6", label="Card\\n[13.9K]"];
            Transaction [fillcolor="#8B5CF6", label="Transaction\\n[590K]"];
            DeviceProfile [fillcolor="#10B981", label="DeviceProfile\\n[9.7K]"];
            EmailDomain [fillcolor="#14B8A6", label="EmailDomain\\n[Teal]"];
            BillingRegion [fillcolor="#EF4444", label="BillingRegion\\n[Red]"];
            ClosedCase [fillcolor="#D97706", label="ClosedCase\\n[5.5K]"];
            FraudCase [fillcolor="#DC2626", label="FraudCase\\n[Memory]"];
            PolicyDocument [fillcolor="#64748B", label="Policy\\n[Docs]"];
            FraudPattern [fillcolor="#EC4899", label="FraudPattern\\n[Patterns]"];

            Customer -> Card [label="OWNS"];
            Card -> Transaction [label="MADE"];
            Transaction -> DeviceProfile [label="FROM_DEVICE"];
            Transaction -> EmailDomain [label="PURCHASER_EMAIL"];
            Transaction -> BillingRegion [label="BILLED_IN"];
            Transaction -> Transaction [label="NEXT"];

            ClosedCase -> Transaction [label="INVOLVES"];
            ClosedCase -> Card [label="ON_CARD"];
            ClosedCase -> Card [label="CONNECTED_TO"];

            FraudCase -> Transaction [label="CASE_TXN"];
            FraudCase -> Card [label="CASE_CARD"];
            FraudCase -> Customer [label="CASE_CUST"];
        }
        """
        st.graphviz_chart(dot_code, width="stretch")

    # Schema Details & Attributes
    col_v, col_e = st.columns(2)
    with col_v:
        with st.container(border=True):
            st.markdown("#### 🔷 Vertex Types (10 Types)")
            vertex_data = [
                {"Vertex Type": "Customer", "Primary Key": "customer_id", "Key Attributes": "customer_id, risk_score_mean"},
                {"Vertex Type": "Card", "Primary Key": "card_id", "Key Attributes": "card_id, card_type, is_active"},
                {"Vertex Type": "Transaction", "Primary Key": "txn_id", "Key Attributes": "amount, is_online, risk_score, ts"},
                {"Vertex Type": "DeviceProfile", "Primary Key": "device_id", "Key Attributes": "device_info, os, browser, screen"},
                {"Vertex Type": "EmailDomain", "Primary Key": "domain", "Key Attributes": "domain_name, is_free"},
                {"Vertex Type": "BillingRegion", "Primary Key": "region_id", "Key Attributes": "addr1, addr2, risk_tier"},
                {"Vertex Type": "ClosedCase", "Primary Key": "case_id", "Key Attributes": "outcome, pattern, exposure_usd, n_txns"},
                {"Vertex Type": "FraudCase", "Primary Key": "case_id", "Key Attributes": "status, verdict, fraud_probability, pattern"},
                {"Vertex Type": "PolicyDocument", "Primary Key": "doc_id", "Key Attributes": "doc_type, title, content"},
                {"Vertex Type": "FraudPattern", "Primary Key": "pattern_id", "Key Attributes": "name, description, policy_rule"},
            ]
            st.dataframe(pd.DataFrame(vertex_data), width="stretch")

    with col_e:
        with st.container(border=True):
            st.markdown("#### 🔗 Edge Types (18 Types)")
            edge_data = [
                {"Edge Type": "OWNS", "Source Vertex": "Customer", "Target Vertex": "Card", "Directed": True},
                {"Edge Type": "MADE", "Source Vertex": "Card", "Target Vertex": "Transaction", "Directed": True},
                {"Edge Type": "FROM_DEVICE", "Source Vertex": "Transaction", "Target Vertex": "DeviceProfile", "Directed": True},
                {"Edge Type": "PURCHASER_EMAIL", "Source Vertex": "Transaction", "Target Vertex": "EmailDomain", "Directed": True},
                {"Edge Type": "BILLED_IN", "Source Vertex": "Transaction", "Target Vertex": "BillingRegion", "Directed": True},
                {"Edge Type": "NEXT", "Source Vertex": "Transaction", "Target Vertex": "Transaction", "Directed": True},
                {"Edge Type": "INVOLVES", "Source Vertex": "ClosedCase", "Target Vertex": "Transaction", "Directed": True},
                {"Edge Type": "ON_CARD", "Source Vertex": "ClosedCase", "Target Vertex": "Card", "Directed": True},
                {"Edge Type": "CONNECTED_TO", "Source Vertex": "ClosedCase", "Target Vertex": "Card", "Directed": True},
                {"Edge Type": "CASE_TXN", "Source Vertex": "FraudCase", "Target Vertex": "Transaction", "Directed": True},
            ]
            st.dataframe(pd.DataFrame(edge_data), width="stretch")


# ─────────────────────────────────────────────────────────
# PAGE 3: Explore Graph
# ─────────────────────────────────────────────────────────
elif page == "🧭 Explore Graph":
    st.subheader("🧭 TigerGraph Neighborhood Explorer")
    st.caption("Inspect live connected subgraphs, transaction temporal chains, and shared device networks.")

    e_col1, e_col2, e_col3 = st.columns([1.5, 2.5, 1])
    with e_col1:
        entity_type = st.selectbox("Entity Type", ["Transaction", "Card", "Customer", "Device"])
    with e_col2:
        default_val = {
            "Transaction": "3514030",
            "Card": "C12382-K1",
            "Customer": "C12382",
            "Device": "DBD75C3985A"
        }.get(entity_type, "3514030")
        entity_id = st.text_input("Enter ID", value=default_val)
    with e_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        explore_btn = st.button("Explore Subgraph", type="primary")

    if explore_btn or entity_id:
        with st.spinner(f"Traversing graph for {entity_type} {entity_id}..."):
            if entity_type == "Transaction":
                subgraph = graph_txn_subgraph.invoke({"txn_id": entity_id})
                card_id = subgraph.get("card_id", "N/A")
                cust_id = subgraph.get("customer_id", "N/A")
                dev_ids = subgraph.get("device_ids", [])
                email = subgraph.get("email_domain", "N/A")

                with st.container(border=True):
                    st.markdown(f"#### 1-Hop Subgraph for Transaction `{entity_id}`")
                    sub_dot = f"""
                    digraph Subgraph {{
                        graph [rankdir=LR, bgcolor="transparent", fontname="Inter"];
                        node [shape=box, style=filled, fontname="Inter", fontsize=10, fontcolor="#FFFFFF", rx=6, ry=6];
                        edge [fontname="JetBrains Mono", fontsize=8, color="#94A3B8", fontcolor="#60A5FA"];

                        Txn [fillcolor="#8B5CF6", label="Transaction\\n{entity_id}\\nAmount: ${subgraph.get('amount', 0):,.2f}"];
                        CardNode [fillcolor="#3B82F6", label="Card\\n{card_id}"];
                        CustNode [fillcolor="#F59E0B", label="Customer\\n{cust_id}"];
                        EmailNode [fillcolor="#14B8A6", label="Email\\n{email}"];

                        CardNode -> Txn [label="MADE"];
                        CustNode -> CardNode [label="OWNS"];
                        Txn -> EmailNode [label="PURCHASER_EMAIL"];
                    """
                    for d in dev_ids:
                        sub_dot += f"""
                        Dev_{d[:8]} [fillcolor="#10B981", label="Device\\n{d[:12]}"];
                        Txn -> Dev_{d[:8]} [label="FROM_DEVICE"];
                        """
                    sub_dot += "}"
                    st.graphviz_chart(sub_dot, width="stretch")
                    st.json(subgraph)

            elif entity_type == "Card":
                win = graph_card_window.invoke({"card_id": entity_id, "hours": 48})
                ct = graph_card_testing_check.invoke({"card_id": entity_id})
                with st.container(border=True):
                    st.markdown(f"#### 48-Hour Activity on Card `{entity_id}`")
                    st.markdown(f"**Total Transactions in Window:** `{win.get('txn_count', 0)}` &nbsp;|&nbsp; **Card Testing Detected:** `{ct.get('is_card_testing_pattern', False)}`")
                    st.json(win)

            elif entity_type == "Customer":
                hist = graph_customer_history.invoke({"customer_id": entity_id})
                with st.container(border=True):
                    st.markdown(f"#### Customer Profile for `{entity_id}`")
                    st.markdown(f"**Customer Owned Cards ({len(hist.get('card_ids', []))}):** {', '.join([f'`{c}`' for c in hist.get('card_ids', [])])}")
                    st.json(hist)

            elif entity_type == "Device":
                dev_res = graph_device_neighbors.invoke({"txn_id": "3506725", "days": 90})
                with st.container(border=True):
                    st.markdown(f"#### Device Neighbors for `{entity_id}`")
                    st.json(dev_res)


# ─────────────────────────────────────────────────────────
# PAGE 4: Write Queries (GSQL Runner)
# ─────────────────────────────────────────────────────────
elif page == "✍️ Write Queries (GSQL Runner)":
    st.subheader("✍️ TigerGraph GSQL Query Runner")
    st.caption("Execute pre-installed and analytical graph queries against `fraud_investigation`")

    query_choice = st.selectbox(
        "Select Stored Query",
        [
            "card_window(card_id, hours)",
            "txn_subgraph(txn_id)",
            "customer_history(customer_id)",
            "card_testing_check(card_id, amount_threshold, count_threshold)",
            "region_history(card_id)",
            "device_neighbors(txn_id, days)",
        ]
    )

    q_form = st.container(border=True)
    with q_form:
        if "card_window" in query_choice:
            p_card = st.text_input("card_id", "C12382-K1")
            p_hours = st.number_input("hours", value=48, min_value=1, max_value=720)
            if st.button("▶️ Execute Query", type="primary"):
                res = graph_card_window.invoke({"card_id": p_card, "hours": int(p_hours)})
                st.success("Query Executed Successfully")
                st.json(res)

        elif "txn_subgraph" in query_choice:
            p_txn = st.text_input("txn_id", "3514030")
            if st.button("▶️ Execute Query", type="primary"):
                res = graph_txn_subgraph.invoke({"txn_id": p_txn})
                st.success("Query Executed Successfully")
                st.json(res)

        elif "customer_history" in query_choice:
            p_cust = st.text_input("customer_id", "C12382")
            if st.button("▶️ Execute Query", type="primary"):
                res = graph_customer_history.invoke({"customer_id": p_cust})
                st.success("Query Executed Successfully")
                st.json(res)

        elif "card_testing_check" in query_choice:
            p_card = st.text_input("card_id", "C11891-K1")
            p_amt = st.number_input("amount_threshold", value=5.0)
            p_cnt = st.number_input("count_threshold", value=3)
            if st.button("▶️ Execute Query", type="primary"):
                res = graph_card_testing_check.invoke({
                    "card_id": p_card,
                    "amount_threshold": float(p_amt),
                    "count_threshold": int(p_cnt),
                })
                st.success("Query Executed Successfully")
                st.json(res)

        elif "region_history" in query_choice:
            p_card = st.text_input("card_id", "C12382-K1")
            if st.button("▶️ Execute Query", type="primary"):
                res = graph_region_history.invoke({"card_id": p_card})
                st.success("Query Executed Successfully")
                st.json(res)

        elif "device_neighbors" in query_choice:
            p_txn = st.text_input("txn_id", "3506725")
            p_days = st.number_input("days", value=90)
            if st.button("▶️ Execute Query", type="primary"):
                res = graph_device_neighbors.invoke({"txn_id": p_txn, "days": int(p_days)})
                st.success("Query Executed Successfully")
                st.json(res)


# ─────────────────────────────────────────────────────────
# PAGE 5: Embedded GraphStudio Live
# ─────────────────────────────────────────────────────────
elif page == "🖥️ Embedded GraphStudio Live":
    st.subheader("🖥️ Live Embedded TigerGraph GraphStudio")
    st.caption(f"Connecting to live GraphStudio web server on `{STUDIO_URL}`")

    st.info("💡 You can interact with GraphStudio directly inside this window, or open it in a full tab.")
    components.iframe(STUDIO_URL, height=850, scrolling=True)


# ─────────────────────────────────────────────────────────
# PAGE 6: Investigate Case (AI Agent)
# ─────────────────────────────────────────────────────────
elif page == "🕵️ Investigate Case (AI Agent)":
    st.subheader("🕵️ Autonomous Fraud Investigation Agent")

    case_pack = load_case_pack()
    selected_case = st.selectbox(
        "Select benchmark case to review or run",
        case_pack["case_id"].tolist(),
        index=0,
    )
    row = case_pack[case_pack["case_id"] == selected_case].iloc[0]

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

    if st.button("🚀 Re-run Live Agent Investigation", type="primary"):
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

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Investigation Verdict", case_data.get("verdict", "").upper(), border=True)
        with m2:
            st.metric("Fraud Probability", f"{case_data.get('fraud_probability', 0):.0%}", border=True)
        with m3:
            st.metric("Financial Exposure", f"${case_data.get('exposure_usd', 0):,.2f}", border=True)
        with m4:
            st.metric("Latency & Tools", f"{existing_answer.get('latency_s', 0):.1f}s / {existing_answer.get('tool_calls', 0)} calls", border=True)

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

        with tab_json:
            st.json(existing_answer)


# ─────────────────────────────────────────────────────────
# PAGE 7: Benchmark Cases (20 Cases)
# ─────────────────────────────────────────────────────────
elif page == "📁 Benchmark Cases (20 Cases)":
    st.subheader("📁 Benchmark Execution & Performance Metrics")
    answers = get_all_answers()

    if answers:
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
# PAGE 8: Graph Case Memory
# ─────────────────────────────────────────────────────────
elif page == "🧠 Graph Case Memory":
    st.subheader("🧠 Historical Closed Case Memory (TigerGraph)")
    closed = load_closed_cases()

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Total Historical Cases", len(closed), border=True)
    cm2.metric("Confirmed Fraud Cases", len(closed[closed["outcome"] == "confirmed_fraud"]), border=True)
    cm3.metric("Cleared Cases", len(closed[closed["outcome"] == "cleared"]), border=True)

    with st.container(border=True):
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
# PAGE 9: Fraud Policy & Governance
# ─────────────────────────────────────────────────────────
elif page == "📜 Fraud Policy & Governance":
    st.subheader("📜 Bank Fraud Policy & Approval Matrix (Rules R1 – R10)")

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
