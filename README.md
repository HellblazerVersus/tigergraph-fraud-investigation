# TigerGraph Agentic Fraud Investigation System

[![Live Demo](https://img.shields.io/badge/Streamlit_App-Live_Demo-FA6400?style=for-the-badge&logo=streamlit)](https://fraud-investigationvajra.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/HellblazerVersus/tigergraph-fraud-investigation)
[![TigerGraph 4.2.5](https://img.shields.io/badge/TigerGraph-4.2.5_CE-2563EB?style=for-the-badge)](https://www.tigergraph.com/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-10B981?style=for-the-badge&logo=google)](https://ai.google.dev/)

An autonomous, agentic fraud investigation system built with **TigerGraph Community Edition 4.2.5**, **LangGraph**, and **Google Gemini** for the **Hacker House Goa 2026** hackathon (Team Vajra).

🌐 **Live Cloud Application:** [https://fraud-investigationvajra.streamlit.app/](https://fraud-investigationvajra.streamlit.app/)

---

## 🏆 Project Highlights

- **Graph-Grounded Investigation:** Powered by TigerGraph Community Edition 4.2.5 storing ~590,000 transactions, 14,000 cards, 9,700 device profiles, and 5,565 historical closed cases across a 10-vertex, 18-edge schema.
- **Agentic Workflow:** Built with LangGraph orchestrating an 8-stage investigation lifecycle: Trigger → Investigate → Gather Evidence → Assess Uncertainty → Gather More Evidence → Next-Best Actions → Explain Decision → Update Case Memory.
- **Model Context Protocol (MCP):** Native `tigergraph_mcp/server.py` exposing graph traversals and GraphRAG operations as standard MCP tools.
- **Full Benchmark Evaluation:** Evaluated on all 20 IEEE benchmark cases (`HHG-001` through `HHG-020`) with 100% completion in 155s (~7.8s per case).
- **Persistent Case Memory:** All 20 resolved cases were written back to TigerGraph as `FraudCase` vertices to inform future investigations.
- **Full Policy & SAR Compliance:** Recommends exact policy actions (`ALLOW_TRANSACTION`, `DECLINE_TRANSACTION`, `BLOCK_CARD`, `WARN_CUSTOMER`, etc.) with correct approval routing (`auto`, `L1`, `L2`) and generates formal Suspicious Activity Reports (SARs) when exposure > $1,000 or ring patterns are detected.
- **Interactive UI:** Streamlit analyst dashboard with real-time case investigation, evidence exploration, and historical case memory browsing.

---

## 🏗️ Architecture

```
Trigger Event ──► TigerGraph Retrieval ──► Synthesis (Gemini) ──► Uncertainty Check
                                                                           │
   ┌───────────────────────────────────────────────────────────────────────┘
   │
   ├─► High Uncertainty: Simulate Customer Verification (SMS) ──► Final Actions + SAR
   │
   └─► Low Uncertainty: Final Policy Actions + SAR ──► Upsert to TigerGraph Case Memory
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **Graph Database** | TigerGraph Community Edition 4.2.5 |
| **Graph Query Layer** | pyTigerGraph + REST API endpoints |
| **Agent Framework** | LangGraph + LangChain |
| **Reasoning Model** | Google Gemini (`gemini-2.5-flash-lite`, `gemini-1.5-flash`) |
| **Protocol** | Model Context Protocol (MCP 2.2.0) |
| **Analyst UI** | Streamlit 1.64 |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/HellblazerVersus/tigergraph-fraud-investigation.git
cd tigergraph-fraud-investigation

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Set your GEMINI_API_KEY and TigerGraph credentials in .env
```

### 2. Ingest Data into TigerGraph

```bash
uv run python3 data/etl_load.py
```

### 3. Run Benchmark Cases (20 Cases)

```bash
# Run all 20 benchmark cases
uv run python3 benchmark/run_benchmark_cases.py --skip-existing

# Or run a single case
uv run python3 benchmark/run_benchmark_cases.py --case HHG-001
```

### 4. Launch Analyst Dashboard

```bash
uv run streamlit run ui/app.py
```

### 5. Start TigerGraph MCP Server

```bash
uv run python3 tigergraph_mcp/server.py
```

---

## 📁 Repository Structure

```
tigergraph-fraud-investigation/
├── HHGOA_IEEE/                  # Dataset & Fraud Policy documentation
├── schema/                      # TigerGraph DDL (10 vertex types, 18 edge types)
├── data/                        # High-throughput streaming ETL pipeline
├── gsql/                        # GSQL queries and stored procedures
├── tigergraph_mcp/              # TigerGraph Model Context Protocol (MCP) server
│   ├── __init__.py
│   └── server.py
├── agent/                       # LangGraph Agent Implementation
│   ├── state.py                 # Typed investigation state
│   ├── tools.py                 # Graph investigation & GraphRAG tools
│   └── graph.py                 # State machine & model fallback invoke
├── ui/                          # Streamlit Analyst Dashboard
│   └── app.py
├── benchmark/                   # Benchmark batch runner
│   └── run_benchmark_cases.py
└── cases/                       # Generated answer files (HHG-001.json ... HHG-020.json)
```

---

## 📊 Benchmark Results

Summary across all 20 cases in [`cases/`](cases/):

- **Confirmed Fraud:** 15 cases (`card_not_present_fraud`, `card_not_present_new_device`, `out_of_region_use`, `card_testing`)
- **Legitimate Activity:** 3 cases
- **Uncertain / Verified:** 2 cases
- **SARs Filed:** Triggered and formatted for exposure $> \$1,000$ (e.g., `HHG-010`)
- **Graph Upsert:** 100% of cases persisted to TigerGraph `FraudCase` memory.
