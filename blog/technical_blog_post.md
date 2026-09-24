# Autonomous Agentic Fraud Investigation with TigerGraph and GraphRAG

**Hacker House Goa 2026 Hackathon Submission**

---

## 1. What We Built

Financial fraud teams face a fundamental bottleneck: investigating a flagged transaction requires an analyst to manually traverse fragmented data systems—querying past transactions, tracing money flow, correlating shared device footprints, verifying regional activity, referencing past cases, and adhering to strict compliance rules. This process takes minutes to hours, often completing only after fraudulent funds have already departed the institution.

We built an **Autonomous Agentic Fraud Investigation System** powered by **TigerGraph Community Edition**, **LangGraph**, and **Google Gemini Reasoning Models**. The agent:
1. Ingests real-world transaction triggers (model anomaly scores, customer dispute reports, analyst flags).
2. Deeply investigates connected entities across a 10-vertex, 18-edge TigerGraph knowledge graph containing ~590,000 transactions, 14,000 cards, and 9,700 device profiles.
3. Quantifies uncertainty and executes policy-compliant next-best actions (`ALLOW_TRANSACTION`, `DECLINE_TRANSACTION`, `BLOCK_CARD`, `STEP_UP_AUTH`, etc.) with correct approval routing (`auto`, `L1`, `L2`).
4. Generates regulatory-grade Suspicious Activity Reports (SARs) when thresholds or organized ring patterns are detected.
5. Closes the learning loop by writing newly resolved cases back into TigerGraph case memory as `FraudCase` vertices to inform all future investigations.

---

## 2. Architecture Overview

```
                                  ┌────────────────────────┐
                                  │   Case Trigger Event   │
                                  │ (Score / Dispute / UI) │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  node_investigate_tg   │
                                  │  (Deterministic Graph) │
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌─────────────────────────┐
         │ TigerGraph DB (CE 4.2)  │                     │  Bank Policy Documents  │
         │ - 48h Card Window       │                     │  - Rules R1 to R10      │
         │ - 1-Hop Txn Subgraph    │                     │  - Approval Routes      │
         │ - Device Sharing Mesh   │                     │  - SAR Criteria         │
         │ - Region Distribution   │                     └────────────┬────────────┘
         │ - Card Testing Detector │                                  │
         │ - 5,565 Closed Cases    │                                  │
         └────────────┬────────────┘                                  │
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │    node_synthesize     │
                                  │   (Gemini Reasoning)   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │node_assess_uncertainty │
                                  └───────┬────────┬───────┘
                     High Uncertainty     │        │ Low Uncertainty
                                          ▼        │
                           ┌─────────────────────┐ │
                           │node_gather_evidence │ │
                           │ (Simulate Customer) │ │
                           └──────────────┬──────┘ │
                                          │        │
                                          ▼        ▼
                                  ┌────────────────────────┐
                                  │ node_recommend_action  │
                                  │ (Final Actions + SAR)  │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │node_update_case_memory │
                                  │ (Upsert to TigerGraph) │
                                  └────────────────────────┘
```

---

## 3. How TigerGraph is Used

Traditional relational databases require complex multi-table joins that struggle under high volumes of transaction data. TigerGraph provides instantaneous native graph traversals and pattern matching:

1. **Sub-second Multi-hop Traversal:** Connecting `Transaction -> MADE -> Card <- OWNS <- Customer -> OWNS -> Card` allows the agent to immediately detect whether an anomaly is isolated to one card or indicates a broader account takeover.
2. **Device Sharing & Ring Detection:** Through the `Transaction -> FROM_DEVICE -> DeviceProfile <- FROM_DEVICE -> Transaction` graph neighborhood, the agent discovers shared devices across ostensibly unrelated cards within a 90-day window, uncovering fraud rings.
3. **Card Testing Pattern Recognition:** Fast temporal querying across transactions on a card reveals rapid sub-$5 authorization spikes preceding large purchases.
4. **Graph-Grounded Retrieval-Augmented Generation (GraphRAG):** Rather than feeding raw, unorganized tabular logs to an LLM, the agent queries TigerGraph REST endpoints to extract structured subgraphs, customer behavioral baselines, and similar historical closed cases (`CC_INVOLVES`, `CC_ON_CARD`).
5. **Living Case Memory:** When an investigation concludes, `node_update_case_memory` persists the finding as a `FraudCase` vertex in TigerGraph, creating an evolving organizational memory.

---

## 4. Agentic Capabilities Implemented

- **Dual-Phase Next-Best Actions:** The agent records its recommendations *before* requesting additional evidence, simulates policy-controlled follow-ups (such as SMS customer verification under Rule R1), and updates its recommendations *after* evidence is received.
- **Dynamic Governance & Approval Routing:** The agent strictly respects bank compliance:
  - `auto`: Executed autonomously (e.g. `ALLOW_TRANSACTION`, `MONITOR_CARD`, `CREATE_CASE`).
  - `L1`: Team Lead approval for authorization declines and card blocks $\le \$2,500$.
  - `L2`: Fraud Manager approval for card blocks $> \$2,500$, `BLOCK_ALL_CARDS`, and `FILE_REPORT`.
- **Explainable Evidence Chains:** Every conclusion cites the exact source (`graph`, `document`, `customer`), the specific query reference (`query:card_window`, `query:device_neighbors`), and the IDs of all entities involved.
- **Automated Regulatory SAR Generation:** If fraud exposure exceeds $\$1,000$ or involves coordinated rings, the agent drafts a complete 6–12 sentence SAR detailing who, what, when, where, how, and why suspicious.
- **Model Fallback Resilience:** The agent features an intelligent multi-model failover system across Google Gemini models (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.8-flash`) with exponential backoff to handle quota restrictions seamlessly.

---

## 5. What We Learned

- **Native REST vs. Stored Procedures on Constrained Environments:** In resource-constrained environments (like a local workstation or container), leveraging pyTigerGraph's native REST endpoints for vertex and edge retrieval provides high throughput without the C++ compiler memory overhead of runtime GSQL compilation.
- **Graph Topology Trumps Raw Anomaly Scores:** In the IEEE dataset, model risk scores alone are frequently misleading—high scores occur on legitimate overseas travel, while sophisticated fraud attacks often stay below initial thresholds. Connecting transactions to device profiles and customer historical baselines in TigerGraph is what reliably differentiates fraud from false positives.
- **Explainability Builds Trust:** Analysts will not accept an AI agent's "black box" decisions. Structuring next-best actions with specific policy rule citations (R1–R10) and evidence claim references is critical for real-world enterprise adoption.

---

## 6. What We Would Improve With More Time

1. **Graph Neural Networks (TigerGraph GSQL ML):** Embed TigerGraph Graph Convolutional Networks (GCN) directly into the graph pipeline to compute real-time structural risk embeddings on vertices.
2. **Interactive Analyst-in-the-Loop Approval:** Extend the Streamlit UI to allow L1/L2 human reviewers to approve or modify queued recommendations directly in the interface with one-click webhook callbacks.
3. **Automated Subgraph Visualization in UI:** Integrate PyVis or Cytoscape directly into the case investigation tab to render interactive force-directed graph diagrams of the flagged episode.
4. **Vector Search on Document Vertices:** Re-enable native HNSW vector index search on policy documents once TigerGraph CE vector indexing is configured.
