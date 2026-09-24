"""
ETL pipeline: Load HHGOA_IEEE dataset into TigerGraph.
Run:  python data/etl_load.py
"""

import os
import hashlib
import pandas as pd
import pyTigerGraph as tg
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai as google_genai
import numpy as np
import json

load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATASET_DIR = os.getenv("DATASET_DIR", "./HHGOA_IEEE")
TG_HOST     = os.getenv("TIGERGRAPH_HOST", "localhost")
TG_USER     = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASS     = os.getenv("TIGERGRAPH_PASSWORD", "tigergraph")
TG_GRAPH    = os.getenv("TIGERGRAPH_GRAPH", "fraud_investigation")
TG_PORT     = os.getenv("TIGERGRAPH_PORT", "9000")
TG_SSL      = os.getenv("TIGERGRAPH_USE_SSL", "false").lower() == "true"

_genai_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BATCH_SIZE = 1_000

# ──────────────────────────────────────────────
# Connect
# ──────────────────────────────────────────────
conn = tg.TigerGraphConnection(
    host=f"https://{TG_HOST}" if TG_SSL else f"http://{TG_HOST}",
    graphname=TG_GRAPH,
    username=TG_USER,
    password=TG_PASS,
    useCert=TG_SSL,
)
conn.getToken(conn.createSecret())
print("✅ Connected to TigerGraph")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def embed_text(text: str) -> list[float]:
    """Generate embedding via Google Generative AI text-embedding-004 (free tier)."""
    result = _genai_client.models.embed_content(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
        contents=text[:8192],
    )
    return result.embeddings[0].values


def device_profile_id(row) -> str | None:
    """Hash DeviceInfo + OS + browser + screen into a stable ID."""
    parts = [
        str(row.get("DeviceInfo", "")),
        str(row.get("id_30", "")),   # OS
        str(row.get("id_31", "")),   # browser
        str(row.get("id_33", "")),   # screen
    ]
    if all(p in ("", "nan", "None") for p in parts):
        return None
    key = "|".join(parts)
    return "D" + hashlib.sha1(key.encode()).hexdigest()[:10].upper()


def upsert_batch(vertex_type: str, data: list) -> None:
    formatted = []
    for item in data:
        if isinstance(item, dict):
            vid = item.get("primary_id") or item.get("customer_id") or item.get("card_id") or item.get("txn_id") or item.get("case_id") or item.get("domain") or item.get("region_code") or item.get("dp_id") or item.get("device_profile_id")
            attrs = {k: v for k, v in item.items() if k != "primary_id"}
            formatted.append((str(vid), attrs))
        else:
            formatted.append(item)
    for i in range(0, len(formatted), BATCH_SIZE):
        batch = formatted[i : i + BATCH_SIZE]
        conn.upsertVertices(vertex_type, batch)


def edge_batch(from_type, to_type, edge_type, data: list[tuple]) -> None:
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        conn.upsertEdges(from_type, edge_type, to_type, batch)


# ──────────────────────────────────────────────
# 1. Transactions
# ──────────────────────────────────────────────
def load_transactions():
    print("\n📦 Loading transactions.csv in streaming chunks …")

    # Map known cards from case_pack and closed_cases
    known_cards = {}
    try:
        cp = pd.read_csv(f"{DATASET_DIR}/case_pack.csv", dtype=str)
        for _, r in cp.iterrows():
            if pd.notna(r.get("flagged_txn_id")) and pd.notna(r.get("card_id")):
                known_cards[str(r["flagged_txn_id"])] = str(r["card_id"])
    except Exception:
        pass

    try:
        cc = pd.read_csv(f"{DATASET_DIR}/closed_cases_history.csv", dtype=str)
        for _, r in cc.iterrows():
            cid = str(r["card_id"])
            txns = str(r.get("txn_ids", "")).split("|")
            for t in txns:
                t = t.strip()
                if t:
                    known_cards[t] = cid
    except Exception:
        pass

    cols = [
        "TransactionID", "customer_id", "ts", "TransactionDT", "TransactionAmt",
        "ProductCD", "channel", "addr1", "addr2", "dist1", "dist2",
        "P_emaildomain", "R_emaildomain", "risk_score",
        "card1", "card2", "card3", "card4", "card5", "card6"
    ]

    seen_customers = set()
    seen_cards = set()
    seen_domains = set()
    seen_regions = set()

    reader = pd.read_csv(
        f"{DATASET_DIR}/transactions.csv",
        usecols=cols,
        dtype={"TransactionID": str, "customer_id": str},
        chunksize=50000,
        low_memory=False,
    )

    for chunk_idx, df in enumerate(reader):
        print(f"  → Processing chunk {chunk_idx + 1} (50,000 rows)...")
        df["txn_id"] = df["TransactionID"].astype(str)
        df["card_id"] = df["txn_id"].map(known_cards).fillna(df["customer_id"] + "-K1")

        # ── Customers ──
        new_cust = df[~df["customer_id"].isin(seen_customers)][["customer_id"]].drop_duplicates()
        if not new_cust.empty:
            cust_rows = [
                {"primary_id": cid, "customer_id": cid, "created_at": "2016-01-01 00:00:00"}
                for cid in new_cust["customer_id"]
            ]
            upsert_batch("Customer", cust_rows)
            seen_customers.update(new_cust["customer_id"])

        # ── Cards & OWNS ──
        new_cards_df = df[~df["card_id"].isin(seen_cards)].drop_duplicates(subset=["card_id"])
        if not new_cards_df.empty:
            card_rows = []
            owns_edges = []
            for _, r in new_cards_df.iterrows():
                card_rows.append({
                    "primary_id": r.card_id,
                    "card_id": r.card_id,
                    "customer_id": r.customer_id,
                    "card1": int(r.card1) if pd.notna(r.card1) else 0,
                    "card2": float(r.card2) if pd.notna(r.card2) else 0.0,
                    "card3": float(r.card3) if pd.notna(r.card3) else 0.0,
                    "card4": str(r.card4) if pd.notna(r.card4) else "",
                    "card5": float(r.card5) if pd.notna(r.card5) else 0.0,
                    "card6": str(r.card6) if pd.notna(r.card6) else "",
                    "status": "active",
                })
                owns_edges.append((r.customer_id, r.card_id, {}))
            upsert_batch("Card", card_rows)
            edge_batch("Customer", "Card", "OWNS", owns_edges)
            seen_cards.update(new_cards_df["card_id"])

        # ── Transactions ──
        txn_rows = []
        made_edges = []
        for r in df.itertuples(index=False):
            txn_rows.append({
                "primary_id": r.txn_id,
                "txn_id": r.txn_id,
                "card_id": r.card_id,
                "customer_id": r.customer_id,
                "ts": str(r.ts),
                "transaction_dt": int(r.TransactionDT),
                "amount": float(r.TransactionAmt),
                "product_cd": str(r.ProductCD),
                "channel": str(r.channel),
                "addr1": float(r.addr1) if pd.notna(r.addr1) else -1.0,
                "addr2": float(r.addr2) if pd.notna(r.addr2) else -1.0,
                "dist1": float(r.dist1) if pd.notna(r.dist1) else -1.0,
                "dist2": float(r.dist2) if pd.notna(r.dist2) else -1.0,
                "p_emaildomain": str(r.P_emaildomain) if pd.notna(r.P_emaildomain) else "",
                "r_emaildomain": str(r.R_emaildomain) if pd.notna(r.R_emaildomain) else "",
                "risk_score": float(r.risk_score) if pd.notna(r.risk_score) else 0.0,
            })
            made_edges.append((r.card_id, r.txn_id, {}))

        upsert_batch("Transaction", txn_rows)
        edge_batch("Card", "Transaction", "MADE", made_edges)

        # ── Email domains ──
        domains = set(df["P_emaildomain"].dropna().unique()) | set(df["R_emaildomain"].dropna().unique())
        domains.discard("")
        new_domains = domains - seen_domains
        if new_domains:
            upsert_batch("EmailDomain", [{"primary_id": d, "domain": d} for d in new_domains])
            seen_domains.update(new_domains)

        p_emails = [(r.txn_id, str(r.P_emaildomain), {}) for r in df[df["P_emaildomain"].notna()].itertuples(index=False)]
        if p_emails:
            edge_batch("Transaction", "EmailDomain", "PURCHASER_EMAIL", p_emails)

        # ── Billing regions ──
        regions_df = df[["addr1", "addr2"]].dropna(subset=["addr1"]).drop_duplicates()
        new_regions = regions_df[~regions_df["addr1"].astype(str).isin(seen_regions)]
        if not new_regions.empty:
            region_rows = [
                {"primary_id": str(r.addr1), "region_code": str(r.addr1), "country_code": float(r.addr2) if pd.notna(r.addr2) else -1.0}
                for r in new_regions.itertuples(index=False)
            ]
            upsert_batch("BillingRegion", region_rows)
            seen_regions.update(new_regions["addr1"].astype(str))

        billed_edges = [(r.txn_id, str(r.addr1), {}) for r in df[df["addr1"].notna()].itertuples(index=False)]
        if billed_edges:
            edge_batch("Transaction", "BillingRegion", "BILLED_IN", billed_edges)

    print("✅ All transactions loaded")


# ──────────────────────────────────────────────
# 2. Identity / Device Profiles
# ──────────────────────────────────────────────
def load_identity(txn_df=None):
    print("\n📦 Loading identity.csv in streaming chunks …")
    cols = ["TransactionID", "DeviceInfo", "DeviceType", "id_30", "id_31", "id_33", "id_15", "id_23"]
    reader = pd.read_csv(
        f"{DATASET_DIR}/identity.csv",
        usecols=cols,
        dtype={"TransactionID": str},
        chunksize=10000,
        low_memory=False,
    )

    seen_devices = set()
    for chunk_idx, idf in enumerate(reader):
        print(f"  → Processing identity chunk {chunk_idx + 1}...")
        idf["txn_id"] = idf["TransactionID"].astype(str)
        idf["dp_id"] = idf.apply(device_profile_id, axis=1)
        idf = idf[idf["dp_id"].notna()]

        # ── Device profiles ──
        new_dp = idf[~idf["dp_id"].isin(seen_devices)].drop_duplicates(subset=["dp_id"])
        if not new_dp.empty:
            dp_rows = [
                {
                    "primary_id": str(r.dp_id),
                    "device_profile_id": str(r.dp_id),
                    "device_info":  str(r.DeviceInfo)  if pd.notna(r.DeviceInfo)  else "",
                    "device_type":  str(r.DeviceType)  if pd.notna(r.DeviceType)  else "",
                    "os":           str(r.id_30)       if pd.notna(r.id_30)       else "",
                    "browser":      str(r.id_31)       if pd.notna(r.id_31)       else "",
                    "screen":       str(r.id_33)       if pd.notna(r.id_33)       else "",
                    "is_new_device":str(r.id_15)       if pd.notna(r.id_15)       else "",
                    "proxy_type":   str(r.id_23)       if pd.notna(r.id_23)       else "",
                }
                for r in new_dp.itertuples(index=False)
            ]
            upsert_batch("DeviceProfile", dp_rows)
            seen_devices.update(new_dp["dp_id"])

        # ── FROM_DEVICE edges ──
        from_device_edges = [
            (str(r.txn_id), str(r.dp_id), {})
            for r in idf.itertuples(index=False)
        ]
        edge_batch("Transaction", "DeviceProfile", "FROM_DEVICE", from_device_edges)

    print("✅ Identity loaded")


# ──────────────────────────────────────────────
# 3. Closed Cases
# ──────────────────────────────────────────────
def load_closed_cases():
    print("\n📦 Loading closed_cases_history.csv …")
    ccdf = pd.read_csv(f"{DATASET_DIR}/closed_cases_history.csv", dtype=str)

    rows = []
    for _, r in tqdm(ccdf.iterrows(), total=len(ccdf)):
        notes = str(r.get("analyst_notes", "")) if pd.notna(r.get("analyst_notes")) else ""
        rows.append({
            "primary_id":       r.case_id,
            "case_id":          r.case_id,
            "customer_id":      str(r.customer_id),
            "card_id":          str(r.card_id),
            "opened_at":        str(r.opened_at),
            "closed_at":        str(r.closed_at),
            "outcome":          str(r.outcome),
            "pattern":          str(r.pattern),
            "first_fraud_txn_id": str(r.first_fraud_txn_id) if pd.notna(r.first_fraud_txn_id) else "",
            "n_txns":           int(r.n_txns) if pd.notna(r.n_txns) else 0,
            "exposure_usd":     float(r.exposure_usd) if pd.notna(r.exposure_usd) else 0.0,
            "actions_taken":    str(r.actions_taken) if pd.notna(r.actions_taken) else "",
            "report_filed":     str(r.report_filed).lower() == "yes",
            "analyst_notes":    notes,
        })

    upsert_batch("ClosedCase", rows)

    # CC_INVOLVES, CC_ON_CARD, CC_CONNECTED_CARD edges
    print("  → Closed case edges")
    involves_edges = []
    on_card_edges = []
    connected_card_edges = []

    for _, r in ccdf.iterrows():
        txn_ids = str(r.txn_ids).split("|") if pd.notna(r.txn_ids) else []
        for t in txn_ids:
            t = t.strip()
            if t:
                involves_edges.append((r.case_id, t, {}))
        if pd.notna(r.card_id):
            on_card_edges.append((r.case_id, str(r.card_id), {}))
        connected = str(r.connected_card_ids).split("|") if pd.notna(r.connected_card_ids) else []
        for c in connected:
            c = c.strip()
            if c:
                connected_card_edges.append((r.case_id, c, {}))

    edge_batch("ClosedCase", "Transaction", "CC_INVOLVES", involves_edges)
    edge_batch("ClosedCase", "Card", "CC_ON_CARD", on_card_edges)
    edge_batch("ClosedCase", "Card", "CC_CONNECTED_CARD", connected_card_edges)

    print("✅ Closed cases loaded")


# ──────────────────────────────────────────────
# 4. Policy & Fraud Patterns
# ──────────────────────────────────────────────
FRAUD_PATTERNS = [
    {
        "pattern_id": "card_testing",
        "name": "Card Testing",
        "policy_rules": "R5",
        "description": (
            "A stolen card number is checked before use: three or more tiny online "
            "authorizations, often under $5, then a larger purchase. Confirmed by "
            "the sequence itself. Policy R5."
        ),
    },
    {
        "pattern_id": "card_not_present_fraud",
        "name": "Card-Not-Present Fraud",
        "policy_rules": "R1,R2,R3,R4",
        "description": (
            "The card number is used online without the card. Amounts and products "
            "that don't fit the cardholder's history, often in a burst of two to four "
            "within 48 hours. On its own, one unusual online purchase is ambiguous: "
            "verify. Policy R1 to R4."
        ),
    },
    {
        "pattern_id": "card_not_present_new_device",
        "name": "Card-Not-Present Fraud from New Device",
        "policy_rules": "R1,R2,R3,R4",
        "description": (
            "Same as card-not-present fraud, with the identity record marking the "
            "device as New for this account, sometimes behind a proxy. Stronger than "
            "pattern 2, still not proof: people buy new phones. Policy R1 to R4."
        ),
    },
    {
        "pattern_id": "out_of_region_use",
        "name": "Out-of-Region Use",
        "policy_rules": "R2,R3",
        "description": (
            "Card-present purchases in a billing region the cardholder has no history "
            "in, while their normal activity continues at home. Several days of "
            "purchases in one new region is a trip, not a clone. Policy R2, R3."
        ),
    },
    {
        "pattern_id": "account_takeover",
        "name": "Account Takeover",
        "policy_rules": "R1,R2,R6",
        "description": (
            "Mixed-channel activity inconsistent with the cardholder, often with "
            "device and match-flag anomalies, pointing to stolen credentials rather "
            "than a stolen number."
        ),
    },
]

POLICY_SECTIONS = [
    {
        "doc_id": "policy_actions",
        "title": "Fraud Policy — Actions",
        "content": (
            "Actions: ALLOW_TRANSACTION (auto), DECLINE_TRANSACTION (L1), "
            "MONITOR_CARD (auto), MONITOR_CONNECTED_CARDS (auto), "
            "WARN_CUSTOMER (auto), VERIFY_WITH_CUSTOMER (auto), "
            "STEP_UP_AUTH (auto), BLOCK_CARD (L1 if ≤$2500, L2 if >$2500), "
            "BLOCK_ALL_CARDS (L2 always), GENERATE_REPORT (auto), "
            "CREATE_CASE (auto), FILE_REPORT (L2 always), "
            "ESCALATE_TO_ANALYST (auto), CLOSE_NO_FRAUD (auto)."
        ),
    },
    {
        "doc_id": "policy_rules",
        "title": "Fraud Policy — Rules R1–R10",
        "content": (
            "R1: Verify before block on weak signal (prob < 0.70, single signal). "
            "R2: Customer denies → BLOCK_CARD + CREATE_CASE; add FILE_REPORT if exposure > $1000 or shared device. "
            "R3: Customer confirms → CLOSE_NO_FRAUD. "
            "R4: No reply within 24h → MONITOR_CARD + DECLINE_TRANSACTION pending; escalate if exposure > $500. "
            "R5: Card testing (3+ small online auth ≤1h then bigger purchase) → DECLINE + STEP_UP_AUTH; "
            "if purchase > $100 already cleared → BLOCK_CARD. "
            "R6: Shared device/region/email across cards → CREATE_CASE + FILE_REPORT + MONITOR_CONNECTED_CARDS. "
            "R7: Disputed recurring charge → CREATE_CASE + VERIFY + WARN; no block. "
            "R8: Uncertain + exposure > $500 or conflicting evidence → ESCALATE_TO_ANALYST. "
            "R9: Undocumented pattern → CREATE_CASE + FILE_REPORT + ESCALATE_TO_ANALYST; describe pattern. "
            "R10: Never BLOCK_ALL_CARDS unless 2+ confirmed fraud cards or credentials confirmed compromised."
        ),
    },
    {
        "doc_id": "policy_cases_vs_sars",
        "title": "Fraud Policy — Cases vs SARs (3a)",
        "content": (
            "CREATE_CASE: open when fraud probability ≥ 0.30, when requesting evidence, or on customer dispute. "
            "FILE_REPORT: file when fraud confirmed/strongly suspected AND "
            "(exposure > $1000 OR shared device/region/card fraud OR coordinated/undocumented pattern). "
            "Most cases never need a report. Report narrative: who, what, when, where, how, why suspicious."
        ),
    },
    {
        "doc_id": "policy_stopping",
        "title": "Fraud Policy — Stopping (Section 6)",
        "content": (
            "Stop when: fraud probability ≥ 0.85 or ≤ 0.15 supported by 2+ independent evidence pieces; "
            "a verification response settles the question; "
            "further steps are unlikely to change the decision."
        ),
    },
]


def load_policy_and_patterns():
    print("\n📦 Loading policy & fraud patterns …")
    for p in FRAUD_PATTERNS:
        conn.upsertVertex("FraudPattern", p["pattern_id"], {
            "pattern_id": p["pattern_id"],
            "name": p["name"],
            "description": p["description"],
            "policy_rules": p["policy_rules"],
        })

    for doc in POLICY_SECTIONS:
        conn.upsertVertex("PolicyDocument", doc["doc_id"], {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "content": doc["content"],
        })
    print("✅ Policy & patterns loaded")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    steps = sys.argv[1:] or ["transactions", "identity", "cases", "policy"]

    if "transactions" in steps:
        load_transactions()
    if "identity" in steps:
        load_identity()
    if "cases" in steps:
        load_closed_cases()
    if "policy" in steps:
        load_policy_and_patterns()

    print("\n🎉 ETL complete.")
