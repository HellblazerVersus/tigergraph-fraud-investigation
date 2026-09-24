"""
benchmark/run_benchmark_cases.py

Runs all 20 cases from case_pack.csv through the investigation agent
and saves one answer JSON file per case to ../cases/<case_id>.json

Usage:
  python benchmark/run_benchmark_cases.py
  python benchmark/run_benchmark_cases.py --case HHG-001   # single case
  python benchmark/run_benchmark_cases.py --workers 4      # parallel
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import run_investigation
from agent.state import InvestigationState

console = Console()
CASES_DIR = Path(os.getenv("CASES_OUTPUT_DIR", "./cases"))
CASES_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────
# State → Answer file
# ─────────────────────────────────────────────────────────
def state_to_answer(state: InvestigationState) -> dict:
    """Convert the final investigation state to the required answer format."""
    return {
        "case_id": state["case_pack_id"],
        "case": {
            "status": state["case_status"],
            "verdict": state["verdict"],
            "fraud_probability": round(state["fraud_probability"], 4),
            "pattern": state["pattern"],
            "pattern_description": state["pattern_description"],
            "affected_txn_ids": state["affected_txn_ids"],
            "first_suspicious_txn_id": state["first_suspicious_txn_id"],
            "connected_card_ids": state["connected_card_ids"],
            "connected_device_profiles": state["connected_device_profiles"],
            "exposure_usd": round(state["exposure_usd"], 2),
            "evidence": state["evidence"],
            "similar_prior_cases": state["similar_prior_cases"],
            "summary": state["summary"],
            "written_to_graph": state["written_to_graph"],
            "graph_case_id": state["graph_case_id"],
        },
        "evidence_requests": state["evidence_requests"],
        "next_best_actions": {
            "initial": state["initial_actions"],
            "final": state["final_actions"],
            "what_changed": state["what_changed"],
        },
        "sar": {
            "file": state["sar_file"],
            "reason": state["sar_reason"],
            "narrative": state["sar_narrative"] if state["sar_file"] else "",
            "subjects": state["sar_subjects"] if state["sar_file"] else [],
            "total_amount_usd": round(state["sar_total_amount_usd"], 2) if state["sar_file"] else 0,
            "activity_dates": state["sar_activity_dates"] if state["sar_file"] else [],
        },
        "stop_reason": state["stop_reason"],
        "tool_calls": state["tool_calls"],
        "tokens": state["tokens_used"],
        "latency_s": round(state["latency_s"], 2),
    }


# ─────────────────────────────────────────────────────────
# Run a single case
# ─────────────────────────────────────────────────────────
def run_case(case_row: dict, skip_existing: bool = False) -> tuple[str, bool, str]:
    """Returns (case_id, success, message)."""
    case_id = case_row["case_id"]
    out_path = CASES_DIR / f"{case_id}.json"

    if skip_existing and out_path.exists() and out_path.stat().st_size > 50:
        return case_id, True, f"⏩ {case_id} (already exists)"

    try:
        final_state = run_investigation(case_row)
        answer = state_to_answer(final_state)

        with open(out_path, "w") as f:
            json.dump(answer, f, indent=2)

        return case_id, True, f"✅ {case_id} → {out_path}"

    except Exception as e:
        err_msg = f"❌ {case_id} failed: {e}"
        # Write partial answer with error info
        with open(CASES_DIR / f"{case_id}_ERROR.json", "w") as f:
            json.dump({"case_id": case_id, "error": str(e)}, f, indent=2)
        return case_id, False, err_msg


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Run a single case, e.g. HHG-001")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (careful with rate limits)")
    parser.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", "./HHGOA_IEEE"))
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip cases that already have a valid JSON answer file")
    args = parser.parse_args()

    case_pack = pd.read_csv(f"{args.dataset_dir}/case_pack.csv",
                             dtype={"flagged_txn_id": str, "card_id": str, "customer_id": str})
    # Normalize risk_score — may be NaN for customer_report triggers
    case_pack["risk_score"] = pd.to_numeric(case_pack["risk_score"], errors="coerce")

    if args.case:
        cases = case_pack[case_pack["case_id"] == args.case]
        if cases.empty:
            console.print(f"[red]Case {args.case} not found in case_pack.csv[/red]")
            sys.exit(1)
    else:
        cases = case_pack

    console.print(f"\n[bold cyan]🔍 HHGOA Fraud Investigation Benchmark[/bold cyan]")
    console.print(f"Running {len(cases)} case(s) with {args.workers} worker(s)\n")

    results = []
    start = time.time()

    if args.workers == 1:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), console=console,
        ) as progress:
            task = progress.add_task("Investigating...", total=len(cases))
            for _, row in cases.iterrows():
                case_id, ok, msg = run_case(row.to_dict(), skip_existing=args.skip_existing)
                results.append((case_id, ok))
                progress.advance(task)
                console.print(msg)
                if not msg.startswith("⏩"):
                    time.sleep(2)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_case, row.to_dict()): row["case_id"]
                       for _, row in cases.iterrows()}
            for fut in as_completed(futures):
                case_id, ok, msg = fut.result()
                results.append((case_id, ok))
                console.print(msg)

    elapsed = time.time() - start
    success = sum(1 for _, ok in results if ok)
    console.print(f"\n[bold]{'='*50}[/bold]")
    console.print(f"✅ {success}/{len(results)} cases completed in {elapsed:.1f}s")
    console.print(f"📁 Answer files written to: {CASES_DIR}/")

    # Summary table
    failed = [(c, ok) for c, ok in results if not ok]
    if failed:
        console.print(f"\n[red]Failed cases:[/red]")
        for c, _ in failed:
            console.print(f"  - {c}")


if __name__ == "__main__":
    main()
