"""
AgriPulse 2.0 — Ablation Experiment Suite Runner
=================================================
Runs the 4-experiment ablation protocol (E1→E4) with frozen hyperparameters.
Produces per-experiment JSON results, confusion matrices, and a comparative
summary table.

Usage:
    python backend/scripts/run_ablation_suite.py --data-dir backend/data
    python backend/scripts/run_ablation_suite.py --data-dir backend/data --use-cache
    python backend/scripts/run_ablation_suite.py --experiments E1 E4  # Run subset

Protocol:
    All experiments use identical seed, split, augmentation, optimizer,
    scheduler, epochs, and batch size as defined in experiment_protocol.json.
    The ONLY variable is the --mode flag.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

EXPERIMENTS = {
    "E1": {"mode": "baseline", "name": "E1_Baseline"},
    "E2": {"mode": "graph_only", "name": "E2_Spatial"},
    "E3": {"mode": "temporal_only", "name": "E3_Temporal"},
    "E4": {"mode": "full", "name": "E4_Full"},
}

PROTOCOL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "experiment_protocol.json"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "ablation_results"
)


def load_protocol() -> dict:
    """Load and validate the frozen experiment protocol."""
    with open(PROTOCOL_PATH, "r") as f:
        protocol = json.load(f)
    logger.info(
        f"Loaded protocol v{protocol['protocol_version']} "
        f"(frozen at {protocol['frozen_at']})"
    )
    return protocol


def run_experiment(exp_id: str, exp_config: dict, protocol: dict, use_cache: bool):
    """Run a single ablation experiment."""
    mode = exp_config["mode"]
    name = exp_config["name"]
    seed = protocol["seed"]

    logger.info("=" * 60)
    logger.info(f"STARTING {name} (mode={mode}, seed={seed})")
    logger.info("=" * 60)

    start_time = time.time()

    # Build command
    cmd = [
        sys.executable,
        "backend/app/training/train.py",
        "--mode", mode,
        "--seed", str(seed),
    ]
    if use_cache:
        cmd.append("--use-cache")

    # Run training
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    )

    elapsed = time.time() - start_time

    # Save result
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, f"{name}_results.json")

    experiment_result = {
        "experiment": name,
        "experiment_id": exp_id,
        "mode": mode,
        "seed": seed,
        "protocol_version": protocol["protocol_version"],
        "training_time_seconds": round(elapsed, 2),
        "return_code": result.returncode,
        "completed_at": datetime.now().isoformat(),
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-1000:] if result.stderr else "",
    }

    with open(result_file, "w") as f:
        json.dump(experiment_result, f, indent=4)

    status = "✅ SUCCESS" if result.returncode == 0 else "❌ FAILED"
    logger.info(f"{name}: {status} ({elapsed:.1f}s)")

    return result.returncode == 0


def generate_comparison_table(experiments_run: list):
    """Generate a comparative summary table from experiment results."""
    table_file = os.path.join(RESULTS_DIR, "comparison_summary.md")

    lines = [
        "# Ablation Comparison Summary\n",
        f"Generated: {datetime.now().isoformat()}\n\n",
        "| Experiment | Mode | Status | Training Time |",
        "|---|---|---|---|",
    ]

    for exp_id in experiments_run:
        name = EXPERIMENTS[exp_id]["name"]
        result_file = os.path.join(RESULTS_DIR, f"{name}_results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            status = "✅" if data["return_code"] == 0 else "❌"
            time_str = f"{data['training_time_seconds']:.1f}s"
        else:
            status = "⏳ Pending"
            time_str = "—"
        lines.append(
            f"| {name} | {EXPERIMENTS[exp_id]['mode']} | {status} | {time_str} |"
        )

    lines.append(
        "\n> **Note**: Detailed per-experiment metrics (F1, ECE, Brier, "
        "confusion matrices) are saved in individual JSON files."
    )

    with open(table_file, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Comparison table saved to {table_file}")


def main():
    parser = argparse.ArgumentParser(
        description="AgriPulse 2.0 — Ablation Experiment Suite"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["E1", "E2", "E3", "E4"],
        choices=["E1", "E2", "E3", "E4"],
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached DINOv2 features for faster training",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="backend/data",
        help="Root data directory containing dataset folders",
    )
    args = parser.parse_args()

    protocol = load_protocol()

    logger.info("🌾 AgriPulse 2.0 — Ablation Experiment Suite")
    logger.info(f"Experiments to run: {args.experiments}")
    logger.info(f"Protocol seed: {protocol['seed']}")

    results = {}
    for exp_id in args.experiments:
        exp_config = EXPERIMENTS[exp_id]
        success = run_experiment(exp_id, exp_config, protocol, args.use_cache)
        results[exp_id] = success

    generate_comparison_table(args.experiments)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUITE COMPLETE")
    logger.info("=" * 60)
    for exp_id, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"  {EXPERIMENTS[exp_id]['name']}: {status}")
    logger.info(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
