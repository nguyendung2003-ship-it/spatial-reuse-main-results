#!/usr/bin/env python3
"""Replay the nine main controllers using the versioned experiment config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "code/neural_bandit_bridge"
CONFIG = ROOT / "configs/main_experiment.json"


def clean_environment(config: dict) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("NSTEST_")}
    # The explicit override is useful when testing an already built ns-3 tree.
    if os.environ.get("NSTEST_NS3_DIR"):
        env["NSTEST_NS3_DIR"] = str(Path(os.environ["NSTEST_NS3_DIR"]).resolve())
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    for key, value in config["native_environment"].items():
        env[key] = str(value)
    return env


def hcm_arguments(config: dict, profile: str | None = None) -> list[str]:
    settings = dict(config["hcm_sampler"])
    if profile is not None:
        settings["hcm_profile"] = profile
    return [token for key, value in settings.items()
            for token in ("--" + key.replace("_", "-"), str(value))]


def command(config: dict, method_id: str, topology: str, args: argparse.Namespace, index: int) -> list[str]:
    method = config["methods"][method_id]
    common = config["simulation"]
    duration = common["duration_seconds"] if args.mode == "full" else 7.5
    seed_end = args.seed_end if args.seed_end is not None else (25 if args.mode == "full" else args.seed_start)
    run_id = f"{args.run_id}/{method_id}_{topology}"
    if method["kind"] == "native":
        cmd = [sys.executable, str(BRIDGE / "run_replicates_low_concurrency.py"),
               "--topology", topology, "--methods", method["runner_label"],
               "--output-dir", str(ROOT / "work/runs" / run_id),
               "--replicates", str(seed_end - args.seed_start + 1),
               "--seed-start", str(args.seed_start), "--jobs", str(args.jobs),
               "--max-attempts", "1", "--run-tag", f"main{index}seed{args.seed_start}",
               "--duration", str(duration), "--test-duration", str(common["measurement_interval_seconds"]),
               "--rate-manager", common["rate_manager"], "--reward", common["reward"],
               "--agent-seed-offset", str(common["agent_seed_offset"]),
               "--hcmts-acquisition-epsilon", str(method.get("acquisition_epsilon", 0.0)),
               *hcm_arguments(config)]
    else:
        cmd = [sys.executable, str(BRIDGE / "run_qnn_parallel.py"),
               "--run-id", run_id, "--topo", topology,
               "--seed-start", str(args.seed_start), "--seed-end", str(seed_end),
               "--jobs", str(args.jobs), "--base-port", str(args.base_port + index * 32),
               "--nice", "0", "--duration", str(duration),
               "--test-duration", str(common["measurement_interval_seconds"]),
               "--rate-manager", common["rate_manager"], "--transport", common["transport"],
               "--agent-seed-offset", str(common["agent_seed_offset"]),
               "--surrogate", method["surrogate"], "--policy", "neural_ts",
               "--sampler", method["sampler"], "--entry", method["entry"],
               "--hidden", "64", "--candidates", str(method["candidates"]),
               "--novel-candidates", str(method["novel_candidates"]),
               "--acquisition-preset", "custom", "--feature-dim", "32",
               "--acquisition-top-k", "1", "--acquisition-temperature", "0.0",
               "--acquisition-epsilon", str(method["acquisition_epsilon"]),
               "--posterior-scale", "1.0",
               "--qnn-architecture", method.get("qnn_architecture", "qcnn_shared_reupload"),
               "--qnn-layers", str(method.get("qnn_layers", 3)), "--qnn-n-qubits", "5", "--qnn-max-z-order", "5",
               "--quantum-lr", str(method["quantum_lr"]), "--quantum-weight-decay", "0.0",
               "--child-attempts", "1", "--orchestrator-attempts", "1",
               "--save-final-checkpoint", *hcm_arguments(config, method["hcm_profile"])]
        for key, value in config["training"].items():
            cmd.extend(["--" + key.replace("_", "-"), str(value)])
        if method["feature_diagnostics"]:
            cmd.append("--feature-diagnostics")
        else:
            cmd.append("--no-feature-diagnostics")
    return cmd


def main() -> int:
    config = json.loads(CONFIG.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--methods", nargs="+", choices=list(config["methods"]), default=list(config["methods"]))
    parser.add_argument("--topologies", nargs="+", choices=list(config["topologies"]), default=list(config["topologies"]))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=27000)
    parser.add_argument("--run-id", default="main_replay")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    end = args.seed_end if args.seed_end is not None else (25 if args.mode == "full" else args.seed_start)
    if not 1 <= args.seed_start <= end <= 25 or not 1 <= args.jobs <= 25:
        parser.error("Use seeds 1..25 and jobs 1..25")
    if Path(args.run_id).is_absolute() or ".." in Path(args.run_id).parts:
        parser.error("run-id must be a relative name within work/runs")
    cells = [(method, topology) for topology in args.topologies for method in args.methods]
    if not 1024 <= args.base_port <= 65535 - len(cells) * 32:
        parser.error("base-port leaves insufficient ports for the requested cells")
    env = clean_environment(config)
    plan = [{"method": method, "topology": topology, "command": command(config, method, topology, args, index)}
            for index, (method, topology) in enumerate(cells)]
    if args.dry_run:
        print(json.dumps({"config": str(CONFIG.relative_to(ROOT)), "cells": plan}, indent=2))
        return 0
    ns3 = Path(env.get("NSTEST_NS3_DIR", str(ROOT / "work/ns-allinone-3.35/ns-3.35")))
    if not (ns3 / "build/scratch/nsTest/nsTest").is_file():
        parser.error("Build ns-3 first with: python scripts/build_ns3.py")
    output = ROOT / "work/runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch_plan.json").write_text(json.dumps({"config": config, "cells": plan}, indent=2) + "\n")
    for cell in plan:
        print(f"Running {cell['method']} on {cell['topology']}", flush=True)
        result = subprocess.run(cell["command"], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    print(f"Completed {len(plan)} cells; output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
