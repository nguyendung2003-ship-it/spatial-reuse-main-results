#!/usr/bin/env python3
"""Replay C6 seed 1 to measure four native C++ controllers in their live control loop.

The network simulation supplies each controller's observation history. Timers
inside C++ measure action selection, or INSPIRE GP prescription/consensus.
This is separate from the frozen learned-feature microbenchmark.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "epsilon_greedy": {"native_method": "egreedy", "display": "epsilon-greedy",
                       "optimizer": "EGREED", "sampler": "UNI", "entry": "DEF",
                       "operation": "inference_end_to_end",
                       "scope": "native optimizer action selection, including exploration configuration generation"},
    "gm_ts": {"native_method": "gmts", "display": "GM-TS",
              "optimizer": "TGNORM", "sampler": "HGM", "entry": "DEF",
              "operation": "inference_end_to_end",
              "scope": "native optimizer action selection, including proposal generation and per-arm sampling"},
    "hcm_ts": {"native_method": "gmts_ring", "display": "HCM-TS",
               "optimizer": "TGNORM", "sampler": "HCM", "entry": "DEGA",
               "operation": "inference_end_to_end",
               "scope": "native HCM-TS action selection at env_step divisible by 3; epsilon_f=0.1; holds excluded"},
    "inspire": {"native_method": "inspire", "display": "INSPIRE",
                "optimizer": "INSPIRE", "sampler": "UNI", "entry": "DEF",
                "operation": "inference_model",
                "scope": "native distributed GP prescription and consensus, internal n_obs>=31 (32 observations available); GP fitting excluded"},
}
INTERVAL = .075


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata() -> dict:
    cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    return {"utc": datetime.now(timezone.utc).isoformat(), "cpu_model": cpu,
            "kernel": platform.release(), "python": platform.python_version(),
            "affinity": sorted(os.sched_getaffinity(0))}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_trace(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def base_environment(ns3: Path, method: dict, duration: float, trace: Path, suffix: str) -> dict:
    # Several C++ feature switches test presence, so remove every inherited
    # NSTEST_* setting before defining this stationary UDP benchmark.
    env = {key: value for key, value in os.environ.items() if not key.startswith("NSTEST_")}
    env["LD_LIBRARY_PATH"] = str(ns3 / "build/lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env.update({
        "NSTEST_TOPO": "C6o", "NSTEST_OPTIMIZER": method["optimizer"],
        "NSTEST_SAMPLER": method["sampler"], "NSTEST_ENTRY": method["entry"],
        "NSTEST_REWARD": "ADHOC", "NSTEST_RATE_MANAGER": "MINSTREL",
        "NSTEST_TRANSPORT": "udp", "NSTEST_NSIMULATIONS": "1",
        "NSTEST_SEED": "1", "NSTEST_AGENT_SEED": "1",
        "NSTEST_DURATION": str(duration), "NSTEST_TEST_DURATION": str(INTERVAL),
        "NSTEST_OUTPUT_SUFFIX": suffix,
        "NSTEST_TIMING_METHOD": method["native_method"], "NSTEST_CPP_TIMING_FILE": str(trace),
        "NSTEST_HCM_PROFILE": "author-restart", "NSTEST_HCM_GLOBAL_EPS": "0.0",
        "NSTEST_HCM_RESTART_PATIENCE": "48", "NSTEST_HCM_RESTART_BURST": "16",
        "NSTEST_HCM_RESTART_MAX_BURSTS": "4", "NSTEST_HCM_RESTART_MIN_DELTA": "0.01",
        "NSTEST_HCM_RESTART_TARGET": "0.89", "NSTEST_ADHOC_TIEBREAK_WEIGHT": "0.0",
        "NSTEST_INSPIRE_WINDOW": "32", "NSTEST_INSPIRE_RESTARTS": "6",
        "NSTEST_INSPIRE_SWEEPS": "3", "NSTEST_INSPIRE_HYPER_PERIOD": "10",
        "NSTEST_TGNORM_ACQUISITION_EPS": "0.1" if method["sampler"] == "HCM" else "0.0",
    })
    if method["sampler"] == "HCM":
        env.update({"NSTEST_TGNORM_CANDIDATES": "512", "NSTEST_TGNORM_NOVEL_CANDIDATES": "128",
                    "NSTEST_TGNORM_COLD_START": "20", "NSTEST_TGNORM_CANDIDATE_ATTEMPTS": "50000"})
    return env


def trace_summary(path: Path, method: dict, duration: float, output: Path) -> tuple[list[dict], dict]:
    rows = read_trace(path)
    if not rows:
        raise ValueError(f"Empty native trace: {path}")
    for row in rows:
        if row["topology"] != "C6o" or row["seed"] != "1" or row["method"] != method["native_method"]:
            raise ValueError(f"Wrong native trace provenance: {path}")
    last_cycle = int(round(duration / INTERVAL))
    selected = [dict(row) for row in rows if row["operation"] == method["operation"]]
    for row in selected:
        row["measurement_cycle"] = int(row["env_step"]) - (method["native_method"] != "inspire")
    terminal_extra = sum(row["measurement_cycle"] > last_cycle for row in selected)
    selected = [row for row in selected if 0 <= row["measurement_cycle"] <= last_cycle]
    if method["native_method"] == "gmts_ring":
        selected = [row for row in selected if int(row["env_step"]) % 3 == 0]
    elif method["native_method"] == "inspire":
        selected = [row for row in selected if int(row["n_obs"]) >= 31]
    if len({int(row["call_index"]) for row in selected}) != len(selected):
        raise ValueError("Duplicate native timing call indices")
    report = []
    for window, subset in (
        ("all_active_calls", selected),
        ("mature_tail_cycles_1901_2000", [row for row in selected if 1901 <= row["measurement_cycle"] <= 2000]),
    ):
        values = np.asarray([int(row["elapsed_ns"]) / 1e6 for row in subset])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Nonfinite or negative native timing sample")
        if len(values):
            stats = {"calls": len(values), "mean_ms": float(np.mean(values)),
                     "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                     "minimum_ms": float(np.min(values)), "maximum_ms": float(np.max(values))}
        else:
            # A short integration smoke cannot reach the GP's 32-observation
            # window or the publication's final 100-cycle reporting window.
            stats = {"calls": 0, "mean_ms": None, "median_ms": None, "p95_ms": None,
                     "minimum_ms": None, "maximum_ms": None}
        report.append({"topology": "C6o", "physical_seed": 1, "method": method["display"],
                       "native_method": method["native_method"], "run_id": output.name,
                       "operation": method["operation"], "window": window, "scope": method["scope"],
                       **stats,
                       "first_env_step": min((int(r["env_step"]) for r in subset), default=None),
                       "last_env_step": max((int(r["env_step"]) for r in subset), default=None),
                       "first_measurement_cycle": min((r["measurement_cycle"] for r in subset), default=None),
                       "last_measurement_cycle": max((r["measurement_cycle"] for r in subset), default=None),
                       "terminal_extra_calls_excluded": terminal_extra,
                       "source_trace": str(path.relative_to(output)), "source_trace_sha256": digest(path),
                       "status": "measured" if len(values) else "window_not_reached"})
    if math.isclose(duration, 150):
        # Full publication replay must reach both requested reporting windows.
        if not selected or report[1]["calls"] == 0:
            raise ValueError("Full replay did not reach a required reporting window")
    return report, {"trace_rows": len(rows), "selected_operation_calls": len(selected),
                    "last_env_step": max(int(r["env_step"]) for r in rows),
                    "terminal_extra_calls_excluded": terminal_extra}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/native_inference")
    parser.add_argument("--ns3-dir", type=Path,
                        default=Path(os.environ.get("NSTEST_NS3_DIR", str(ROOT / "work/ns-allinone-3.35/ns-3.35"))))
    parser.add_argument("--cpu", type=int, help="Allowed logical CPU; defaults to the first allowed CPU")
    parser.add_argument("--duration", type=float, default=150.0,
                        help="Simulated seconds; use 1.5 for an integration smoke only")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--timeout", type=float, default=1800.0, help="Wall-clock timeout in seconds per native run")
    args = parser.parse_args()
    if args.duration < .075 or not math.isclose(args.duration / INTERVAL, round(args.duration / INTERVAL), abs_tol=1e-8):
        parser.error("Duration must be a positive multiple of 75 ms")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Choose a new output directory to preserve existing measurements")
    ns3 = args.ns3_dir.resolve()
    binary = ns3 / "build/scratch/nsTest/nsTest"
    if not binary.is_file():
        parser.error("Build ns-3 first with python scripts/build_ns3.py, or supply --ns3-dir")
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    if cpu not in os.sched_getaffinity(0):
        parser.error("cpu is not in the current process's allowed affinity")
    os.sched_setaffinity(0, {cpu})
    output.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "code/nsTest/topos/C6o.json", binary]
    sources += [p for name in ("main.cc", "simulation.cc", "optimizers.cc", "optimizers.hh", "samplers.cc", "runtime_timing.hh")
                if (p := ns3 / "scratch/nsTest" / name).is_file()]
    pins = {str(p): digest(p) for p in sources}
    protocol = {"status": "running", "scope": "native C++ controller timing in a fresh packet-level network simulation",
                "topology": "C6o", "physical_seeds": [1], "agent_seed": 1,
                "simulated_seconds": args.duration, "measurement_cycle_seconds": INTERVAL,
                "full_publication_horizon": math.isclose(args.duration, 150), "logical_cpu": cpu,
                "before": metadata(), "binary": str(binary), "source_sha256": pins, "groups": [],
                "reporting_windows": ["all_active_calls", "mature_tail_cycles_1901_2000"],
                "timed_operations": "native proposals/action selection; INSPIRE GP prescription/consensus with a full 32-observation window, excluding fitting",
                "network_role": "ns-3 traffic supplies live measurements and controller history; simulator execution is outside each C++ operation timer",
                "isolated_working_directories": True}
    protocol_path = output / "native_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    reports, completion = [], []
    for method_id in args.methods:
        method = METHODS[method_id]
        working = output / "isolated" / method_id
        topology_directory = working / "scratch/nsTest/topos"
        data_directory = working / "scratch/nsTest/data"
        topology_directory.mkdir(parents=True)
        data_directory.mkdir(parents=True)
        shutil.copy2(ROOT / "code/nsTest/topos/C6o.json", topology_directory / "C6o.json")
        traces = output / "native_traces"
        traces.mkdir(exist_ok=True)
        trace = traces / f"{method['native_method']}_C6_seed001.tsv"
        suffix = "nativereplay" + method_id.replace("_", "")
        env = base_environment(ns3, method, args.duration, trace, suffix)
        protocol["groups"].append({"method_id": method_id, "method": method["display"],
                                   "environment": {k: v for k, v in env.items()
                                                   if k.startswith("NSTEST_") or k.endswith("_NUM_THREADS")},
                                   "cwd": str(working), "trace": str(trace.relative_to(output))})
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        print(f"Run {method['display']} on C6, seed 1, {args.duration} s; CPU {cpu}", flush=True)
        log_path = output / f"{method_id}.log"
        started = time.perf_counter()
        with log_path.open("wb") as log:
            process = subprocess.run([str(binary)], cwd=working, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
        elapsed = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(f"Native replay failed ({process.returncode}); see {log_path}")
        reward_files = list(data_directory.glob("*_rew.tsv"))
        if len(reward_files) != 1:
            raise ValueError(f"Require one isolated reward matrix: {data_directory}")
        reward = np.loadtxt(reward_files[0], delimiter="\t", skiprows=1, ndmin=2)
        expected = int(round(args.duration/INTERVAL)) + 1
        if reward.shape[0] != 1 or reward.shape[1] not in (expected, expected+1):
            raise ValueError("Incomplete native simulation measurement horizon")
        rows, counts = trace_summary(trace, method, args.duration, output)
        reports.extend(rows)
        completion.append({"method": method["display"], "status": "done", "wall_seconds": elapsed,
                           "measurement_cycles": reward.shape[1], **counts})
        print(f"Completed {method['display']}: {counts['selected_operation_calls']} active timed calls", flush=True)
    for source, pinned in pins.items():
        if digest(Path(source)) != pinned:
            raise ValueError(f"Replay source changed: {source}")
    write_csv(output / "native_baseline_latency.csv", reports)
    write_csv(output / "completion.csv", completion)
    protocol.update(status="complete", sources_unchanged=True, after=metadata(),
                    native_runs=len(completion), summary_rows=len(reports))
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
