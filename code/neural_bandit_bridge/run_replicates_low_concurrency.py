#!/usr/bin/env python3
"""Run ns-3 replicates with bounded concurrency and unique output files."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
NS3_DIR = Path(os.environ.get("NSTEST_NS3_DIR", str(REPO / "work" / "ns-allinone-3.35" / "ns-3.35"))).resolve()
DATA_DIR = NS3_DIR / "scratch" / "nsTest" / "data"
METHODS = (
    ("idle", "IDLE", "UNI", "DEF"),
    ("egreedy", "EGREED", "UNI", "DEF"),
    ("tgnorm_paper", "TGNORM", "HGM", "DEF"),
    ("tgnorm_ring", "TGNORM", "HCM", "DEGA"),
    ("inspire", "INSPIRE", "UNI", "DEF"),
)
DEFAULT_METHODS = tuple(method[0] for method in METHODS)
METHOD_BY_LABEL = {method[0]: method for method in METHODS}
METRICS = ("rew", "fair", "cum", "aps", "stas", "pers", "conf", "agent_state", "state")


def stamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def append_log(path: Path, message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def source_files(
    topo: str,
    optimizer: str,
    sampler: str,
    reward: str,
    rate: str,
    suffix: str,
    duration: float,
    test_duration: float,
) -> list[Path]:
    sampler_id = "HGMT" if sampler == "HGM" else sampler
    prefix = (
        f"{topo}_*_{duration:.1f}_{optimizer}_{sampler_id}_{reward}_"
        f"{test_duration:.3f}_BOTH_{rate}_{suffix}_"
    )
    return [p for p in DATA_DIR.glob(prefix + "*.tsv") if any(p.name.endswith(f"_{m}.tsv") for m in METRICS)]


def run_process(env: dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(NS3_DIR / "build" / "scratch" / "nsTest" / "nsTest")],
            cwd=NS3_DIR,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--rate-manager", default="IDEAL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=22)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="First logical replicate seed; --replicates is the number of seeds.",
    )
    parser.add_argument(
        "--physical-seed",
        type=int,
        default=None,
        help="Fix the ns-3 environment seed while replicate ids still vary the optimizer seed.",
    )
    parser.add_argument(
        "--static-config",
        default=None,
        help="Optional semicolon-separated per-AP sens,power pairs for replay diagnostics.",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--run-tag",
        default="fair20260902",
        help="Alphanumeric tag used to isolate raw ns-3 output files.",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--test-duration", type=float, default=0.075)
    parser.add_argument("--reward", choices=("LOGPF", "ADHOC"), default="LOGPF")
    parser.add_argument(
        "--hcm-profile",
        choices=("batch", "paper", "author", "author-restart", "author-fixed", "author-fixed-restart"),
        default="batch",
        help="HCM implementation profile; ignored by non-HCM methods.",
    )
    parser.add_argument("--hcm-global-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--hcmts-acquisition-epsilon",
        type=float,
        default=0.0,
        help="Optional unseen-candidate exploration floor for the per-arm Thompson-Gamma-Normal optimizer.",
    )
    parser.add_argument("--hcm-restart-patience", type=int, default=48)
    parser.add_argument("--hcm-restart-burst", type=int, default=16)
    parser.add_argument("--hcm-restart-max-bursts", type=int, default=4)
    parser.add_argument("--hcm-restart-min-delta", type=float, default=0.01)
    parser.add_argument("--hcm-restart-target", type=float, default=0.89)
    parser.add_argument("--adhoc-tiebreak-weight", type=float, default=0.0)
    parser.add_argument(
        "--agent-seed-offset",
        type=int,
        default=1_000_003,
        help="Deterministic offset separating optimizer/HCM RNG from the ns-3 environment RNG.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_BY_LABEL),
        default=DEFAULT_METHODS,
        help="Method labels to run (default: the five main classic methods).",
    )
    args = parser.parse_args()
    if not args.run_tag.isalnum():
        raise SystemExit("--run-tag must be alphanumeric")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest_dir = args.output_dir / "data" / args.topology
    dest_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "runner.log"
    env_base = os.environ.copy()
    env_base["PATH"] = os.environ.get("PATH", "")
    env_base["LD_LIBRARY_PATH"] = str(NS3_DIR / "build" / "lib") + os.pathsep + env_base.get("LD_LIBRARY_PATH", "")
    if not 0.0 <= args.hcm_global_epsilon <= 1.0:
        raise SystemExit("--hcm-global-epsilon must be in [0, 1]")
    if not 0.0 <= args.hcmts_acquisition_epsilon <= 1.0:
        raise SystemExit("--hcmts-acquisition-epsilon must be in [0, 1]")
    if min(args.hcm_restart_patience, args.hcm_restart_burst, args.hcm_restart_max_bursts) < 1:
        raise SystemExit("HCM restart patience, burst, and max-bursts must be >= 1")
    if not 0.0 <= args.hcm_restart_min_delta <= 1.0:
        raise SystemExit("--hcm-restart-min-delta must be in [0, 1]")
    if not 0.0 <= args.hcm_restart_target <= 1.0:
        raise SystemExit("--hcm-restart-target must be in [0, 1]")
    if not 0.0 <= args.adhoc_tiebreak_weight <= 0.05:
        raise SystemExit("--adhoc-tiebreak-weight must be in [0, 0.05]")
    append_log(log_path, f"settings topology={args.topology} rate={args.rate_manager} duration={args.duration} seed_start={args.seed_start} reps={args.replicates} jobs={args.jobs} reward={args.reward} hcm_profile={args.hcm_profile} hcm_global_epsilon={args.hcm_global_epsilon} hcmts_acquisition_epsilon={args.hcmts_acquisition_epsilon} hcm_restart_patience={args.hcm_restart_patience} hcm_restart_burst={args.hcm_restart_burst} hcm_restart_max_bursts={args.hcm_restart_max_bursts} hcm_restart_min_delta={args.hcm_restart_min_delta} hcm_restart_target={args.hcm_restart_target} adhoc_tiebreak_weight={args.adhoc_tiebreak_weight} physical_seed={args.physical_seed} static_config={args.static_config or ''} agent_seed_offset={args.agent_seed_offset} methods={','.join(args.methods)}")

    for label in args.methods:
        _, optimizer, sampler, entry = METHOD_BY_LABEL[label]
        method_dir = dest_dir / label
        method_dir.mkdir(parents=True, exist_ok=True)
        for old in method_dir.glob("*.tsv"):
            old.unlink()
        append_log(log_path, f"START method={label}")
        pending = {
            seed: 0
            for seed in range(args.seed_start, args.seed_start + args.replicates)
        }
        while pending:
            batch = list(pending.items())[: max(1, args.jobs)]
            processes: list[tuple[int, int, str, subprocess.Popen]] = []
            logs: list[object] = []
            for seed, attempt in batch:
                suffix = f"{args.run_tag}{label}rep{seed}try{attempt}"
                env = env_base.copy()
                env.update({
                    "NSTEST_TOPO": args.topology,
                    "NSTEST_OPTIMIZER": optimizer,
                    "NSTEST_SAMPLER": sampler,
                    "NSTEST_ENTRY": entry,
                    "NSTEST_REWARD": args.reward,
                    "NSTEST_RATE_MANAGER": args.rate_manager,
                    "NSTEST_NSIMULATIONS": "1",
                    "NSTEST_SEED": str(args.physical_seed if args.physical_seed is not None else seed),
                    "NSTEST_AGENT_SEED": str(seed + args.agent_seed_offset),
                    "NSTEST_DURATION": str(args.duration),
                    "NSTEST_TEST_DURATION": str(args.test_duration),
                    "NSTEST_OUTPUT_SUFFIX": suffix,
                    "NSTEST_HCM_PROFILE": args.hcm_profile,
                    "NSTEST_HCM_GLOBAL_EPS": str(args.hcm_global_epsilon),
                    "NSTEST_TGNORM_ACQUISITION_EPS": str(args.hcmts_acquisition_epsilon),
                    "NSTEST_TGNORM_CANDIDATES": "512",
                    "NSTEST_TGNORM_NOVEL_CANDIDATES": "128",
                    "NSTEST_TGNORM_CANDIDATE_ATTEMPTS": "50000",
                    "NSTEST_TGNORM_COLD_START": "20",
                    "NSTEST_HCM_RESTART_PATIENCE": str(args.hcm_restart_patience),
                    "NSTEST_HCM_RESTART_BURST": str(args.hcm_restart_burst),
                    "NSTEST_HCM_RESTART_MAX_BURSTS": str(args.hcm_restart_max_bursts),
                    "NSTEST_HCM_RESTART_MIN_DELTA": str(args.hcm_restart_min_delta),
                    "NSTEST_HCM_RESTART_TARGET": str(args.hcm_restart_target),
                    "NSTEST_ADHOC_TIEBREAK_WEIGHT": str(args.adhoc_tiebreak_weight),
                })
                if args.static_config:
                    env["NSTEST_STATIC_CONFIG"] = args.static_config
                child_log = (args.output_dir / "logs").joinpath(f"{label}_rep{seed}.log")
                child_log.parent.mkdir(parents=True, exist_ok=True)
                handle = child_log.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    [str(NS3_DIR / "build" / "scratch" / "nsTest" / "nsTest")],
                    cwd=NS3_DIR,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                processes.append((seed, attempt, suffix, proc))
                logs.append(handle)
            for seed, attempt, suffix, proc in processes:
                rc = proc.wait()
                logs.pop(0).close()
                files = source_files(
                    args.topology,
                    optimizer,
                    sampler,
                    args.reward,
                    args.rate_manager,
                    suffix,
                    args.duration,
                    args.test_duration,
                )
                rows_by_metric: dict[str, tuple[Path, str, str]] = {}
                for src in files:
                    lines = src.read_text(encoding="utf-8").splitlines()
                    rows = [line for line in lines[1:] if line.strip()]
                    metric = next(
                        (m for m in METRICS if src.name.endswith(f"_{m}.tsv")),
                        None,
                    )
                    if metric is None or len(rows) != 1:
                        continue
                    if metric in rows_by_metric:
                        rows_by_metric.pop(metric)
                        continue
                    rows_by_metric[metric] = (src, lines[0], rows[0])

                widths = {len(row.split("\t")) for _, _, row in rows_by_metric.values()}
                if rc != 0 or len(rows_by_metric) != len(METRICS) or len(widths) != 1:
                    if attempt + 1 < args.max_attempts:
                        pending[seed] = attempt + 1
                        append_log(
                            log_path,
                            f"retry method={label} seed={seed} rc={rc} "
                            f"metrics={len(rows_by_metric)}/{len(METRICS)} "
                            f"attempt={attempt + 1}",
                        )
                        continue
                    append_log(
                        log_path,
                        f"FAIL method={label} seed={seed} rc={rc} "
                        f"metrics={len(rows_by_metric)}/{len(METRICS)}",
                    )
                    return 1

                copied = 0
                for src, header, row in rows_by_metric.values():
                    dst_name = re.sub(rf"_{re.escape(suffix)}(?=_(?:{'|'.join(METRICS)})\.tsv$)", "", src.name)
                    dst = method_dir / dst_name
                    if not dst.exists():
                        dst.write_text(header + "\n" + row + "\n", encoding="utf-8")
                    else:
                        with dst.open("a", encoding="utf-8") as handle:
                            handle.write(row + "\n")
                    copied += 1
                del pending[seed]
                append_log(log_path, f"replicate method={label} seed={seed} rc={rc} files={copied}")
        rew = list(method_dir.glob("*_rew.tsv"))
        if not rew:
            append_log(log_path, f"FAIL method={label}: no reward output")
            return 1
        rows = len(rew[0].read_text(encoding="utf-8").splitlines()) - 1
        append_log(log_path, f"DONE method={label} reward_rows={rows}")
    append_log(log_path, "RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
