#!/usr/bin/env python3
"""Run independent MLP or QNN Neural Bandit replicates concurrently.

Each worker delegates one seed to ``run_qnn_full.py`` and assigns a unique
``NSTEST_OUTPUT_SUFFIX`` plus bridge port.  The suffix prevents concurrent ns-3
processes from clearing or appending to one another's fixed output filenames.
The quantum model and ns-3 RNG are reset from the logical seed, so concurrency
changes wall-clock time only, not the experiment definition.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "work" / "runs"
RUNNER = REPO / "code" / "neural_bandit_bridge" / "run_qnn_full.py"
METRICS = ("rew", "fair", "cum", "aps", "stas", "conf", "pers", "state", "agent_state")


def scenario_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "off", "no")


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(path: Path, message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def completed_seeds(run_dir: Path, topo: str) -> set[int]:
    path = run_dir / "progress.csv"
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            int(row["seed"])
            for row in csv.DictReader(handle)
            if row.get("status") == "done"
            and row.get("topo") == topo
            and row.get("seed")
        }


def child_dir(run_dir: Path, seed: int) -> Path:
    return run_dir / "shards" / f"seed_{seed:03d}"


def child_command(args: argparse.Namespace, seed: int, port: int) -> list[str]:
    child_id = str(child_dir(args.run_dir, seed).relative_to(RUNS))
    suffix = f"qnnp{args.base_port}s{seed}"
    command = [
        "/usr/bin/nice", "-n", str(args.nice),
        sys.executable, str(RUNNER),
        "--run-id", child_id,
        "--topos", args.topo,
        "--seed-start", str(seed),
        "--seed-end", str(seed),
        "--agent-seed-offset", str(args.agent_seed_offset),
        "--duration", str(args.duration),
        "--test-duration", str(args.test_duration),
        "--rate-manager", args.rate_manager,
        "--transport", args.transport,
        "--port", str(port),
        "--output-suffix", suffix,
        "--sampler", args.sampler,
        "--entry", args.entry,
        "--hcm-profile", args.hcm_profile,
        "--hcm-global-epsilon", str(args.hcm_global_epsilon),
        "--hcm-restart-patience", str(args.hcm_restart_patience),
        "--hcm-restart-burst", str(args.hcm_restart_burst),
        "--hcm-restart-max-bursts", str(args.hcm_restart_max_bursts),
        "--hcm-restart-min-delta", str(args.hcm_restart_min_delta),
        "--hcm-restart-target", str(args.hcm_restart_target),
        "--hcm-batch-explore-fraction", str(args.hcm_batch_explore_fraction),
        "--hcm-batch-min-radius", str(args.hcm_batch_min_radius),
        "--hcm-batch-max-radius", str(args.hcm_batch_max_radius),
        "--adhoc-tiebreak-weight", str(args.adhoc_tiebreak_weight),
        "--surrogate", args.surrogate,
        "--policy", args.policy,
        "--matched-params", str(args.matched_params),
        "--ucb-beta", str(args.ucb_beta),
        "--gp-features", str(args.gp_features),
        "--gp-length-scale", str(args.gp_length_scale),
        "--gp-noise", str(args.gp_noise),
        "--hidden", str(args.hidden),
        "--candidates", str(args.candidates),
        "--novel-candidates", str(args.novel_candidates),
        "--acquisition-preset", "custom",
        "--feature-dim", str(args.feature_dim),
        "--qnn-layers", str(args.qnn_layers),
        "--qnn-n-qubits", str(args.qnn_n_qubits),
        "--qnn-max-z-order", str(args.qnn_max_z_order),
        "--qnn-architecture", args.qnn_architecture,
        "--qnn-unitary-fusion" if args.qnn_architecture == "legacy" else "--no-qnn-unitary-fusion",
        "--no-qnn-edge-specific",
        "--no-qnn-input-tanh" if args.qnn_architecture == "qcnn_angle_reupload" else "--qnn-input-tanh",
        "--cold-start", str(args.cold_start),
        "--retrain-interval", str(args.retrain_interval),
        "--early-retrain-interval", str(args.early_retrain_interval),
        "--max-retrains", str(args.max_retrains),
        "--freeze-after-step", str(args.freeze_after_step),
        "--train-epochs", str(args.train_epochs),
        "--update-epochs", str(args.update_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--quantum-lr", str(args.quantum_lr),
        "--quantum-weight-decay", str(args.quantum_weight_decay),
        "--checkpoint-every", "0",
        "--save-final-checkpoint" if args.save_final_checkpoint else "--no-save-final-checkpoint",
        "--restore-checkpoint-state" if args.restore_checkpoint_state else "--no-restore-checkpoint-state",
        "--freeze-loaded-model" if args.freeze_loaded_model else "--no-freeze-loaded-model",
        "--qnn-frozen-fusion" if args.qnn_frozen_fusion else "--no-qnn-frozen-fusion",
        "--qnn-frozen-torchscript" if args.qnn_frozen_torchscript else "--no-qnn-frozen-torchscript",
        "--feature-cache-size", str(args.feature_cache_size),
        "--acquisition-top-k", str(args.acquisition_top_k),
        "--acquisition-temperature", str(args.acquisition_temperature),
        "--acquisition-epsilon", str(args.acquisition_epsilon),
        "--posterior-scale", str(args.posterior_scale),
        "--feature-diagnostics" if args.feature_diagnostics else "--no-feature-diagnostics",
        "--torch-threads", "1",
        "--max-seed-attempts", str(args.child_attempts),
        "--no-write-comparisons",
    ]
    if args.load_checkpoint is not None:
        command.extend(["--load-checkpoint", str(args.load_checkpoint)])
    return command


def source_dir_for_seed(
    args: argparse.Namespace,
    seed: int,
    reused: set[int],
) -> tuple[Path, bool]:
    if seed in reused:
        # Sequential runs keep one seed below ``per_seed/<topo>`` whereas
        # bounded-parallel runs keep it below ``shards``.  Accept both so a
        # successful pilot can be reused verbatim in the 25-seed validation
        # instead of being simulated again.
        sequential = args.reuse_run_dir / "per_seed" / args.topo / f"seed_{seed:03d}"
        if sequential.is_dir():
            return sequential, True
        parallel = args.reuse_run_dir / "shards" / f"seed_{seed:03d}" / "clean_data"
        if parallel.is_dir():
            return parallel, True
        raise RuntimeError(
            f"seed {seed} is marked complete in {args.reuse_run_dir}, "
            "but neither the sequential nor parallel seed directory exists"
        )
    return child_dir(args.run_dir, seed) / "clean_data", False


def standardized_name(path: Path, raw: bool, surrogate: str) -> str:
    if raw and surrogate == "qnn":
        return path.name.replace("_NB_", "_QNN_", 1)
    return path.name


def validate_and_merge(
    args: argparse.Namespace,
    seeds: list[int],
    reused: set[int],
) -> None:
    expected_steps = int(args.duration / args.test_duration) + 1
    rows_by_name: dict[str, tuple[str, list[str]]] = {}
    for seed in seeds:
        source_dir, raw = source_dir_for_seed(args, seed, reused)
        if not source_dir.is_dir():
            raise RuntimeError(f"missing source directory for seed {seed}: {source_dir}")
        for metric in METRICS:
            matches = [
                path
                for path in sorted(source_dir.glob(f"*_{metric}.tsv"))
                if not (
                    metric == "state" and path.name.endswith("_agent_state.tsv")
                )
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"seed {seed} metric {metric}: expected one file in {source_dir}, "
                    f"found {len(matches)}"
                )
            src = matches[0]
            lines = src.read_text(encoding="utf-8").splitlines()
            data_rows = [line for line in lines[1:] if line.strip()]
            if len(data_rows) != 1:
                raise RuntimeError(
                    f"seed {seed} metric {metric}: expected one row, found {len(data_rows)}"
                )
            if len(data_rows[0].split("\t")) != expected_steps:
                raise RuntimeError(
                    f"seed {seed} metric {metric}: expected {expected_steps} steps, "
                    f"found {len(data_rows[0].split(chr(9)))}"
                )
            name = standardized_name(src, raw, args.surrogate)
            header, aggregate_rows = rows_by_name.setdefault(name, (lines[0], []))
            if header != lines[0]:
                raise RuntimeError(f"header mismatch while merging {name}")
            aggregate_rows.append(data_rows[0])

    clean_dir = args.run_dir / "clean_data"
    stage_dir = args.run_dir / "clean_data.staging"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for name, (header, rows) in rows_by_name.items():
        if len(rows) != len(seeds):
            raise RuntimeError(f"{name}: expected {len(seeds)} rows, found {len(rows)}")
        (stage_dir / name).write_text(
            header + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
        )
    if clean_dir.exists():
        backup = args.run_dir / f"clean_data.backup-{time.strftime('%Y%m%d-%H%M%S')}"
        clean_dir.rename(backup)
    stage_dir.rename(clean_dir)

    progress = args.run_dir / "progress.csv"
    with progress.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("timestamp", "topo", "seed", "status", "source")
        )
        writer.writeheader()
        for seed in seeds:
            writer.writerow({
                "timestamp": stamp(),
                "topo": args.topo,
                "seed": seed,
                "status": "done",
                "source": "reused-sequential" if seed in reused else "parallel-shard",
            })


def write_manifest(args: argparse.Namespace, reused: set[int]) -> None:
    lines = [
        f"run_id={args.run_dir.name}",
        f"mode={args.surrogate.upper()} Neural Bandit bounded parallel seeds with isolated ns-3 suffixes",
        f"topos={args.topo}",
        f"seeds={args.seed_start}..{args.seed_end}",
        f"agent_seed_offset={args.agent_seed_offset}",
        f"jobs={args.jobs}",
        f"reused_seeds={','.join(map(str, sorted(reused)))}",
        f"duration={args.duration}",
        f"test_duration={args.test_duration}",
        f"rate_manager={args.rate_manager}",
        f"transport={args.transport}",
        f"dynamic={int(scenario_flag_enabled('NSTEST_DYNAMIC'))}",
        f"mobility={int(scenario_flag_enabled('NSTEST_MOBILITY'))}",
        f"dynamic_profile={os.environ.get('NSTEST_DYNAMIC_PROFILE', '')}",
        f"sampler={args.sampler}",
        f"entry={args.entry}",
        f"hcm_profile={args.hcm_profile}",
        f"hcm_global_epsilon={args.hcm_global_epsilon}",
        f"hcm_restart_patience={args.hcm_restart_patience}",
        f"hcm_restart_burst={args.hcm_restart_burst}",
        f"hcm_restart_max_bursts={args.hcm_restart_max_bursts}",
        f"hcm_restart_min_delta={args.hcm_restart_min_delta}",
        f"hcm_restart_target={args.hcm_restart_target}",
        f"hcm_batch_explore_fraction={args.hcm_batch_explore_fraction}",
        f"hcm_batch_min_radius={args.hcm_batch_min_radius}",
        f"hcm_batch_max_radius={args.hcm_batch_max_radius}",
        f"adhoc_tiebreak_weight={args.adhoc_tiebreak_weight}",
        "reward=ADHOC",
        f"surrogate={args.surrogate}",
        f"policy={args.policy}",
        f"backend={'rff-rbf-gp' if args.policy == 'gp_ts' else ('numpy-per-arm' if args.policy == 'per_arm_ts' else ('torch-statevector' if args.surrogate.startswith('qnn') else 'torch-mlp'))}",
        f"hidden={args.hidden}",
        f"matched_params={args.matched_params}",
        f"ucb_beta={args.ucb_beta}",
        f"gp_features={args.gp_features}",
        f"gp_length_scale={args.gp_length_scale}",
        f"gp_noise={args.gp_noise}",
        f"candidates={args.candidates}",
        f"novel_candidates={args.novel_candidates}",
        f"acquisition_preset={args.acquisition_preset}",
        f"feature_dim={args.feature_dim}",
        f"qnn_layers={args.qnn_layers}",
        f"qnn_n_qubits={args.qnn_n_qubits}",
        f"qnn_max_z_order={args.qnn_max_z_order}",
        f"qnn_architecture={args.qnn_architecture}",
        f"qnn_unitary_fusion={args.qnn_architecture == 'legacy'}",
        "qnn_edge_specific=False",
        f"qnn_input_tanh={args.qnn_architecture != 'qcnn_angle_reupload'}",
        f"cold_start={args.cold_start}",
        f"retrain_interval={args.retrain_interval}",
        f"early_retrain_interval={args.early_retrain_interval}",
        f"max_retrains={args.max_retrains}",
        f"freeze_after_step={args.freeze_after_step}",
        f"surrogate_policy={f'freeze-after-step-{args.freeze_after_step}' if args.freeze_after_step >= 0 else ('periodic-online' if args.max_retrains < 0 else ('fixed-random-features' if args.max_retrains == 0 else f'bootstrap-{args.max_retrains}-then-frozen'))}",
        f"train_epochs={args.train_epochs}",
        f"update_epochs={args.update_epochs}",
        f"batch_size={args.batch_size}",
        f"lr={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"quantum_lr={args.quantum_lr}",
        f"quantum_weight_decay={args.quantum_weight_decay}",
        f"acquisition_top_k={args.acquisition_top_k}",
        f"acquisition_temperature={args.acquisition_temperature}",
        f"acquisition_epsilon={args.acquisition_epsilon}",
        f"posterior_scale={args.posterior_scale}",
        "prior_lambda=1.0",
        "prior_a=0.5",
        "prior_b=0.025",
        f"feature_diagnostics={args.feature_diagnostics}",
        f"load_checkpoint={args.load_checkpoint or ''}",
        f"restore_checkpoint_state={args.restore_checkpoint_state}",
        f"save_final_checkpoint={args.save_final_checkpoint}",
        f"freeze_loaded_model={args.freeze_loaded_model}",
        f"qnn_frozen_fusion={args.qnn_frozen_fusion}",
        f"qnn_frozen_torchscript={args.qnn_frozen_torchscript}",
        f"feature_cache_size={args.feature_cache_size}",
    ]
    (args.run_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--topo", required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=25)
    parser.add_argument("--agent-seed-offset", type=int, default=1_000_003)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--reuse-run-dir", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--test-duration", type=float, default=0.075)
    parser.add_argument("--rate-manager", choices=("MINSTREL", "CONSTANT", "IDEAL"), default="MINSTREL")
    parser.add_argument("--transport", choices=("udp", "tcp"), default="udp")
    parser.add_argument("--sampler", choices=("HGM", "HCM"), default="HCM")
    parser.add_argument("--entry", choices=("DEF", "DEGA"), default="DEGA")
    parser.add_argument(
        "--hcm-profile",
        choices=("batch", "paper", "author", "author-restart", "author-fixed", "author-fixed-restart"),
        default="batch",
        help="HCM implementation profile; ignored when --sampler is HGM.",
    )
    parser.add_argument("--hcm-global-epsilon", type=float, default=0.0)
    parser.add_argument("--hcm-restart-patience", type=int, default=48)
    parser.add_argument("--hcm-restart-burst", type=int, default=16)
    parser.add_argument("--hcm-restart-max-bursts", type=int, default=4)
    parser.add_argument("--hcm-restart-min-delta", type=float, default=0.01)
    parser.add_argument("--hcm-restart-target", type=float, default=0.89)
    parser.add_argument("--hcm-batch-explore-fraction", type=float, default=0.0)
    parser.add_argument("--hcm-batch-min-radius", type=float, default=5.0)
    parser.add_argument("--hcm-batch-max-radius", type=float, default=5.0)
    parser.add_argument("--adhoc-tiebreak-weight", type=float, default=0.0)
    parser.add_argument(
        "--surrogate",
        choices=("mlp_norm", "qnn"),
        default="qnn",
    )
    parser.add_argument("--policy", choices=("neural_ts",), default="neural_ts")
    parser.add_argument("--matched-params", type=int, default=0)
    parser.add_argument("--ucb-beta", type=float, default=1.0)
    parser.add_argument("--gp-features", type=int, default=128)
    parser.add_argument("--gp-length-scale", type=float, default=0.5)
    parser.add_argument("--gp-noise", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, choices=(64,), default=64)
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--novel-candidates", type=int, default=256)
    parser.add_argument(
        "--acquisition-preset",
        choices=("custom", "neural-ring-202606"),
        default="custom",
    )
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--qnn-layers", type=int, choices=(3,), default=3)
    parser.add_argument("--qnn-n-qubits", type=int, choices=(5,), default=5)
    parser.add_argument("--qnn-max-z-order", type=int, choices=(5,), default=5)
    parser.add_argument(
        "--qnn-architecture",
        choices=("qcnn_shared_reupload",),
        default="qcnn_shared_reupload",
    )
    parser.add_argument("--cold-start", type=int, default=20)
    parser.add_argument("--retrain-interval", type=int, default=20)
    parser.add_argument("--early-retrain-interval", type=int, default=10)
    parser.add_argument(
        "--max-retrains",
        type=int,
        default=-1,
        help=(
            "Maximum surrogate retrains: -1 keeps periodic online training, "
            "0 fixes random features, and 1 bootstrap-trains once then freezes."
        ),
    )
    parser.add_argument(
        "--freeze-after-step",
        type=int,
        default=-1,
        help="Freeze after a shared environment-step budget; -1 disables.",
    )
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=0,
        help="Epochs after the initial fit; 0 preserves the legacy full retrain.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--quantum-lr", type=float, default=3.0e-3)
    parser.add_argument("--quantum-weight-decay", type=float, default=0.0)
    parser.add_argument("--acquisition-top-k", type=int, default=1)
    parser.add_argument("--acquisition-temperature", type=float, default=0.0)
    parser.add_argument("--acquisition-epsilon", type=float, default=0.0)
    parser.add_argument("--posterior-scale", type=float, default=1.0)
    parser.add_argument(
        "--feature-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect per-round posterior condition number and feature norm in each seed shard.",
    )
    parser.add_argument("--child-attempts", type=int, default=3)
    parser.add_argument("--orchestrator-attempts", type=int, default=3)
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Load one separately pretrained checkpoint in every seed worker.",
    )
    parser.add_argument(
        "--restore-checkpoint-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore complete saved policy state when loading a checkpoint.",
    )
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save one final model and Bayesian posterior checkpoint for every seed.",
    )
    parser.add_argument(
        "--freeze-loaded-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the loaded surrogate and keep only BLR updates online.",
    )
    parser.add_argument(
        "--qnn-frozen-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--qnn-frozen-torchscript",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--feature-cache-size", type=int, default=65536)
    args = parser.parse_args()
    if args.acquisition_preset == "neural-ring-202606":
        if args.surrogate != "mlp":
            parser.error("neural-ring-202606 is an audited MLP Neural Bandit preset, not a QNN preset")
        args.candidates = 2048
        args.novel_candidates = 32
        args.feature_dim = 16
        args.acquisition_top_k = 32
        args.acquisition_temperature = 0.7
        args.acquisition_epsilon = 0.0
    if not 0.0 <= args.acquisition_epsilon <= 1.0:
        parser.error("--acquisition-epsilon must be in [0, 1]")
    if args.posterior_scale < 0.0:
        parser.error("--posterior-scale must be >= 0")
    if args.ucb_beta < 0.0 or args.matched_params < 0:
        parser.error("--ucb-beta and --matched-params must be non-negative")
    if args.gp_features < 1 or args.gp_length_scale <= 0 or args.gp_noise <= 0:
        parser.error("GP feature count, length scale, and noise must be positive")
    if args.max_retrains < -1:
        parser.error("--max-retrains must be -1, 0, or a positive integer")
    if args.freeze_after_step < -1:
        parser.error("--freeze-after-step must be -1 or a non-negative integer")
    if args.update_epochs < 0:
        parser.error("--update-epochs must be >= 0")
    if args.feature_cache_size < 0:
        parser.error("--feature-cache-size must be >= 0")
    if args.load_checkpoint is not None:
        args.load_checkpoint = args.load_checkpoint.expanduser().resolve()
        if not args.load_checkpoint.is_file():
            parser.error(f"--load-checkpoint does not exist: {args.load_checkpoint}")
    if not 0.0 <= args.hcm_global_epsilon <= 1.0:
        parser.error("--hcm-global-epsilon must be in [0, 1]")
    if min(args.hcm_restart_patience, args.hcm_restart_burst, args.hcm_restart_max_bursts) < 1:
        parser.error("HCM restart patience, burst, and max-bursts must be >= 1")
    if not 0.0 <= args.hcm_restart_min_delta <= 1.0:
        parser.error("--hcm-restart-min-delta must be in [0, 1]")
    if not 0.0 <= args.hcm_restart_target <= 1.0:
        parser.error("--hcm-restart-target must be in [0, 1]")
    if not 0.0 <= args.hcm_batch_explore_fraction <= 1.0:
        parser.error("--hcm-batch-explore-fraction must be in [0, 1]")
    if not 1.0 <= args.hcm_batch_min_radius <= 20.0:
        parser.error("--hcm-batch-min-radius must be in [1, 20]")
    if not args.hcm_batch_min_radius <= args.hcm_batch_max_radius <= 20.0:
        parser.error("--hcm-batch-max-radius must be in [min-radius, 20]")
    if not 0.0 <= args.adhoc_tiebreak_weight <= 0.05:
        parser.error("--adhoc-tiebreak-weight must be in [0, 0.05]")
    args.run_dir = RUNS / args.run_id
    return args


def main() -> int:
    args = parse_args()
    if args.seed_end < args.seed_start or args.jobs < 1:
        raise SystemExit("invalid seed range or --jobs")
    if (args.sampler, args.entry) not in {("HGM", "DEF"), ("HCM", "DEGA")}:
        raise SystemExit("use HGM+DEF or HCM+DEGA")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    runner_log = args.run_dir / "runner.log"
    seeds = list(range(args.seed_start, args.seed_end + 1))

    reused: set[int] = set()
    if args.reuse_run_dir is not None:
        args.reuse_run_dir = args.reuse_run_dir.resolve()
        reused = completed_seeds(args.reuse_run_dir, args.topo).intersection(seeds)
    write_manifest(args, reused)

    already_done = {
        seed
        for seed in seeds
        if seed not in reused
        and completed_seeds(child_dir(args.run_dir, seed), args.topo) == {seed}
    }
    pending = [seed for seed in seeds if seed not in reused and seed not in already_done]
    log(
        runner_log,
        f"start topo={args.topo} seeds={args.seed_start}..{args.seed_end} "
        f"jobs={args.jobs} reused={sorted(reused)} resumed={sorted(already_done)}",
    )

    attempts = {seed: 0 for seed in pending}
    while pending:
        batch = pending[: args.jobs]
        processes: list[tuple[int, subprocess.Popen, object]] = []
        for slot, seed in enumerate(batch):
            shard = child_dir(args.run_dir, seed)
            shard.mkdir(parents=True, exist_ok=True)
            handle = (shard / "runner.log").open("a", encoding="utf-8")
            port = args.base_port + slot
            command = child_command(args, seed, port)
            log(runner_log, f"launch seed={seed} port={port} attempt={attempts[seed] + 1}")
            proc = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((seed, proc, handle))

        for seed, proc, handle in processes:
            rc = proc.wait()
            handle.close()
            if rc == 0 and completed_seeds(child_dir(args.run_dir, seed), args.topo) == {seed}:
                pending.remove(seed)
                log(runner_log, f"done seed={seed}")
                continue
            attempts[seed] += 1
            log(runner_log, f"retry seed={seed} rc={rc} attempt={attempts[seed]}")
            if attempts[seed] >= args.orchestrator_attempts:
                log(runner_log, f"FAILED seed={seed}")
                return 1

    validate_and_merge(args, seeds, reused)
    log(runner_log, f"complete topo={args.topo} rows={len(seeds)} steps={int(args.duration / args.test_duration) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
