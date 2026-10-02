#!/usr/bin/env python3
"""Run an MLP or QNN Neural Bandit benchmark sequentially and resumably.

The ns-3 driver writes NB output to a fixed data directory with the same file
names used by the MLP Neural Bandit. This runner copies each single-seed result
into a dedicated run directory, appends it to an aggregate TSV, and restores the
main data directory when the run ends.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
NS3_DIR = Path(os.environ.get("NSTEST_NS3_DIR", str(REPO / "work" / "ns-allinone-3.35" / "ns-3.35"))).resolve()
NS3_DATA = NS3_DIR / "scratch" / "nsTest" / "data"

DEFAULT_TOPOS = ("C6o", "MER_FLOORS_CH20_S5", "MER_FLOORS_BAD_DIM")
METRICS = ("rew", "fair", "cum", "aps", "stas", "conf", "pers", "state", "agent_state")


def scenario_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "off", "no")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def wait_for_port(host: str, port: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    last_error: OSError | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"server {host}:{port} did not become ready: {last_error}")


def sampler_label(sampler: str) -> str:
    return "HGMT" if sampler == "HGM" else sampler


def output_glob(
    topo: str,
    duration: float,
    metric: str,
    sampler: str,
    entry: str,
    output_suffix: str | None = None,
    test_duration: float = 0.075,
    rate_manager: str = "MINSTREL",
    transport: str = "udp",
) -> str:
    suffix = f"_{output_suffix}" if output_suffix else ""
    transport_suffix = "_TCP" if transport == "tcp" else ""
    return (
        f"{topo}_*_20_{entry}_STD_{duration:.1f}_NB_{sampler_label(sampler)}_ADHOC_"
        f"{test_duration:g}_BOTH_{rate_manager}{transport_suffix}{suffix}_{metric}.tsv"
    )


def qnn_output_files(
    topo: str,
    duration: float,
    sampler: str,
    entry: str,
    output_suffix: str | None = None,
    test_duration: float = 0.075,
    rate_manager: str = "MINSTREL",
    transport: str = "udp",
) -> list[Path]:
    files: list[Path] = []
    for metric in METRICS:
        files.extend(
            path for path in sorted(NS3_DATA.glob(
                output_glob(topo, duration, metric, sampler, entry, output_suffix,
                            test_duration, rate_manager, transport)
            )) if metric != "state" or not path.name.endswith("_agent_state.tsv")
        )
    return files


def clean_output_name(
    raw_name: str,
    surrogate: str,
    output_suffix: str | None = None,
) -> str:
    """Name QNN files distinctly while retaining NB names for the MLP."""
    name = raw_name.replace("_NB_", "_QNN_", 1) if surrogate == "qnn" else raw_name
    if output_suffix:
        marker = f"_{output_suffix}_"
        if marker not in name:
            raise RuntimeError(f"output suffix {output_suffix!r} missing from {raw_name}")
        name = name.replace(marker, "_", 1)
    return name


def read_tsv_rows(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"{path} has no data rows")
    return lines[0], [line for line in lines[1:] if line.strip()]


def append_single_seed_file(src: Path, dst: Path) -> None:
    header, rows = read_tsv_rows(src)
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one data row in {src}, got {len(rows)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_text(header + "\n" + rows[0] + "\n", encoding="utf-8")
        return
    with dst.open("a", encoding="utf-8") as handle:
        handle.write(rows[0] + "\n")


def done_keys(progress_path: Path) -> set[tuple[str, int]]:
    if not progress_path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with progress_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "done":
                keys.add((row["topo"], int(row["seed"])))
    return keys


def append_progress(progress_path: Path, row: dict[str, object]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    exists = progress_path.exists()
    fields = ["timestamp", "topo", "seed", "status", "seconds", "message"]
    with progress_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def backup_main_data(
    run_dir: Path,
    topos: Iterable[str],
    duration: float,
    sampler: str,
    entry: str,
    output_suffix: str | None = None,
    test_duration: float = 0.075,
    rate_manager: str = "MINSTREL",
    transport: str = "udp",
) -> None:
    backup_dir = run_dir / "main_data_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for topo in topos:
        for path in qnn_output_files(
            topo, duration, sampler, entry, output_suffix,
            test_duration, rate_manager, transport,
        ):
            dst = backup_dir / path.name
            if not dst.exists():
                shutil.copy2(path, dst)


def restore_main_data(
    run_dir: Path,
    topos: Iterable[str],
    duration: float,
    sampler: str,
    entry: str,
    output_suffix: str | None = None,
    test_duration: float = 0.075,
    rate_manager: str = "MINSTREL",
    transport: str = "udp",
) -> None:
    backup_dir = run_dir / "main_data_backup"
    for topo in topos:
        for path in qnn_output_files(
            topo, duration, sampler, entry, output_suffix,
            test_duration, rate_manager, transport,
        ):
            path.unlink(missing_ok=True)
    if backup_dir.exists():
        for path in backup_dir.glob("*.tsv"):
            shutil.copy2(path, NS3_DATA / path.name)


def clear_main_outputs(
    topo: str,
    duration: float,
    sampler: str,
    entry: str,
    output_suffix: str | None = None,
    test_duration: float = 0.075,
    rate_manager: str = "MINSTREL",
    transport: str = "udp",
) -> None:
    for path in qnn_output_files(
        topo, duration, sampler, entry, output_suffix,
        test_duration, rate_manager, transport,
    ):
        path.unlink(missing_ok=True)


def collect_single_seed_outputs(files: list[Path]) -> list[tuple[Path, str, str]]:
    collected: list[tuple[Path, str, str]] = []
    for src in files:
        header, rows = read_tsv_rows(src)
        if len(rows) != 1:
            raise RuntimeError(f"expected exactly one data row in {src}, got {len(rows)}")
        collected.append((src, header, rows[0]))
    return collected


def append_collected_seed_file(src: Path, header: str, row: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_text(header + "\n" + row + "\n", encoding="utf-8")
        return
    with dst.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")


def start_server(args: argparse.Namespace, run_dir: Path) -> subprocess.Popen:
    server_log = (run_dir / "qnn_server.log").open("ab")
    command = [
        sys.executable,
        str(REPO / "code" / "neural_bandit_bridge" / "nb_server.py"),
        "--surrogate", args.surrogate,
        "--policy", args.policy,
        "--matched-params", str(args.matched_params),
        "--ucb-beta", str(args.ucb_beta),
        "--gp-features", str(args.gp_features),
        "--gp-length-scale", str(args.gp_length_scale),
        "--gp-noise", str(args.gp_noise),
        "--hidden", str(args.hidden),
        "--feature-dim", str(args.feature_dim),
        "--qnn-layers", str(args.qnn_layers),
        "--qnn-n-qubits", str(args.qnn_n_qubits),
        "--qnn-max-z-order", str(args.qnn_max_z_order),
        "--qnn-architecture", args.qnn_architecture,
        "--qnn-unitary-fusion" if args.qnn_unitary_fusion else "--no-qnn-unitary-fusion",
        "--qnn-edge-specific" if args.qnn_edge_specific else "--no-qnn-edge-specific",
        "--qnn-input-tanh" if args.qnn_input_tanh else "--no-qnn-input-tanh",
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
        "--train-log", str(run_dir / "qnn_train_loss.tsv"),
        "--checkpoint-dir", str(run_dir / "checkpoints"),
        "--checkpoint-every", str(args.checkpoint_every),
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
        "--prior-lambda", "1.0",
        "--prior-a", "0.5",
        "--prior-b", "0.025",
        "--timing-method", args.surrogate,
        "--timing-topology", args.topos[0] if len(args.topos) == 1 else "multiple",
        "--port", str(args.port),
        "--torch-threads", str(args.torch_threads),
        "--verbose",
    ]
    if args.load_checkpoint is not None:
        command.extend(["--load-checkpoint", str(args.load_checkpoint)])
    if args.feature_diagnostics:
        command.extend([
            "--feature-diagnostics-log", str(run_dir / "feature_diagnostics.csv"),
        ])
    log(f"starting {args.surrogate.upper()} server")
    proc = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    wait_for_port("127.0.0.1", args.port)
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # A runner launched as a background shell job can inherit SIGINT as
        # ignored.  SIGTERM is therefore the bounded shutdown fallback.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)


def run_one_seed(args: argparse.Namespace, run_dir: Path, topo: str, seed: int) -> None:
    # ``seed`` is the logical replicate id used in filenames and progress
    # records.  A recovery run may supply a different ns-3 RNG seed when a
    # particular logical seed hits a simulator-level crash; keeping the two
    # explicit makes that exception auditable instead of silently changing the
    # experiment protocol.
    simulation_seed = args.physical_seed if args.physical_seed is not None else seed
    final_checkpoint_pattern = (
        f"{args.surrogate}_seed{seed + args.agent_seed_offset:06d}_final_*.pt"
    )
    if args.save_final_checkpoint:
        for stale in (run_dir / "checkpoints").glob(final_checkpoint_pattern):
            stale.unlink()
    log_dir = run_dir / "logs" / topo
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # The ns-3 implementation treats presence of these names as true.
    for flag in ("NSTEST_DYNAMIC", "NSTEST_MOBILITY"):
        if not scenario_flag_enabled(flag):
            env.pop(flag, None)
    env["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(NS3_DIR / "build" / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env.update({
        "NSTEST_NB_BRIDGE": "1",
        "NSTEST_NB_HOST": "127.0.0.1",
        "NSTEST_NB_PORT": str(args.port),
        "NSTEST_NB_CANDIDATES": str(args.candidates),
        "NSTEST_NB_NOVEL_CANDIDATES": str(args.novel_candidates),
        "NSTEST_NB_GUIDE_FALLBACK": str(args.guide_fallback),
        "NSTEST_NB_COLD_START": str(args.cold_start),
        "NSTEST_NSIMULATIONS": "1",
        "NSTEST_SEED": str(simulation_seed),
        "NSTEST_AGENT_SEED": str(seed + args.agent_seed_offset),
        "NSTEST_TOPO": topo,
        "NSTEST_OPTIMIZER": "NB",
        "NSTEST_SAMPLER": args.sampler,
        "NSTEST_ENTRY": args.entry,
        "NSTEST_HCM_PROFILE": args.hcm_profile,
        "NSTEST_HCM_GLOBAL_EPS": str(args.hcm_global_epsilon),
        "NSTEST_HCM_RESTART_PATIENCE": str(args.hcm_restart_patience),
        "NSTEST_HCM_RESTART_BURST": str(args.hcm_restart_burst),
        "NSTEST_HCM_RESTART_MAX_BURSTS": str(args.hcm_restart_max_bursts),
        "NSTEST_HCM_RESTART_MIN_DELTA": str(args.hcm_restart_min_delta),
        "NSTEST_HCM_RESTART_TARGET": str(args.hcm_restart_target),
        "NSTEST_HCM_BATCH_EXPLORE_FRACTION": str(args.hcm_batch_explore_fraction),
        "NSTEST_HCM_BATCH_MIN_RADIUS": str(args.hcm_batch_min_radius),
        "NSTEST_HCM_BATCH_MAX_RADIUS": str(args.hcm_batch_max_radius),
        "NSTEST_ADHOC_TIEBREAK_WEIGHT": str(args.adhoc_tiebreak_weight),
        "NSTEST_REWARD": "ADHOC",
        "NSTEST_DURATION": f"{args.duration:g}",
        "NSTEST_TEST_DURATION": f"{args.test_duration:g}",
        "NSTEST_RATE_MANAGER": args.rate_manager.lower(),
        "NSTEST_TRANSPORT": args.transport,
    })
    if args.output_suffix:
        env["NSTEST_OUTPUT_SUFFIX"] = args.output_suffix
    log_path = log_dir / f"seed_{seed:03d}.log"
    start = time.time()
    log(f"run topo={topo} seed={seed} physical_seed={simulation_seed}")
    clear_main_outputs(
        topo, args.duration, args.sampler, args.entry, args.output_suffix,
        args.test_duration, args.rate_manager, args.transport,
    )
    with log_path.open("wb") as handle:
        result = subprocess.run(
            [str(NS3_DIR / "build" / "scratch" / "nsTest" / "nsTest")],
            cwd=NS3_DIR,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    seconds = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"ns-3 failed for {topo} seed={seed}; see {log_path}")
    if args.save_final_checkpoint:
        deadline = time.monotonic() + 30.0
        checkpoint_matches: list[Path] = []
        while time.monotonic() < deadline:
            checkpoint_matches = list(
                (run_dir / "checkpoints").glob(final_checkpoint_pattern)
            )
            if len(checkpoint_matches) == 1 and checkpoint_matches[0].stat().st_size > 0:
                break
            time.sleep(0.05)
        if len(checkpoint_matches) != 1 or checkpoint_matches[0].stat().st_size <= 0:
            raise RuntimeError(
                f"final checkpoint was not saved for {topo} seed={seed}: "
                f"matches={checkpoint_matches}"
            )

    files = qnn_output_files(
        topo, args.duration, args.sampler, args.entry, args.output_suffix,
        args.test_duration, args.rate_manager, args.transport,
    )
    if len(files) != len(METRICS):
        raise RuntimeError(
            f"expected {len(METRICS)} unique output TSV files for {topo} seed={seed}, "
            f"found {len(files)}: {[path.name for path in files]}"
        )
    collected = collect_single_seed_outputs(files)
    seed_dir = run_dir / "per_seed" / topo / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for src, header, row in collected:
        shutil.copy2(src, seed_dir / src.name)
        append_collected_seed_file(
            src,
            header,
            row,
            run_dir / "clean_data" / clean_output_name(
                src.name, args.surrogate, args.output_suffix
            ),
        )

    append_progress(run_dir / "progress.csv", {
        "timestamp": now(),
        "topo": topo,
        "seed": seed,
        "status": "done",
        "seconds": f"{seconds:.1f}",
        "message": "" if simulation_seed == seed else f"physical_seed={simulation_seed}",
    })
    log(f"done topo={topo} seed={seed} seconds={seconds:.1f}")


def run_seed_with_retries(args: argparse.Namespace, run_dir: Path, topo: str, seed: int) -> None:
    attempt = 1
    while True:
        try:
            run_one_seed(args, run_dir, topo, seed)
            return
        except Exception as exc:  # noqa: BLE001
            max_attempts = max(0, args.max_seed_attempts)
            message = str(exc)
            if max_attempts and attempt >= max_attempts:
                raise
            append_progress(run_dir / "progress.csv", {
                "timestamp": now(),
                "topo": topo,
                "seed": seed,
                "status": "retry",
                "seconds": "",
                "message": f"attempt {attempt} failed: {message}",
            })
            log(f"retry topo={topo} seed={seed} attempt={attempt} reason={message}")
            clear_main_outputs(
                topo, args.duration, args.sampler, args.entry, args.output_suffix,
                args.test_duration, args.rate_manager, args.transport,
            )
            attempt += 1


def write_manifest(args: argparse.Namespace, run_dir: Path) -> None:
    lines = [
        f"run_id={run_dir.name}",
        f"mode={args.surrogate.upper()} Neural Bandit full sequential seeds",
        f"topos={','.join(args.topos)}",
        f"seeds={args.seed_start}..{args.seed_end}",
        f"physical_seed={'' if args.physical_seed is None else args.physical_seed}",
        f"agent_seed_offset={args.agent_seed_offset}",
        f"duration={args.duration}",
        f"test_duration={args.test_duration}",
        f"rate_manager={args.rate_manager}",
        f"transport={args.transport}",
        f"dynamic={int(scenario_flag_enabled('NSTEST_DYNAMIC'))}",
        f"mobility={int(scenario_flag_enabled('NSTEST_MOBILITY'))}",
        f"dynamic_profile={os.environ.get('NSTEST_DYNAMIC_PROFILE', '')}",
        f"candidates={args.candidates}",
        f"novel_candidates={args.novel_candidates}",
        f"acquisition_preset={args.acquisition_preset}",
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
        f"guide_fallback={args.guide_fallback}",
        f"surrogate={args.surrogate}",
        f"policy={args.policy}",
        f"backend={'rff-rbf-gp' if args.policy == 'gp_ts' else ('numpy-per-arm' if args.policy == 'per_arm_ts' else ('torch-statevector' if args.surrogate.startswith('qnn') else 'torch-mlp'))}",
        f"hidden={args.hidden}",
        f"matched_params={args.matched_params}",
        f"ucb_beta={args.ucb_beta}",
        f"gp_features={args.gp_features}",
        f"gp_length_scale={args.gp_length_scale}",
        f"gp_noise={args.gp_noise}",
        f"feature_dim={args.feature_dim}",
        f"qnn_layers={args.qnn_layers}",
        f"qnn_n_qubits={args.qnn_n_qubits}",
        f"qnn_max_z_order={args.qnn_max_z_order}",
        f"qnn_architecture={args.qnn_architecture}",
        f"qnn_unitary_fusion={args.qnn_unitary_fusion}",
        f"qnn_edge_specific={args.qnn_edge_specific}",
        f"qnn_input_tanh={args.qnn_input_tanh}",
        f"measurements=Z correlations up to order {args.qnn_max_z_order}",
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
        f"train_log={run_dir / 'qnn_train_loss.tsv'}",
        f"checkpoint_dir={run_dir / 'checkpoints'}",
        f"checkpoint_every={args.checkpoint_every}",
        f"save_final_checkpoint={args.save_final_checkpoint}",
        f"load_checkpoint={args.load_checkpoint or ''}",
        f"restore_checkpoint_state={args.restore_checkpoint_state}",
        f"freeze_loaded_model={args.freeze_loaded_model}",
        f"qnn_frozen_fusion={args.qnn_frozen_fusion}",
        f"qnn_frozen_torchscript={args.qnn_frozen_torchscript}",
        f"feature_cache_size={args.feature_cache_size}",
        f"acquisition_top_k={args.acquisition_top_k}",
        f"acquisition_temperature={args.acquisition_temperature}",
        f"acquisition_epsilon={args.acquisition_epsilon}",
        f"posterior_scale={args.posterior_scale}",
        "prior_lambda=1.0",
        "prior_a=0.5",
        "prior_b=0.025",
        f"feature_diagnostics={args.feature_diagnostics}",
        f"port={args.port}",
        f"output_suffix={args.output_suffix or ''}",
    ]
    (run_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparisons(args: argparse.Namespace, run_dir: Path) -> None:
    raise RuntimeError("Use scripts/analyze_main.py and scripts/reproduce_figures.py "
                       "for the main publication outputs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S_qnn_full_all3"))
    parser.add_argument("--topos", nargs="+", default=list(DEFAULT_TOPOS))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=25)
    parser.add_argument(
        "--agent-seed-offset",
        type=int,
        default=1_000_003,
        help="Deterministic offset separating model/HCM RNG from the ns-3 environment RNG.",
    )
    parser.add_argument(
        "--physical-seed",
        type=int,
        default=None,
        help=(
            "Override the ns-3 RNG seed while retaining the logical seed id; "
            "intended only for an explicitly recorded recovery replicate."
        ),
    )
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--test-duration", type=float, default=0.075)
    parser.add_argument("--port", type=int, default=9996)
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "Unique ns-3 output suffix. This isolates concurrent replicates; "
            "the suffix is removed again from clean_data filenames."
        ),
    )
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--novel-candidates", type=int, default=256)
    parser.add_argument(
        "--acquisition-preset",
        choices=("custom", "neural-ring-202606"),
        default="custom",
        help=(
            "Use 'neural-ring-202606' to reproduce the acquisition settings of "
            "the saved June 2026 Neural Bandit ring benchmark."
        ),
    )
    parser.add_argument("--guide-fallback", choices=(0, 1), type=int, default=1)
    parser.add_argument("--sampler", choices=("HGM", "HCM"), default="HGM")
    parser.add_argument("--entry", choices=("DEF", "DEGA"), default="DEF")
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
    parser.add_argument("--rate-manager", choices=("MINSTREL", "CONSTANT", "IDEAL"), default="MINSTREL")
    parser.add_argument("--transport", choices=("udp", "tcp"), default="udp")
    parser.add_argument("--hidden", type=int, choices=(64,), default=64)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--qnn-layers", type=int, choices=(3,), default=3)
    parser.add_argument("--qnn-n-qubits", type=int, choices=(5,), default=5)
    parser.add_argument("--qnn-max-z-order", type=int, choices=(5,), default=5)
    parser.add_argument(
        "--qnn-architecture",
        choices=("qcnn_shared_reupload",),
        default="qcnn_shared_reupload",
        help="Circuit layout used by the Torch state-vector QNN.",
    )
    parser.add_argument(
        "--qnn-unitary-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exact dense-unitary fusion; disable with --no-qnn-unitary-fusion.",
    )
    parser.add_argument(
        "--qnn-edge-specific",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use independent U_SU4 weights per ring edge (ablation; shared is the tuned default).",
    )
    parser.add_argument(
        "--qnn-input-tanh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply tanh before L2 amplitude normalization.",
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save a model checkpoint every N retrains. 0 disables checkpoints.",
    )
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save exactly one final model and Bayesian posterior checkpoint per seed.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Load a separately pretrained QNN/MLP checkpoint before each episode.",
    )
    parser.add_argument(
        "--restore-checkpoint-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore the full saved Bayesian policy state, not only model weights.",
    )
    parser.add_argument(
        "--freeze-loaded-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze a loaded checkpoint so the surrogate only performs forward inference.",
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
    parser.add_argument("--acquisition-top-k", type=int, default=1)
    parser.add_argument("--acquisition-temperature", type=float, default=0.0)
    parser.add_argument("--acquisition-epsilon", type=float, default=0.0)
    parser.add_argument("--posterior-scale", type=float, default=1.0)
    parser.add_argument(
        "--feature-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log per-round posterior condition number and feature norm.",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--write-comparisons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy comparison output is disabled; use the main analysis scripts.",
    )
    parser.add_argument(
        "--max-seed-attempts",
        type=int,
        default=0,
        help="0 means retry a failed/empty seed forever; otherwise stop after N attempts.",
    )
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
    return args


def main() -> int:
    args = parse_args()
    if args.qnn_architecture != "legacy":
        args.qnn_unitary_fusion = False
    if args.qnn_architecture == "qcnn_angle_reupload":
        args.qnn_input_tanh = False
    elif args.qnn_architecture == "qcnn_shared_reupload":
        args.qnn_input_tanh = True
    if not 0.0 <= args.acquisition_epsilon <= 1.0:
        raise SystemExit("--acquisition-epsilon must be in [0, 1]")
    if args.posterior_scale < 0.0:
        raise SystemExit("--posterior-scale must be >= 0")
    if args.ucb_beta < 0.0 or args.matched_params < 0:
        raise SystemExit("--ucb-beta and --matched-params must be non-negative")
    if args.gp_features < 1 or args.gp_length_scale <= 0 or args.gp_noise <= 0:
        raise SystemExit("GP feature count, length scale, and noise must be positive")
    if args.max_retrains < -1:
        raise SystemExit("--max-retrains must be -1, 0, or a positive integer")
    if args.freeze_after_step < -1:
        raise SystemExit("--freeze-after-step must be -1 or a non-negative integer")
    if args.update_epochs < 0:
        raise SystemExit("--update-epochs must be >= 0")
    if args.feature_cache_size < 0:
        raise SystemExit("--feature-cache-size must be >= 0")
    if args.load_checkpoint is not None:
        args.load_checkpoint = args.load_checkpoint.expanduser().resolve()
        if not args.load_checkpoint.is_file():
            raise SystemExit(f"--load-checkpoint does not exist: {args.load_checkpoint}")
    if args.output_suffix and not all(
        ch.isalnum() or ch in "-." for ch in args.output_suffix
    ):
        raise SystemExit("--output-suffix may contain only letters, digits, '-' and '.'")
    if (args.sampler, args.entry) not in {("HGM", "DEF"), ("HCM", "DEGA")}:
        raise SystemExit("QNN supports HGM+DEF or HCM+DEGA for a clean comparison")
    if not 0.0 <= args.hcm_global_epsilon <= 1.0:
        raise SystemExit("--hcm-global-epsilon must be in [0, 1]")
    if min(args.hcm_restart_patience, args.hcm_restart_burst, args.hcm_restart_max_bursts) < 1:
        raise SystemExit("HCM restart patience, burst, and max-bursts must be >= 1")
    if not 0.0 <= args.hcm_restart_min_delta <= 1.0:
        raise SystemExit("--hcm-restart-min-delta must be in [0, 1]")
    if not 0.0 <= args.hcm_restart_target <= 1.0:
        raise SystemExit("--hcm-restart-target must be in [0, 1]")
    if not 0.0 <= args.hcm_batch_explore_fraction <= 1.0:
        raise SystemExit("--hcm-batch-explore-fraction must be in [0, 1]")
    if not 1.0 <= args.hcm_batch_min_radius <= 20.0:
        raise SystemExit("--hcm-batch-min-radius must be in [1, 20]")
    if not args.hcm_batch_min_radius <= args.hcm_batch_max_radius <= 20.0:
        raise SystemExit("--hcm-batch-max-radius must be in [min-radius, 20]")
    if not 0.0 <= args.adhoc_tiebreak_weight <= 0.05:
        raise SystemExit("--adhoc-tiebreak-weight must be in [0, 0.05]")
    if args.physical_seed is not None and args.seed_start != args.seed_end:
        raise SystemExit("--physical-seed requires a single logical seed")
    run_dir = REPO / "work" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, run_dir)

    log(f"run_dir={run_dir}")
    backup_main_data(
        run_dir,
        args.topos,
        args.duration,
        args.sampler,
        args.entry,
        args.output_suffix,
        args.test_duration,
        args.rate_manager,
        args.transport,
    )
    server: subprocess.Popen | None = None
    try:
        server = start_server(args, run_dir)
        completed = done_keys(run_dir / "progress.csv")
        for topo in args.topos:
            for seed in range(args.seed_start, args.seed_end + 1):
                if (topo, seed) in completed:
                    log(f"skip completed topo={topo} seed={seed}")
                    continue
                run_seed_with_retries(args, run_dir, topo, seed)
        log("all QNN runs completed")
        if args.write_comparisons:
            write_comparisons(args, run_dir)
    except Exception as exc:  # noqa: BLE001
        append_progress(run_dir / "progress.csv", {
            "timestamp": now(),
            "topo": "",
            "seed": "",
            "status": "error",
            "seconds": "",
            "message": str(exc),
        })
        log(f"ERROR: {exc}")
        return 1
    finally:
        stop_server(server)
        restore_main_data(
            run_dir,
            args.topos,
            args.duration,
            args.sampler,
            args.entry,
            args.output_suffix,
            args.test_duration,
            args.rate_manager,
            args.transport,
        )
        log("server stopped and main data restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
