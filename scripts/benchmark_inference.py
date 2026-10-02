#!/usr/bin/env python3
"""Measure frozen inference for the four main trained controllers, without ns-3."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

# Establish a shared single-thread CPU environment before loading NumPy/Torch.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "code" / "neural_bandit_bridge"
sys.path.insert(0, str(CORE))

import numpy as np
import torch
from torch import nn

from nb_server import BanditConfig, NeuralLinearBandit, QNNSurrogate, NormalizedSurrogateMLP
from frozen_inference import FrozenHybridNativeQNN, FrozenNativeQNN

SCOPES = ("fresh_feature_forward", "fresh_forward_posterior_prediction")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_digest(model) -> str:
    hasher = hashlib.sha256()
    for name, value in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feasible_pairs() -> np.ndarray:
    pairs = [(s, p) for s in range(21) for p in range(21)
             if -82+s <= max(-82, min(-62, -82+(20-(1+p))))]
    if len(pairs) != 211:
        raise ValueError("Feasible local action grid changed")
    return np.asarray(pairs, dtype=np.float32)


def input_bank(n_ap: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lattice = feasible_pairs()
    indices = rng.integers(0, len(lattice), size=(count, n_ap))
    return np.ascontiguousarray(lattice[indices].reshape(count, 2*n_ap) / np.float32(20))


def boundary_bank(n_ap: int) -> np.ndarray:
    lattice = feasible_pairs()
    indices = (np.arange(211)[:, None] + np.arange(n_ap)[None, :]*37) % 211
    return np.ascontiguousarray(lattice[indices].reshape(211, 2*n_ap) / np.float32(20))


class FrozenNumpyMLP:
    backend = "numpy_frozen_mlp"

    def __init__(self, model):
        if not isinstance(model, NormalizedSurrogateMLP):
            raise ValueError("Require the normalized full-width main MLP")
        affine = [m for m in model.feature_net if isinstance(m, nn.Linear)]
        if len(affine) != 3:
            raise ValueError("Require three affine layers")
        self.weights = [np.ascontiguousarray(m.weight.detach().numpy().T) for m in affine]
        self.biases = [np.ascontiguousarray(m.bias.detach().numpy()) for m in affine]

    def features_np(self, xs: np.ndarray) -> np.ndarray:
        values = np.maximum(xs @ self.weights[0] + self.biases[0], np.float32(0))
        values = np.maximum(values @ self.weights[1] + self.biases[1], np.float32(0))
        values = np.tanh(values @ self.weights[2] + self.biases[2])
        norm = np.maximum(np.sqrt(np.sum(values*values, axis=1, keepdims=True)), np.float32(1e-6))
        return (values / norm * np.float32(values.shape[1]**.5)).astype(np.float64)


class Case:
    def __init__(self, record: dict, batches: list[int]):
        self.topology, self.method = record["topology"], record["method"]
        self.path = ROOT / record["checkpoint"]
        if digest(self.path) != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {self.path}")
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        allowed = {f.name for f in fields(BanditConfig)}
        configuration = {k: v for k, v in payload["cfg"].items() if k in allowed}
        configuration.update(load_checkpoint=str(self.path), restore_checkpoint_state=True,
                             freeze_loaded_model=False, device="cpu", feature_cache_size=0,
                             checkpoint_dir="", checkpoint_every=0, save_final_checkpoint=False,
                             timing_log="", train_log="", feature_diagnostics_log="")
        self.bandit = NeuralLinearBandit(BanditConfig(**configuration))
        self.bandit._ensure_model(record["input_dim"])
        self.model = self.bandit.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if payload["checkpoint_kind"] != "final" or self.bandit.train_calls < 1:
            raise ValueError("Require a trained final checkpoint")
        if not np.array_equal(self.bandit.precision, payload["precision"]):
            raise ValueError("Posterior was not restored exactly")
        if self.bandit.cfg.acquisition_epsilon != (.1 if "HCM" in self.method else 0.):
            raise ValueError("Checkpoint has unexpected acquisition epsilon")
        self.posterior_mean = self.bandit._posterior_state()[0].copy()
        self.initial_digest = model_digest(self.model)
        self.initial_counts = self.bandit.n_obs, self.bandit.train_calls
        n_ap = record["input_dim"] // 2
        validation_inputs = [input_bank(n_ap, 1, 914001+n_ap), input_bank(n_ap, 7, 914007+n_ap),
                             boundary_bank(n_ap), input_bank(n_ap, 512, 914512+n_ap)]
        if isinstance(self.model, QNNSurrogate):
            self.model.qlayer.clear_frozen_inference()
        with torch.inference_mode():
            references = [self.model.features(torch.from_numpy(xs)).numpy().copy()
                          for xs in validation_inputs]
        self.runners = {}
        if isinstance(self.model, QNNSurrogate):
            for batch in batches:
                self.runners[batch] = FrozenNativeQNN(self.model) if batch == 1 else FrozenHybridNativeQNN(self.model)
        else:
            runner = FrozenNumpyMLP(self.model)
            self.runners = {batch: runner for batch in batches}
        validations = []
        for batch, runner in self.runners.items():
            for xs, reference in zip(validation_inputs, references):
                actual = np.asarray(runner.features_np(xs), dtype=np.float64)
                if actual.shape != reference.shape or not np.allclose(actual, reference, atol=3e-5, rtol=1e-4):
                    raise ValueError(f"Frozen feature mismatch: {self.topology}, {self.method}")
                expected_scores = reference.astype(np.float64) @ self.posterior_mean
                actual_scores = actual @ self.posterior_mean
                if not np.allclose(actual_scores, expected_scores, atol=3e-5, rtol=1e-4):
                    raise ValueError("Frozen posterior prediction mismatch")
                validations.append({"selected_timing_batch": batch, "validation_batch": len(xs),
                                    "backend": runner.backend,
                                    "max_abs_feature_error": float(np.max(np.abs(actual-reference))),
                                    "max_abs_prediction_error": float(np.max(np.abs(actual_scores-expected_scores))),
                                    "posterior_mean_argmax_matches": bool(np.argmax(actual_scores)==np.argmax(expected_scores))})
        self.inputs = {batch: [input_bank(n_ap, batch, 2026100200+n_ap*10000+batch*10+j)
                              for j in range(16)] for batch in batches}
        self.info = {"topology": self.topology, "method": self.method,
                     "checkpoint": str(self.path.relative_to(ROOT)),
                     "checkpoint_sha256": record["checkpoint_sha256"],
                     "model_state_sha256": self.initial_digest, "seed": 1,
                     "input_dim": record["input_dim"], "feature_dim": 32,
                     "frozen_weights": True, "feature_memoization": False,
                     "observations": self.bandit.n_obs, "train_calls": self.bandit.train_calls,
                     "validation": validations,
                     "selected_backend_by_batch": {str(b): r.backend for b, r in self.runners.items()},
                     "timing_input_sha256": {str(b): [hashlib.sha256(xs.tobytes()).hexdigest() for xs in banks]
                                             for b, banks in self.inputs.items()}}

    def check_unchanged(self):
        if model_digest(self.model) != self.initial_digest or (self.bandit.n_obs, self.bandit.train_calls) != self.initial_counts:
            raise ValueError("Inference changed the model or posterior history")


def metadata() -> dict:
    cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    return {"utc": datetime.now(timezone.utc).isoformat(), "cpu_model": cpu,
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
            "thread_environment": {v: os.environ[v] for v in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work" / "inference")
    parser.add_argument("--cpu", type=int, help="Pin to this allowed logical CPU; default first allowed CPU")
    parser.add_argument("--batches", type=int, nargs="+", choices=(1, 512), default=[1, 512])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--topologies", nargs="+", choices=("C6o", "MER_FLOORS_CH20_S5", "MER_FLOORS_BAD_DIM"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"Choose a new output directory: {args.output_dir}")
    if args.repetitions < 2 or args.repetitions % 2 or args.warmup < 0:
        raise ValueError("Use a positive even repetition count and nonnegative warmup")
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    imported = json.loads((ROOT / "results" / "inference" / "protocol.json").read_text())
    records = [r for r in imported["cases"] if args.topologies is None or r["topology"] in args.topologies]
    cases = []
    for record in records:
        print(f"Prepare {record['topology']}: {record['method']}", flush=True)
        cases.append(Case(record, args.batches))
    args.output_dir.mkdir(parents=True)
    sources = [Path(__file__), *(CORE / name for name in ("nb_server.py", "tensor_quantum.py", "frozen_inference.py"))]
    protocol = {"status": "running", "scope": "optimized frozen steady-state fresh-input CPU inference",
                "physical_seeds": [1], "batch_sizes": args.batches, "feature_dimension": 32,
                "warmup_per_timing_block": args.warmup, "blocks_per_group": 2,
                "calls_per_block": args.repetitions//2,
                "repetitions_per_case_backend_batch_scope": args.repetitions,
                "timing_order": "deterministically shuffled homogeneous groups; second pass reverses group order",
                "feature_scope": "normalized contiguous float32 input to new float64 features; no feature memoization",
                "prediction_scope": "same fresh features plus dot product with restored posterior mean",
                "excluded": ["loading", "freeze/fusion/compilation", "raw-action normalization", "candidate generation",
                             "epsilon exploration", "retraining", "posterior update/sampling", "ns-3", "IPC"],
                "before": metadata(), "cases": [case.info for case in cases],
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources}}
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    items = [(case, batch, runner, scope) for case in cases
             for batch, runner in case.runners.items() for scope in SCOPES]
    order = np.random.default_rng(20261002).permutation(len(items)).tolist()
    samples = []
    with torch.inference_mode():
        for block in range(2):
            for index in (order if block == 0 else list(reversed(order))):
                case, batch, runner, scope = items[index]
                banks = case.inputs[batch]
                for warmup in range(args.warmup):
                    values = np.asarray(runner.features_np(banks[warmup % 16]), dtype=np.float64)
                    if scope == SCOPES[1]:
                        values = values @ case.posterior_mean
                for local in range(args.repetitions//2):
                    repetition = block*(args.repetitions//2) + local
                    xs = banks[repetition % 16]
                    start = time.perf_counter_ns()
                    values = np.asarray(runner.features_np(xs), dtype=np.float64)
                    if scope == SCOPES[1]:
                        values = values @ case.posterior_mean
                    elapsed = time.perf_counter_ns()-start
                    samples.append({"topology": case.topology, "method": case.method,
                                    "backend": runner.backend, "batch": batch, "scope": scope,
                                    "block": block, "repetition": repetition, "elapsed_ns": elapsed})
    grouped = defaultdict(list)
    for row in samples:
        grouped[tuple(row[k] for k in ("topology", "method", "backend", "batch", "scope"))].append(row["elapsed_ns"]/1e6)
    summary = []
    for (topology, method, backend, batch, scope), values in grouped.items():
        summary.append({"topology": topology, "method": method, "backend": backend,
                        "batch": batch, "scope": scope, "calls": len(values),
                        "mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
                        "p95_ms": float(np.percentile(values, 95)),
                        "minimum_ms": float(np.min(values)), "maximum_ms": float(np.max(values))})
    for case in cases:
        case.check_unchanged()
        if digest(case.path) != case.info["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during inference")
    for relative, pinned in protocol["source_sha256"].items():
        if digest(ROOT / relative) != pinned:
            raise ValueError("Measured source changed during inference")
    write_csv(args.output_dir / "raw_timings.csv", samples)
    write_csv(args.output_dir / "inference_summary.csv", summary)
    protocol.update(status="complete", raw_samples=len(samples), summary_groups=len(summary),
                    after=metadata(), checkpoint_and_model_state_unchanged=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Complete: {args.output_dir}")


if __name__ == "__main__":
    main()
