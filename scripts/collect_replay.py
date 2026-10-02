#!/usr/bin/env python3
"""Collect a complete run_main replay into fresh, analyzable result files.

Require all nine controllers, three topologies, 25 ordered physical seeds and
2,001 measurement cycles. This command reads existing outputs and never runs
ns-3 or replaces the bundled publication results.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from analyze_main import summarize, write_csv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/main_experiment.json"
TEMPLATE = ROOT / "results/performance/protocol.json"
RAW_METRICS = {"reward": "rew", "fairness": "fair", "throughput": "cum"}
SEEDS = list(range(1, 26))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def argument(command: list[str], option: str) -> str:
    if command.count(option) != 1:
        raise ValueError(f"Launch command must specify {option} exactly once")
    index = command.index(option)
    if index + 1 >= len(command):
        raise ValueError(f"Missing argument for {option}")
    return command[index + 1]


def load_launch_plan(run: Path, config: dict, protocol: dict) -> dict:
    plan = json.loads((run / "launch_plan.json").read_text())
    if plan.get("config") != config:
        raise ValueError("Replay launch config differs from configs/main_experiment.json")
    if config["simulation"]["physical_seeds"] != SEEDS or config["simulation"]["duration_seconds"] != 150:
        raise ValueError("The main config must specify seeds 1..25 and a 150-second horizon")
    if config["simulation"]["measurement_cycles"] != 2001:
        raise ValueError("The main config must specify 2,001 measurement cycles")
    expected = {(method["id"], topology["id"]) for method in protocol["methods"]
                for topology in protocol["topologies"]}
    if set(config["methods"]) != {method["id"] for method in protocol["methods"]}:
        raise ValueError("Config and protocol have different method sets")
    if set(config["topologies"]) != {topology["id"] for topology in protocol["topologies"]}:
        raise ValueError("Config and protocol have different topology sets")
    cells = plan.get("cells", [])
    keys = [(cell["method"], cell["topology"]) for cell in cells]
    if len(keys) != 27 or len(set(keys)) != 27 or set(keys) != expected:
        raise ValueError("Require exactly 27 unique method/topology launch cells")
    for cell in cells:
        command = cell["command"]
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("Launch command must be a list of strings")
        kind = config["methods"][cell["method"]]["kind"]
        if float(argument(command, "--duration")) != 150.0:
            raise ValueError("Smoke or incomplete runs cannot be collected as final results")
        if float(argument(command, "--test-duration")) != 0.075:
            raise ValueError("Require 75-ms measurement cycles")
        if int(argument(command, "--seed-start")) != 1:
            raise ValueError("Require first seed 1")
        if kind == "native":
            if int(argument(command, "--replicates")) != 25:
                raise ValueError("Require 25 native replicates")
            method = config["methods"][cell["method"]]
            if argument(command, "--methods") != method["runner_label"]:
                raise ValueError("Native runner label differs from the method config")
            if argument(command, "--topology") != cell["topology"]:
                raise ValueError("Native launch topology differs from its cell")
        else:
            if int(argument(command, "--seed-end")) != 25:
                raise ValueError("Require final learned seed 25")
            if argument(command, "--topo") != cell["topology"]:
                raise ValueError("Learned launch topology differs from its cell")
        if argument(command, "--rate-manager") != "MINSTREL":
            raise ValueError("Require the main Minstrel-HT rate manager")
    return plan


def parse_manifest(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def verify_seed_order(cell_dir: Path, topology: str, method: dict) -> dict:
    if method["kind"] == "native":
        path = cell_dir / "runner.log"
        text = path.read_text()
        label = re.escape(method["runner_label"])
        pattern = rf"\breplicate method={label} seed=(\d+) rc=0 files=9\b"
        ordered = list(map(int, re.findall(pattern, text)))
        if ordered != SEEDS:
            raise ValueError(f"Native successful replicate order must be 1..25: {path}")
        if f"DONE method={method['runner_label']} reward_rows=25" not in text:
            raise ValueError(f"Native run has no 25-row completion record: {path}")
        if "RUN COMPLETE" not in text:
            raise ValueError(f"Native runner did not complete: {path}")
        return {"mechanism": "ordered successful replicate records in runner.log",
                "evidence": [{"path": portable_path(path), "sha256": digest(path)}],
                "ordered_physical_seeds": ordered}

    manifest_path = cell_dir / "manifest.txt"
    manifest = parse_manifest(manifest_path)
    requirements = {"topos": topology, "seeds": "1..25", "duration": 150.0,
                    "test_duration": 0.075, "surrogate": method["surrogate"],
                    "sampler": method["sampler"], "entry": method["entry"],
                    "candidates": method["candidates"],
                    "novel_candidates": method["novel_candidates"],
                    "acquisition_epsilon": method["acquisition_epsilon"],
                    "feature_dim": 32, "hidden": 64}
    for key, expected in requirements.items():
        actual = manifest.get(key)
        matches = actual == expected if isinstance(expected, str) else (
            actual is not None and float(actual) == float(expected))
        if not matches:
            raise ValueError(f"Learned manifest {key} differs from main config: {manifest_path}")
    progress = cell_dir / "progress.csv"
    with progress.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 25 or any(row["status"] != "done" or row["topo"] != topology for row in rows):
        raise ValueError(f"Require exactly 25 successful learned progress records: {progress}")
    ordered = [int(row["seed"]) for row in rows]
    if ordered != SEEDS:
        raise ValueError(f"Learned progress seed order must be 1..25: {progress}")
    return {"mechanism": "aggregate manifest and ordered merge progress.csv",
            "evidence": [{"path": portable_path(path), "sha256": digest(path)}
                         for path in (manifest_path, progress)],
            "ordered_physical_seeds": ordered}


def load_matrix(path: Path, native: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    header = np.loadtxt(path, delimiter="\t", max_rows=1, ndmin=1)
    if header.size not in (2001, 2002) or not np.array_equal(header[:2001], np.arange(2001)):
        raise ValueError(f"Invalid measurement-cycle header: {path}")
    matrix = np.loadtxt(path, delimiter="\t", skiprows=1, ndmin=2)
    allowed = {(25, 2001), (25, 2002)} if native else {(25, 2001)}
    if matrix.shape not in allowed or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid matrix shape or nonfinite entries: {path}; got {matrix.shape}")
    # Native final callbacks can emit one additional terminal value. Compare
    # exactly cycles 0..2000, matching the archived analysis horizon.
    return matrix[:, :2001], matrix.shape


def trim_terminal_column(source: Path, destination: Path) -> None:
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    if len(lines) != 26:
        raise ValueError(f"Expected a header and exactly 25 seed rows: {source}")
    trimmed = []
    for index, line in enumerate(lines):
        cells = line.split("\t")
        if index and len(cells) != 2002:
            raise ValueError(f"Inconsistent terminal-column layout: {source}")
        trimmed.append("\t".join(cells[:2001]))
    destination.write_text("\n".join(trimmed) + "\n")


def collect(run: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output already exists; choose a fresh --output-dir: {output}")
    config = json.loads(CONFIG.read_text())
    protocol = json.loads(TEMPLATE.read_text())
    plan = load_launch_plan(run, config, protocol)
    matrices, inputs, seed_checks = {}, [], []
    for topology in protocol["topologies"]:
        t = topology["id"]
        for description in protocol["methods"]:
            m = description["id"]
            method = config["methods"][m]
            cell_dir = run / f"{m}_{t}"
            order = verify_seed_order(cell_dir, t, method)
            seed_checks.append({"topology": t, "method_id": m, **order})
            data_dir = (cell_dir / "data" / t / method["runner_label"] if method["kind"] == "native"
                        else cell_dir / "clean_data")
            for metric, raw_suffix in RAW_METRICS.items():
                files = sorted(data_dir.glob(f"*_{raw_suffix}.tsv"))
                if len(files) != 1:
                    raise ValueError(f"Require exactly one {raw_suffix} input in {data_dir}; found {len(files)}")
                source = files[0]
                matrices[t, m, metric], raw_shape = load_matrix(source, method["kind"] == "native")
                relative = Path("data") / t / f"{m}_{metric}.tsv"
                inputs.append({"topology": t, "method": description["display"],
                               "method_id": m, "metric": metric,
                               "path": portable_path(output / relative),
                               "original_filename": source.name, "sha256": digest(source),
                               "raw_source_sha256": digest(source), "raw_shape": list(raw_shape),
                               "terminal_columns_trimmed": int(raw_shape[1] == 2002),
                               "replay_source": portable_path(source), "relative": str(relative)})
    if len(inputs) != 81:
        raise ValueError("Require exactly 81 validated main replay matrices")
    summaries, _ = summarize(protocol, matrices)
    provenance = {"collected_at_utc": datetime.now(timezone.utc).isoformat(),
                  "run_directory": portable_path(run), "launch_plan_sha256": digest(run / "launch_plan.json"),
                  "main_config_sha256": digest(CONFIG), "protocol_template_sha256": digest(TEMPLATE),
                  "collector_sha256": digest(Path(__file__)),
                  "input_matrices": 81, "method_topology_cells": 27,
                  "physical_seeds": SEEDS, "measurement_cycles": 2001,
                  "simulation_run_by_collector": False,
                  "reference_policy": "retain bundled frozen empirical references; do not recompute from replay",
                  "seed_order_checks": seed_checks,
                  "terminal_column_policy": "Native 2,002-column outputs retain first 2,001 columns; learned outputs must have exactly 2,001",
                  "matrices_with_native_terminal_column_trimmed": sum(row["terminal_columns_trimmed"] for row in inputs)}
    protocol["replay_provenance"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".collect-replay-", dir=output.parent))
    try:
        for row in inputs:
            destination = stage / row["relative"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = ROOT / row["replay_source"]
            if digest(source) != row["raw_source_sha256"]:
                raise ValueError(f"Input changed while collecting: {source}")
            if row["terminal_columns_trimmed"]:
                trim_terminal_column(source, destination)
                actual, _ = load_matrix(destination)
                if not np.array_equal(actual, matrices[row["topology"], row["method_id"], row["metric"]]):
                    raise ValueError(f"Terminal-column trimming altered retained values: {source}")
                row["sha256"] = digest(destination)
            else:
                shutil.copy2(source, destination)
                if digest(destination) != row["sha256"]:
                    raise ValueError(f"Input changed while collecting: {source}")
        protocol["dataset_files"] = [{key: value for key, value in row.items() if key != "relative"}
                                     for row in inputs]
        (stage / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        (stage / "collection_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (stage / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        write_csv(stage / "table_v_source.csv", summaries)
        check = {"status": "PASS", "input_matrices": 81, "table_v_rows": 27,
                 "seed_order_checks": 27, "measurement_cycles": 2001,
                 "physical_seed_rows": 25,
                 "source_bytes_preserved_for_2001_column_inputs": True,
                 "matrices_with_native_terminal_column_trimmed": provenance["matrices_with_native_terminal_column_trimmed"],
                 "simulation_run_by_collector": False}
        (stage / "collection_verification.json").write_text(json.dumps(check, indent=2) + "\n")
        if output.exists():
            raise ValueError("Output appeared during collection; refusing to replace it")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="main_replay")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/replay_results")
    args = parser.parse_args()
    if Path(args.run_id).is_absolute() or ".." in Path(args.run_id).parts:
        parser.error("run-id must be a relative name within work/runs")
    run = ROOT / "work/runs" / args.run_id
    try:
        check = collect(run, args.output_dir.expanduser().resolve())
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Collection failed: {error}\n")
    print(json.dumps({**check, "output_directory": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
