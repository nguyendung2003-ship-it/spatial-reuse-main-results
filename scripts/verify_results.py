#!/usr/bin/env python3
"""Verify included main measurements, tables, checkpoints and timing samples."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_main import DEFAULT_RESULTS, ROOT, load_results, paired_contrasts, summarize, write_csv

INFERENCE = ROOT / "results" / "inference"
STATS = ("mean_ms", "median_ms", "p95_ms", "minimum_ms", "maximum_ms")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def statistics(samples: list[float]) -> dict:
    data = np.asarray(samples)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Timing samples must be finite and nonnegative")
    return {"calls": len(data), "mean_ms": float(np.mean(data)),
            "median_ms": float(np.median(data)), "p95_ms": float(np.quantile(data, .95)),
            "minimum_ms": float(np.min(data)), "maximum_ms": float(np.max(data))}


def compare_numeric(expected: dict, actual: dict, fields, identifier: str) -> None:
    for field in fields:
        if not np.isclose(float(expected[field]), float(actual[field]), atol=1e-9, rtol=1e-12):
            raise ValueError(f"Numeric mismatch: {identifier}, {field}")


def verify_performance() -> dict:
    protocol, matrices = load_results()
    rows, per_seed = summarize(protocol, matrices)
    fields = ("tail_reward", "final_cumulative_loss_ref", "tail_fairness", "tail_throughput_Mbps")
    saved = {(r["topology"], r["method"]): r for r in read_csv(DEFAULT_RESULTS / "table_v.csv")}
    if len(saved) != 27:
        raise ValueError("Require 27 main Table V rows")
    for row in rows:
        compare_numeric(saved[row["topology"], row["method"]], row, fields, row["method"])
    contrasts = paired_contrasts(protocol, matrices)
    saved_ci = {(r["topology"], r["contrast"]): r for r in read_csv(DEFAULT_RESULTS / "paired_contrasts.csv")}
    if len(saved_ci) != 9:
        raise ValueError("Require 9 main paired comparisons")
    for row in contrasts:
        compare_numeric(saved_ci[row["topology"], row["contrast"]], row,
                        ("estimate", "ci95_low", "ci95_high"), row["contrast"])
    figure_pins = json.loads((DEFAULT_RESULTS / "figure_sha256.json").read_text())
    if len(figure_pins) != 24:
        raise ValueError("Require twelve PDF and twelve PNG main panels")
    for record in figure_pins:
        if digest(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Imported figure hash mismatch: {record['path']}")
    return {"input_matrices": len(matrices), "table_v_rows": len(rows),
            "per_seed_rows": len(per_seed), "paired_contrasts": len(contrasts),
            "imported_figure_files": len(figure_pins)}


def verify_inference() -> tuple[dict, list[dict], list[dict]]:
    protocol = json.loads((INFERENCE / "protocol.json").read_text())
    if len(protocol["cases"]) != 12 or protocol["physical_seeds"] != [1]:
        raise ValueError("Require 12 main trained seed 1 checkpoints")
    for case in protocol["cases"]:
        if digest(ROOT / case["checkpoint"]) != case["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {case['checkpoint']}")
        if not case["frozen_weights"] or case["feature_memoization"]:
            raise ValueError("Unexpected trained-feature timing scope")
    selected = (read_csv(INFERENCE / "selected_feature_latency.csv")
                + read_csv(INFERENCE / "selected_prediction_latency.csv"))
    grouped = defaultdict(list)
    repetitions = defaultdict(list)
    for row in read_csv(INFERENCE / "raw_timings.csv"):
        key = (row["topology"], row["method"], row["backend"], row["batch"], row["scope"])
        grouped[key].append(int(row["elapsed_ns"]) / 1e6)
        repetitions[key].append(int(row["repetition"]))
    if len(selected) != 48 or len(grouped) != 48:
        raise ValueError("Require 48 selected main inference groups")
    for row in selected:
        key = (row["topology"], row["method"], row["backend"], row["batch"], row["scope"])
        if sorted(repetitions[key]) != list(range(1000)):
            raise ValueError(f"Missing or duplicate timing repetitions: {key}")
        actual = statistics(grouped[key])
        compare_numeric(row, actual, ("calls",) + STATS, str(key))
    native_rows = read_csv(INFERENCE / "native_baseline_latency.csv")
    if len(native_rows) != 8:
        raise ValueError("Require four native controllers and two reporting windows")
    for report in native_rows:
        trace_path = ROOT / report["source_trace"]
        if digest(trace_path) != report["source_trace_sha256"]:
            raise ValueError("Native trace hash mismatch")
        method = report["native_method"]
        rows = [dict(row) for row in read_csv(trace_path, "\t")
                if row["operation"] == report["operation"]]
        for row in rows:
            row["measurement_cycle"] = int(row["env_step"]) - (method != "inspire")
        rows = [row for row in rows if 0 <= row["measurement_cycle"] <= 2000]
        if method == "gmts_ring":
            rows = [row for row in rows if int(row["env_step"]) % 3 == 0]
        elif method == "inspire":
            rows = [row for row in rows if int(row["n_obs"]) >= 31]
        if report["window"] == "mature_tail_cycles_1901_2000":
            rows = [row for row in rows if 1901 <= row["measurement_cycle"] <= 2000]
        actual = statistics([int(row["elapsed_ns"]) / 1e6 for row in rows])
        compare_numeric(report, actual, ("calls",) + STATS, method)
    return {"trained_checkpoints": 12, "selected_timing_groups": 48,
            "selected_raw_samples": sum(map(len, grouped.values())),
            "native_traces": 4, "native_reporting_windows": 8}, selected, native_rows


def export_inference(selected: list[dict], native_rows: list[dict]) -> None:
    feature = [r for r in selected if r["scope"] == "fresh_feature_forward"]
    current = [r for r in feature if r["topology"] == "C6o" and r["batch"] == "512"]
    order = ("Neural Bandit (GM)", "Neural Bandit (HCM)", "QNN Bandit (GM)", "QNN Bandit (HCM)")
    lookup = {r["method"]: r for r in current}
    best = min(float(r["mean_ms"]) for r in current)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{CPU Latency of Native Controllers and Optimized Frozen Feature Extractors "
             r"(C6, Seed 1; Bold Indicates the Fastest Feature Extractor).}",
             r"\label{tab:latency}", r"\footnotesize", r"\setlength{\tabcolsep}{2.8pt}",
             r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"& & \multicolumn{3}{c}{\textbf{Latency [ms]}} \\",
             r"\cmidrule(lr){3-5}",
             r"\textbf{Method} & \textbf{Implementation} & \textbf{Mean} & \textbf{Median} & \textbf{P95} \\",
             r"\midrule", r"\multicolumn{5}{@{}l@{}}{\textit{Native controller operations}} \\"]
    native = {r["method"]: r for r in native_rows if r["window"] == "all_active_calls"}
    for method in ("epsilon-greedy", "GM-TS", "HCM-TS", "INSPIRE"):
        row = native[method]
        display = r"$\epsilon$-greedy" if method == "epsilon-greedy" else method
        backend = "C++ (custom GP)" if method == "INSPIRE" else "C++"
        lines.append(display + " & " + backend + " & "
                     + " & ".join(f"{float(row[k]):.4f}" for k in ("mean_ms", "median_ms", "p95_ms")) + r" \\")
    lines += [r"\midrule",
              r"\multicolumn{5}{@{}l@{}}{\textit{Optimized frozen feature inference (batch 512)}} \\"]
    for method in order:
        row = lookup[method]
        values = [f"{float(row[k]):.4f}" for k in ("mean_ms", "median_ms", "p95_ms")]
        display = method
        if float(row["mean_ms"]) == best:
            display = rf"\textbf{{{display}}}"
            values = [rf"\textbf{{{v}}}" for v in values]
        backend = "NumPy" if method.startswith("Neural") else "C++ + NumPy"
        lines.append(display + " & " + backend + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
              r"\parbox{\columnwidth}{\scriptsize\emph{Note:} AMD Ryzen 9 9950X3D2, one CPU thread. "
              r"The blocks time different operations. Native $\epsilon$-greedy, GM-TS and HCM-TS "
              r"include proposals and action selection; HCM-TS uses $\epsilon_{\mathrm f}=0.1$ with "
              r"hold calls excluded. INSPIRE measures GP prescription and consensus over a full "
              r"32-observation window and excludes GP fitting. Feature inference computes new "
              r"features for all 512 profiles with frozen trained weights, $d=32$ and 1{,}000 timing "
              r"repetitions. Bayesian scoring, candidate generation, history scans, retraining "
              r"and IPC are excluded from this block.}", r"\end{table}"]
    (INFERENCE / "TableVII.tex").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "axes.labelsize": 20, "xtick.labelsize": 17,
                         "ytick.labelsize": 17, "legend.fontsize": 15, "pdf.fonttype": 42})
    topos = (("C6o", "C6"), ("MER_FLOORS_CH20_S5", "ENT-Clustered"),
             ("MER_FLOORS_BAD_DIM", "ENT-Dispersed"))
    colors = ("#2878B5", "#00A69C", "#C44E52", "#6C757D")
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, method in enumerate(order):
        rows = [next(r for r in feature if r["topology"] == t and r["method"] == method
                     and r["batch"] == "512") for t, _ in topos]
        x = np.arange(3) + (i-1.5)*.18
        ax.bar(x, [float(r["mean_ms"]) for r in rows], width=.18, label=method, color=colors[i])
    ax.set_xticks(np.arange(3), [d for _, d in topos])
    ax.set_ylabel("Feature inference latency (ms)")
    ax.grid(axis="y", alpha=.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.2))
    fig.tight_layout()
    fig.savefig(INFERENCE / "feature_inference.pdf", bbox_inches="tight")
    fig.savefig(INFERENCE / "feature_inference.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-inference", action="store_true",
                        help="Also export Table VII and the main feature-inference plot from verified samples")
    args = parser.parse_args()
    performance = verify_performance()
    inference, selected, native = verify_inference()
    if args.export_inference:
        export_inference(selected, native)
    report = {"status": "PASS", "performance": performance, "inference": inference,
              "simulations_run": 0}
    (ROOT / "results" / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
