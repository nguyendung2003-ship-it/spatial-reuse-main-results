#!/usr/bin/env python3
"""Recompute the main paper tables from the included 25-seed measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "performance"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_results(results: Path = DEFAULT_RESULTS, verify_hashes: bool = True):
    protocol = json.loads((results / "protocol.json").read_text())
    if protocol["seeds"] != list(range(1, 26)) or protocol["cycles"] != list(range(2001)):
        raise ValueError("Require 25 seeds in order and measurement cycles 0..2000")
    matrices = {}
    for record in protocol["dataset_files"]:
        # The protocol pins repository-relative paths; resolve the data filename
        # under --results-dir so the package remains relocatable.
        path = results / "data" / record["topology"] / Path(record["path"]).name
        if verify_hashes and digest(path) != record["sha256"]:
            raise ValueError(f"Input SHA256 mismatch: {path}")
        header = np.loadtxt(path, delimiter="\t", max_rows=1, ndmin=1)
        # Several native exporters write one additional header label for the
        # terminal callback, but contain exactly 2,001 measured values per row.
        if header.size not in (2001, 2002) or not np.array_equal(header[:2001], np.arange(2001)):
            raise ValueError(f"Unexpected measurement-cycle header: {path}")
        data = np.loadtxt(path, delimiter="\t", skiprows=1, ndmin=2)
        if data.shape != (25, 2001) or not np.isfinite(data).all():
            raise ValueError(f"Invalid matrix shape or nonfinite entries: {path}")
        key = (record["topology"], record["method_id"], record["metric"])
        if key in matrices:
            raise ValueError(f"Duplicate dataset: {key}")
        matrices[key] = data
    expected = {(t["id"], m["id"], metric) for t in protocol["topologies"]
                for m in protocol["methods"] for metric in ("reward", "fairness", "throughput")}
    if set(matrices) != expected or len(matrices) != 81:
        raise ValueError("Require exactly 81 main result matrices")
    return protocol, matrices


def summarize(protocol: dict, matrices: dict):
    summaries, per_seed = [], []
    for topology in protocol["topologies"]:
        t = topology["id"]
        reference = topology["frozen_reference"]
        for method in protocol["methods"]:
            m = method["id"]
            rewards = matrices[t, m, "reward"]
            values = {
                "tail_reward": rewards[:, 1901:2001].mean(axis=1),
                "final_cumulative_loss_ref": np.cumsum(reference - rewards, axis=1)[:, -1],
                "tail_fairness": matrices[t, m, "fairness"][:, 1901:2001].mean(axis=1),
                "tail_throughput_Mbps": matrices[t, m, "throughput"][:, 1901:2001].mean(axis=1) * 1e-6,
            }
            summary = {"topology": t, "display_topology": topology["display"],
                       "method": method["display"], "replicates": 25,
                       "measurement_cycles": 2001, "reference_reward": reference,
                       "tail_cycles": "1901--2000"}
            summary.update({name: float(np.median(data)) for name, data in values.items()})
            summaries.append(summary)
            for index, seed in enumerate(protocol["seeds"]):
                per_seed.append({"topology": t, "method": method["display"], "seed": seed,
                                 **{name: float(data[index]) for name, data in values.items()}})
    return summaries, per_seed


def paired_contrasts(protocol: dict, matrices: dict) -> list[dict]:
    comparisons = (("qnn_gm", "gm_ts"), ("neural_gm", "gm_ts"), ("qnn_gm", "neural_gm"))
    names = {m["id"]: m["display"] for m in protocol["methods"]}
    settings = protocol["paired_bootstrap"]
    rows = []
    for topology in protocol["topologies"]:
        t = topology["id"]
        rng = np.random.default_rng(settings["random_seed"])
        indices = rng.integers(0, 25, size=(settings["resamples"], 25))
        for left, right in comparisons:
            a = matrices[t, left, "reward"][:, 1901:2001].mean(axis=1)
            b = matrices[t, right, "reward"][:, 1901:2001].mean(axis=1)
            samples = np.median(a[indices], axis=1) - np.median(b[indices], axis=1)
            low, high = np.quantile(samples, (.025, .975))
            rows.append({"topology": t, "display_topology": topology["display"],
                         "contrast": f"{names[left]} - {names[right]}",
                         "estimate": float(np.median(a) - np.median(b)),
                         "ci95_low": float(low), "ci95_high": float(high),
                         "paired_seeds": 25, "bootstrap_resamples": settings["resamples"],
                         "bootstrap_seed": settings["random_seed"],
                         "multiple_comparison_adjustment": "none"})
    return rows


def tex_method(name: str) -> str:
    if name == r"\epsilon-greedy":
        return r"$\epsilon$-greedy"
    citation = {"GM-TS": "bardou2021improving", "HCM-TS": "bardou2023mitigating",
                "INSPIRE": "bardou2022inspire"}.get(name)
    return name + (rf" \cite{{{citation}}}" if citation else "")


def write_tables(output: Path, protocol: dict, summaries: list[dict], contrasts: list[dict]) -> None:
    lookup = {(row["topology"], row["method"]): row for row in summaries}
    columns = (("tail_reward", 3, max), ("final_cumulative_loss_ref", 1, min),
               ("tail_fairness", 3, max), ("tail_throughput_Mbps", 1, max))
    best = {(t["id"], field): operation(lookup[t["id"], m["display"]][field]
                                      for m in protocol["methods"])
            for t in protocol["topologies"] for field, _, operation in columns}
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Tail Reward $\bar r$ (Cycles 1901--2000), Final Cumulative Performance Loss "
             r"$\hat L_{\mathrm{ref}}$ Relative to the Frozen Empirical Reference (Cycle 2000), "
             r"Jain's Fairness Index $J$, and Aggregate Throughput $S_{\mathrm{agg}}$ (Mbps). "
             r"Medians over 25 Seeds; Bold Indicates the Best Unrounded Value in Each Column.}",
             r"\label{tab:final}", r"\small", r"\setlength{\tabcolsep}{4.2pt}",
             r"\begin{tabular}{l|cccc|cccc|cccc}", r"\hline"]
    groups = [rf"\multicolumn{{4}}{{c{'|' if index < 2 else ''}}}{{{t['display']} "
              rf"($\hat\mu_{{\mathrm{{ref}}}}={t['frozen_reference']:.3f}$)}}"
              for index, t in enumerate(protocol["topologies"])]
    lines += ["& " + " & ".join(groups) + r" \\",
              "Method & " + " & ".join([r"$\bar r$ & $\hat L_{\mathrm{ref}}$ & $J$ & $S_{\mathrm{agg}}$"] * 3) + r" \\",
              r"\hline"]
    for index, method in enumerate(protocol["methods"]):
        if index == 5:
            lines.append(r"\hline")
        cells = []
        for t in protocol["topologies"]:
            row = lookup[t["id"], method["display"]]
            for field, decimals, _ in columns:
                value = row[field]
                formatted = f"{value:.{decimals}f}"
                cells.append(rf"\textbf{{{formatted}}}" if value == best[t["id"], field] else formatted)
        lines.append(tex_method(method["display"]) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", "",
              r"\begin{table*}[!t]", r"\centering",
              r"\caption{Paired Tail-Reward Contrasts Under the GM Sampler. Differences of Cross-Seed "
              r"Medians with Individual 95\% Percentile Confidence Intervals from 10{,}000 Paired "
              r"Bootstrap Resamples.}", r"\label{tab:contrasts}", r"\small",
              r"\begin{tabular}{lccc}", r"\hline",
              r"Contrast & C6 & ENT-Clustered & ENT-Dispersed \\", r"\hline"]
    labels = []
    for row in contrasts:
        if row["contrast"] not in labels:
            labels.append(row["contrast"])
    for label in labels:
        cells = []
        for t in protocol["topologies"]:
            row = next(r for r in contrasts if r["topology"] == t["id"] and r["contrast"] == label)
            cells.append(f"${row['estimate']:+.3f}$ [${row['ci95_low']:.3f}$, ${row['ci95_high']:.3f}$]")
        lines.append(label.replace(" - ", " $-$ ") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    (output / "tables_v_vi.tex").write_text("\n".join(lines) + "\n")


def write_parameter_counts(output: Path) -> None:
    rows = []
    for n_ap in (6, 10):
        h, d, q, layers = 64, 32, 5, 3
        mlp = 2*n_ap*h+h + h*h+h + h*d+d
        qnn = (1 << q)*(2*n_ap+1) + 15*layers + (1 << q)*d
        rows.extend([
            {"extractor": "Full-width MLP", "access_points": n_ap, "latent_dimension": d,
             "trainable_feature_parameters": mlp, "quantum_parameters": 0},
            {"extractor": "Hybrid classical-quantum", "access_points": n_ap, "latent_dimension": d,
             "trainable_feature_parameters": qnn, "quantum_parameters": 15*layers},
        ])
    write_csv(output / "parameter_counts.csv", rows)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Trainable Parameters of the Main Feature Extractors ($d=32$, $h=64$, $q=5$, $l=3$).}",
             r"\label{tab:param_count}", r"\footnotesize", r"\begin{tabular}{llrr}", r"\toprule",
             r"Component & Formula & $N_A=6$ & $N_A=10$ \\", r"\midrule",
             r"\multicolumn{4}{l}{\textit{Full-width MLP}} \\",
             r"Input layer & $2N_Ah+h$ & 832 & 1{,}344 \\",
             r"Hidden layer & $h^2+h$ & 4{,}160 & 4{,}160 \\",
             r"Output layer & $hd+d$ & 2{,}080 & 2{,}080 \\",
             r"\textbf{Total} & & \textbf{7{,}072} & \textbf{7{,}584} \\", r"\midrule",
             r"\multicolumn{4}{l}{\textit{Hybrid classical-quantum}} \\",
             r"Encoder & $2^q(2N_A+1)$ & 416 & 672 \\",
             r"Shared SU(4) blocks & $15l$ & 45 & 45 \\",
             r"Readout & $2^qd$ & 1{,}024 & 1{,}024 \\",
             r"\textbf{Total} & & \textbf{1{,}485} & \textbf{1{,}741} \\", r"\midrule",
             r"MLP / hybrid & & $4.76\times$ & $4.36\times$ \\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "TableIV.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.results_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol, matrices = load_results(args.results_dir)
    summaries, per_seed = summarize(protocol, matrices)
    contrasts = paired_contrasts(protocol, matrices)
    source = list(csv.DictReader((args.results_dir / "table_v_source.csv").open()))
    expected = {(r["topology"], r["method"]): r for r in source}
    for row in summaries:
        match = expected[row["topology"], row["method"]]
        for field in ("tail_reward", "final_cumulative_loss_ref", "tail_fairness", "tail_throughput_Mbps"):
            if abs(row[field] - float(match[field])) > 1e-9:
                raise ValueError(f"Source-table mismatch: {row['topology']}, {row['method']}, {field}")
    write_csv(output / "table_v.csv", summaries)
    write_csv(output / "per_seed_metrics.csv", per_seed)
    write_csv(output / "paired_contrasts.csv", contrasts)
    write_tables(output, protocol, summaries, contrasts)
    write_parameter_counts(output)
    checks = {"status": "PASS", "input_matrices": len(matrices), "table_v_rows": len(summaries),
              "per_seed_rows": len(per_seed), "paired_contrasts": len(contrasts),
              "source_table_fields_match": 108, "simulation_rerun": False}
    (output / "analysis_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
