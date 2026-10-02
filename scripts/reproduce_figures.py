#!/usr/bin/env python3
"""Export twelve main result panels using only the included data and protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_main import DEFAULT_RESULTS, load_results, summarize, write_csv

ORDER = ("default", "epsilon_greedy", "gm_ts", "hcm_ts", "inspire",
         "neural_gm", "neural_hcm", "qnn_gm", "qnn_hcm")
STYLES = {
    "default": ("#4D9221", "--", 2.4),
    "epsilon_greedy": ("#E68613", "--", 2.4),
    "gm_ts": ("#8064A2", "--", 2.6),
    "hcm_ts": ("#5E3C99", "-", 2.8),
    "inspire": ("#D65F9E", "--", 2.6),
    "neural_gm": ("#2878B5", "-", 2.8),
    "neural_hcm": ("#00A69C", "-", 3.0),
    "qnn_gm": ("#C44E52", "-", 3.1),
    "qnn_hcm": ("#6C757D", "-", 3.2),
}


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with a shrinking window at both boundaries."""
    if window <= 1:
        return values
    kernel = np.ones(window)
    return np.convolve(values, kernel, "same") / np.convolve(np.ones(len(values)), kernel, "same")


def configure_style(protocol: dict) -> None:
    style = protocol["plotting"]
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.labelsize": style["axis_font_pt"],
        "xtick.labelsize": style["ticks_legend_font_pt"],
        "ytick.labelsize": style["ticks_legend_font_pt"],
        "legend.fontsize": style["ticks_legend_font_pt"],
        "savefig.dpi": 240, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def render(output: Path, topology: dict, metric: str, ylabel: str,
           matrices: dict, protocol: dict, loss: bool = False) -> None:
    names = {m["id"]: m["display"] for m in protocol["methods"]}
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    for method in ORDER:
        values = matrices[topology["id"], method, metric]
        if loss:
            values = np.cumsum(topology["frozen_reference"] - values, axis=1)
        elif metric == "throughput":
            values = values * 1e-6
        median = np.median(values, axis=0)
        low, high = np.percentile(values, (25, 75), axis=0)
        if not loss:
            window = protocol["plotting"]["smoothing_window"]
            median, low, high = (smooth(v, window) for v in (median, low, high))
        color, linestyle, width = STYLES[method]
        name = r"$\epsilon$-greedy" if method == "epsilon_greedy" else names[method]
        x = np.arange(2001)
        ax.plot(x, median, label=name, color=color, linestyle=linestyle,
                linewidth=width, zorder=4 if method.startswith("qnn") else 3)
        ax.fill_between(x, low, high, color=color, alpha=.09, linewidth=0)
    ax.set_xlabel("Measurement cycle")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 2000)
    ax.set_xticks(np.arange(0, 2001, 200))
    ax.grid(True, linestyle=":", linewidth=.9, alpha=.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left" if loss else "lower right", ncol=3,
              frameon=True, fancybox=False, edgecolor="#555555", facecolor="white",
              framealpha=.90, columnspacing=1.25, handlelength=2.5)
    fig.subplots_adjust(left=.11, right=.98, bottom=.13, top=.96)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.results_dir / "reproduced_Figures"
    output.mkdir(parents=True, exist_ok=True)
    protocol, matrices = load_results(args.results_dir)
    configure_style(protocol)
    for topology in protocol["topologies"]:
        directory = output / topology["id"]
        render(directory / "01_reward", topology, "reward", "Reward", matrices, protocol)
        # Preserve the historical filename used by the manuscript's LaTeX.
        render(directory / "02_cumulative_regret", topology, "reward",
               "Cumulative performance loss", matrices, protocol, loss=True)
        render(directory / "03_jain_fairness", topology, "fairness",
               "Jain fairness index", matrices, protocol)
        render(directory / "04_aggregate_throughput", topology, "throughput",
               "Throughput (Mbit/s)", matrices, protocol)
    summaries, _ = summarize(protocol, matrices)
    write_csv(output / "summary.csv", summaries)
    endpoints = [{"topology": row["topology"], "method": row["method"],
                  "final_cumulative_loss_ref": row["final_cumulative_loss_ref"]}
                 for row in summaries]
    write_csv(output / "loss_endpoints.csv", endpoints)
    metadata = {"panels": 12, "pdf_files": 12, "png_files": 12,
                "methods": 9, "seeds": 25, "measurement_cycles": 2001,
                "titles": False, "plotting": protocol["plotting"],
                "simulation_rerun": False,
                "frozen_references": {t["id"]: t["frozen_reference"] for t in protocol["topologies"]}}
    (output / "figure_export.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported 12 main panels (PDF and PNG): {output}")


if __name__ == "__main__":
    main()
