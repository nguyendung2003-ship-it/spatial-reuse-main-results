# Representation-aware spatial reuse: main results

This repository contains the final main results, experiment code, recorded configurations, and reproduction commands for three dense IEEE 802.11ax WLAN topologies. The performance data cover nine controllers, physical seeds 1–25, and 2,001 measurement cycles per run.

Neural Bandit uses the full-width MLP with ReLU hidden layers and a `tanh` output normalized to norm `sqrt(d)`. The HCM results use candidate-selection `epsilon_f = 0.1`. Figure legends retain the main method names.

## Find the results

| Artifact | Location |
| --- | --- |
| Main reward, cumulative loss, fairness, and throughput figures; PDF and PNG | [results/performance/Figures](results/performance/Figures) |
| Table IV: feature extractor parameter counts | [TableIV.tex](results/performance/TableIV.tex) |
| Tables V–VI: main results and paired confidence intervals | [tables_v_vi.tex](results/performance/tables_v_vi.tex) |
| Main results at full precision | [table_v.csv](results/performance/table_v.csv) |
| Metrics for every physical seed | [per_seed_metrics.csv](results/performance/per_seed_metrics.csv) |
| Paired bootstrap contrasts | [paired_contrasts.csv](results/performance/paired_contrasts.csv) |
| Table VII: current CPU latency measurements | [TableVII.tex](results/inference/TableVII.tex) |
| Inference figure and timing samples | [results/inference](results/inference) |
| Run configuration | [main_experiment.json](configs/main_experiment.json) |
| Expanded commands for all 27 method/topology combinations | [replay_plan.json](configs/replay_plan.json) |

The included controllers are Default 802.11ax, epsilon-greedy, GM-TS, HCM-TS, INSPIRE, Neural Bandit (GM), QNN Bandit (GM), Neural Bandit (HCM), and QNN Bandit (HCM).

| Topology identifier | Display name | APs | STAs |
| --- | --- | ---: | ---: |
| `C6o` | C6 | 6 | 12 |
| `MER_FLOORS_CH20_S5` | ENT-Clustered | 10 | 50 |
| `MER_FLOORS_BAD_DIM` | ENT-Dispersed | 10 | 50 |

## Install the environment

Use Linux and Python 3.10. The bundled ns-3.35 Waf build requires this Python version. The Conda environment installs the C/C++ toolchain, OpenBLAS, and the pinned Python packages:

```bash
conda env create -f environment.yml
conda activate spatial-reuse-main
```

If a Python 3.10 environment and C/C++ toolchain are already available:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Run all commands below from the repository root. New simulations, builds, and benchmarks are written under `work/`, which Git ignores.

## Verify and reproduce the included tables and figures

These commands use the included measurements:

```bash
python scripts/verify_results.py
python scripts/analyze_main.py --output-dir work/recomputed_tables
python scripts/reproduce_figures.py --output-dir work/reproduced_Figures
```

To regenerate Table VII and the inference figure from the included timing samples:

```bash
python scripts/verify_results.py --export-inference
```

The checks validate all 81 performance matrices, the 27 main result rows, nine paired contrasts, the archived figure hashes, 12 trained checkpoints, and 48,000 feature/prediction timing samples. Recreated figures use the labels “Measurement cycle” and “Cumulative performance loss”, with larger fonts and no plot titles.

### Metric definitions

Each TSV contains physical seeds 1–25 in row order and measurement cycles 0–2000 in column order. Reward, fairness, and throughput summaries are the cross-seed medians of per-seed averages over cycles 1901–2000. Throughput is stored in bit/s and reported in Mbps.

Cumulative performance loss for each seed is `sum(reference - reward)` over all 2,001 cycles. The reported value is its cross-seed median. The frozen empirical references are `0.92556556` for C6, `0.95215664` for ENT-Clustered, and `0.85254904` for ENT-Dispersed. They are empirical references, not theoretical optima. The `cum.tsv` suffix of native runner outputs denotes aggregate throughput; the analysis script computes cumulative loss from rewards.

Table VI uses differences of cross-seed median tail rewards, with 10,000 paired bootstrap resamples and RNG seed `20260923`. The same seed indices are resampled for both methods. Its displayed intervals are individual 95% percentile intervals.

## Build and replay the ns-3 experiments

The ns-3.35 source archive is bundled. The build script applies the two recorded Wi-Fi stability fixes and copies the included experiment sources and three topology files into the simulator:

```bash
python scripts/build_ns3.py --jobs 4
```

First check the full run plan and run a short C6 check. The smoke mode runs seed 1 for 7.5 simulated seconds, enough to exercise the first feature retraining:

```bash
python scripts/run_main.py --dry-run
python scripts/run_main.py --mode smoke --topologies C6o --jobs 1 --run-id smoke_check
```

To replay all nine controllers across all three topologies and 25 seeds:

```bash
python scripts/run_main.py --mode full --jobs 4 --run-id main_replay
python scripts/collect_replay.py --run-id main_replay
python scripts/analyze_main.py --results-dir work/replay_results
python scripts/reproduce_figures.py --results-dir work/replay_results --output-dir work/replay_results/Figures
```

This full replay consists of 675 simulations, each spanning 150 simulated seconds. Seeds run in parallel within a method/topology combination. `--jobs` controls that concurrency. The collector requires all 27 combinations and 25 seeds; the archived measurements under `results/` remain the reference package.

For a selected controller or seed range:

```bash
python scripts/run_main.py --mode full --methods neural_hcm qnn_hcm --topologies C6o --seed-start 1 --seed-end 5 --jobs 2 --run-id selected_replay
```

The runner reads [main_experiment.json](configs/main_experiment.json), sets the experiment environment explicitly, and writes a launch plan with its commands beside each run. The shared setup uses ns-3.35, UDP traffic, Minstrel-HT, a 20 MHz channel, 75 ms measurement cycles, and log-distance propagation with exponent 3.0. Learned controllers use `d=32`, prior `(lambda0, alpha0, beta0)=(1.0, 0.5, 0.025)`, and Adam with learning rate `0.001`, batch size 64, and 80 epochs. Retraining starts at round 20, occurs every 10 rounds through round 300, and every 20 rounds afterward. Each method's candidate and acquisition settings are recorded in the configuration.

## Repeat the inference measurements

This benchmark loads the 12 included trained checkpoints and freezes their weights. Each timed call computes fresh features; preparation, gate fusion, compilation, training, posterior sampling, ns-3, and IPC are outside the measured region:

```bash
python scripts/benchmark_inference.py --warmup 30 --repetitions 1000 --output-dir work/inference
```

It measures batch sizes 1 and 512 for the four learned controllers on all three topologies. Neural Bandit uses NumPy. QNN Bandit uses a fused C++ kernel with NumPy for the batched readout. The benchmark verifies outputs against the trained PyTorch model and checks that weights and history do not change. Pass `--cpu` to choose an allowed logical CPU; otherwise it selects the first allowed CPU.

Native controller timings can be replayed separately after building ns-3:

```bash
python scripts/benchmark_native.py --output-dir work/native_inference
```

These native traces measure C++ controller operations while the network supplies the observation history. Table VII identifies their scope separately from feature inference. The static Default configuration has no action-selection operation to time. The archived latency measurements were taken on an AMD Ryzen 9 9950X3D2 using one CPU thread.

## Repository layout and validation

```text
code/       Experiment sources, frozen inference, ns-3 archive, patches, notices
configs/    Recorded run configuration and replay commands
results/    Main performance data, figures, tables, checkpoints, timing samples
scripts/    Build, run, collect, analyze, plot, verify, and benchmark commands
work/       Local builds and new runs; excluded from Git and the release ZIP
```

The release was checked with a fresh ns-3 build, all nine controllers on C6, and retraining smoke runs for all four learned controllers. Main tables and curve endpoints were checked against the included measurements. Details are in [configs/validation.json](configs/validation.json), [code/build_validation.json](code/build_validation.json), and [results/verification.json](results/verification.json).

Source provenance and third-party license details are in [code/NOTICE.txt](code/NOTICE.txt) and [code/source_manifest.json](code/source_manifest.json). [MANIFEST.sha256](MANIFEST.sha256) pins the released files.
