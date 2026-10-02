MAIN INFERENCE RESULTS

Four learned controllers, three topologies, physical seed 1, frozen final trained weights.
checkpoints/ includes all 12 measured models; protocol.json pins every checkpoint.
The 48 selected timing groups contain 1,000 calls each, with batch sizes 1 and 512 and two scopes.
raw_timings.csv includes only those selected main timing groups.
Each of two homogeneous timing blocks starts with 30 warmups and contains 500 measured calls.
Each call recomputes features for a normalized float32 input bank; no feature memoization.
Feature inference returns float64 features. The prediction scope additionally uses the restored posterior mean.
The original host is AMD Ryzen 9 9950X3D2, logical CPU 31, one Torch/BLAS thread.

Neural Bandit uses the frozen NumPy implementation.
QNN Bandit uses a fused native C++ kernel for batch 1, and C++ plus NumPy for batch 512.
Fixed gates and the readout are fused during model preparation; trained weights are unchanged.
Loading, preparation/compilation, raw-action normalization, candidates, exploration, posterior updates/draws,
retraining, ns-3 and IPC are excluded from the feature timings.

Native controller timing is recorded separately in native_baseline_latency.csv and native_traces/.
These rows time C++ action selection; INSPIRE times GP prescription/consensus with its full 32-sample window.
A static Default 802.11ax configuration has no action-selection operation to time.
TableVII.tex labels the native and feature timing scopes separately.

Reproduce measurements from the repository root, choosing an allowed logical CPU:
  python scripts/benchmark_inference.py --cpu 31 --warmup 30 --repetitions 1000 --output-dir work/inference
Actual timing varies with CPU and runtime conditions.
Verify included measurements and regenerate the table/plot:
  python scripts/verify_results.py --export-inference
