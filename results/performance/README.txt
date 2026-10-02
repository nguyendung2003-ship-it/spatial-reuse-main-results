MAIN PERFORMANCE RESULTS

Three topologies, nine methods, 25 physical seeds (1--25), measurement cycles 0--2000.
Each data directory contains reward, fairness and aggregate throughput matrices.
The first TSV row is the measurement-cycle header; the subsequent rows are seeds in order 1--25.
Some original exporters append the unused terminal header label 2001; every measured row contains exactly 2,001 values.
Frozen empirical references and the 81 original source hashes are recorded in protocol.json.

Run from the repository root:
  python scripts/analyze_main.py
  python scripts/reproduce_figures.py
  python scripts/verify_results.py

Figures/ contains the current publication export (12 PDF + 12 PNG panels).
reproduce_figures.py writes newly rendered panels to reproduced_Figures/ by default.
Table V uses medians of per-seed tail averages; Table VI uses individual paired percentile bootstrap intervals.
TableIV.tex and parameter_counts.csv describe the two main feature extractors.
