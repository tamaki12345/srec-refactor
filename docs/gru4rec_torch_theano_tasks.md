# GRU4Rec Torch-Theano Alignment Tasks

This file tracks the next implementation steps in order.

## Task List

- [x] 1. Add a strict Torch config for Theano-compatible comparison.
  - Goal: keep behavior closest to Theano settings.
  - Output: experiment/m4a_torch_theano_strict.yml

- [x] 2. Extend the comparison runner to execute multiple targets in sequence.
  - Goal: run Torch strict, Torch optimized, and Theano from one script.
  - Output: scripts/run_m4a_torch_theano_compare.sh updates

- [x] 3. Add quick-run mode and automatic throughput summary.
  - Goal: support short verification runs and produce a markdown table.
  - Output: results/compare/summary_<timestamp>.md

- [x] 4. Execute the script and validate generated artifacts.
  - Goal: confirm logs and summary are generated correctly.

- [x] 5. Update this file with completion state and key notes.

## Execution Notes

- Quick run command:
  - QUICK_RUN=1 QUICK_TIMEOUT_SEC=65 USE_GPU=1 bash scripts/run_m4a_torch_theano_compare.sh
- Summary artifact:
  - results/compare/summary_20260526_154520.md
- Observed quick-run throughput:
  - Torch strict parity: 1687.14 pair/s
  - Torch optimized: 3996.91 pair/s
  - Theano GPU: 85559.82 pair/s
