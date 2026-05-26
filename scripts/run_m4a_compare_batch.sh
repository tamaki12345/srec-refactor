#!/usr/bin/env bash
set -euo pipefail

# Run the existing Torch/Theano compare script multiple times and aggregate pair/s.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPARE_SCRIPT="$ROOT_DIR/scripts/run_m4a_torch_theano_compare.sh"
OUT_DIR="$ROOT_DIR/results/compare"
TS="$(date +%Y%m%d_%H%M%S)"

RUNS="${RUNS:-3}"
OUT_MD="$OUT_DIR/aggregate_${TS}.md"
OUT_CSV="$OUT_DIR/aggregate_${TS}.csv"

if [[ ! -f "$COMPARE_SCRIPT" ]]; then
  echo "[ERROR] Missing compare script: $COMPARE_SCRIPT"
  exit 1
fi

mkdir -p "$OUT_DIR"

declare -a STRICT_VALUES=()
declare -a OPT_VALUES=()
declare -a THEANO_VALUES=()

to_number_or_empty() {
  local value="$1"
  if [[ "$value" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "$value"
  else
    echo ""
  fi
}

extract_pair_s_from_summary() {
  local summary_file="$1"
  local label="$2"
  awk -F'|' -v key="$label" '
    /^\|/ && $2 ~ key {
      gsub(/^[ \t]+|[ \t]+$/, "", $3)
      print $3
      exit
    }
  ' "$summary_file"
}

mean_of_values() {
  local values="$1"
  awk '
    {
      if ($1 != "") {
        sum += $1
        n += 1
      }
    }
    END {
      if (n == 0) {
        print "N/A"
      } else {
        printf "%.2f", sum / n
      }
    }
  ' <<< "$values"
}

ratio_or_na() {
  local num="$1"
  local den="$2"
  awk -v n="$num" -v d="$den" '
    BEGIN {
      if (n == "N/A" || d == "N/A") {
        print "N/A"
      } else if (d + 0 == 0) {
        print "N/A"
      } else {
        printf "%.4f", (n + 0) / (d + 0)
      }
    }
  '
}

echo "[INFO] Batch compare start: RUNS=$RUNS"
for i in $(seq 1 "$RUNS"); do
  echo "[INFO] Run $i/$RUNS"

  BEFORE_COUNT="$(find "$OUT_DIR" -maxdepth 1 -type f -name 'summary_*.md' | wc -l)"
  QUICK_RUN="${QUICK_RUN:-1}" QUICK_TIMEOUT_SEC="${QUICK_TIMEOUT_SEC:-60}" USE_GPU="${USE_GPU:-0}" \
    bash "$COMPARE_SCRIPT"
  AFTER_COUNT="$(find "$OUT_DIR" -maxdepth 1 -type f -name 'summary_*.md' | wc -l)"

  if [[ "$AFTER_COUNT" -le "$BEFORE_COUNT" ]]; then
    echo "[ERROR] No new summary file was created for run $i"
    exit 2
  fi

  SUMMARY_FILE="$(ls -t "$OUT_DIR"/summary_*.md | head -n 1)"

  STRICT_RAW="$(extract_pair_s_from_summary "$SUMMARY_FILE" "Torch strict parity")"
  OPT_RAW="$(extract_pair_s_from_summary "$SUMMARY_FILE" "Torch optimized")"
  THEANO_RAW="$(extract_pair_s_from_summary "$SUMMARY_FILE" "Theano")"

  STRICT_VALUES+=("$(to_number_or_empty "$STRICT_RAW")")
  OPT_VALUES+=("$(to_number_or_empty "$OPT_RAW")")
  THEANO_VALUES+=("$(to_number_or_empty "$THEANO_RAW")")
done

STRICT_LINES="$(printf "%s\n" "${STRICT_VALUES[@]}")"
OPT_LINES="$(printf "%s\n" "${OPT_VALUES[@]}")"
THEANO_LINES="$(printf "%s\n" "${THEANO_VALUES[@]}")"

STRICT_MEAN="$(mean_of_values "$STRICT_LINES")"
OPT_MEAN="$(mean_of_values "$OPT_LINES")"
THEANO_MEAN="$(mean_of_values "$THEANO_LINES")"

RATIO_OPT_TO_STRICT="$(ratio_or_na "$OPT_MEAN" "$STRICT_MEAN")"
RATIO_OPT_TO_THEANO="$(ratio_or_na "$OPT_MEAN" "$THEANO_MEAN")"

{
  echo "run,torch_strict_pair_s,torch_optimized_pair_s,theano_pair_s"
  for idx in $(seq 1 "$RUNS"); do
    echo "$idx,${STRICT_VALUES[$((idx-1))]:-},${OPT_VALUES[$((idx-1))]:-},${THEANO_VALUES[$((idx-1))]:-}"
  done
  echo "mean,$STRICT_MEAN,$OPT_MEAN,$THEANO_MEAN"
} > "$OUT_CSV"

{
  echo "# M4A Compare Batch Summary ($TS)"
  echo
  echo "- RUNS: $RUNS"
  echo "- QUICK_RUN: ${QUICK_RUN:-1}"
  echo "- QUICK_TIMEOUT_SEC: ${QUICK_TIMEOUT_SEC:-60}"
  echo "- USE_GPU: ${USE_GPU:-0}"
  echo
  echo "## Per-run pair/s"
  echo "| Run | Torch strict | Torch optimized | Theano |"
  echo "|---|---:|---:|---:|"
  for idx in $(seq 1 "$RUNS"); do
    echo "| $idx | ${STRICT_VALUES[$((idx-1))]:-N/A} | ${OPT_VALUES[$((idx-1))]:-N/A} | ${THEANO_VALUES[$((idx-1))]:-N/A} |"
  done
  echo
  echo "## Mean pair/s"
  echo "| Metric | Value |"
  echo "|---|---:|"
  echo "| Torch strict mean | $STRICT_MEAN |"
  echo "| Torch optimized mean | $OPT_MEAN |"
  echo "| Theano mean | $THEANO_MEAN |"
  echo "| optimized / strict | $RATIO_OPT_TO_STRICT |"
  echo "| optimized / theano | $RATIO_OPT_TO_THEANO |"
  echo
  echo "CSV: $OUT_CSV"
} > "$OUT_MD"

echo "[INFO] Aggregate markdown: $OUT_MD"
echo "[INFO] Aggregate csv: $OUT_CSV"
