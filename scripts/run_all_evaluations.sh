#!/bin/bash
# Run both FASDD and VisDrone evaluations sequentially
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  Running All Evaluations"
echo "  $(date)"
echo "=========================================="

bash "${SCRIPT_DIR}/run_fasdd_evaluation.sh" "$@"
bash "${SCRIPT_DIR}/run_visdrone_evaluation.sh" "$@"

echo ""
echo "=========================================="
echo "  All evaluations complete."
echo "  $(date)"
echo "=========================================="
