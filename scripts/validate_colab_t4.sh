#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK="${1:-$ROOT/notebooks/alphago_2016_tutorial.ipynb}"
SESSION="${COLAB_SESSION:-alphago-t4-validation}"
OUTPUT="${NOTEBOOK%.ipynb}_output.ipynb"

if ! command -v colab >/dev/null 2>&1; then
  echo "Install the official CLI: uv tool install 'google-colab-cli==0.6.0'" >&2
  exit 2
fi

cleanup() {
  colab stop --session "$SESSION" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

colab version
colab new --session "$SESSION" --gpu T4
colab status --session "$SESSION"
colab exec --session "$SESSION" --file "$NOTEBOOK" --timeout 1800

uv run --project "$ROOT" python "$ROOT/scripts/scan_notebook.py" "$OUTPUT" \
  --require-text "✓ Gymnasium API" \
  --require-device cuda \
  --allow-output-only-execution

echo "Managed Colab T4 validation artifact: $OUTPUT"
