#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Virtualenv missing. Run: ./scripts/bootstrap.sh"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

# Load .env if present
if [[ -f ".env" ]]; then
  set -a
  . "./.env"
  set +a
fi

echo "Environment activated."
python --version
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-not-set}"

