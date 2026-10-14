#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate.sh"
PORT="${STREAMLIT_PORT:-8501}"
streamlit run src/models/app7.py --server.port "$PORT" --server.address 0.0.0.0
