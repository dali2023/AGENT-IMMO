#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/activate.sh"
jupyter lab --NotebookApp.default_url=/lab/tree/notebooks --port 8888 --no-browser --ip 0.0.0.0

