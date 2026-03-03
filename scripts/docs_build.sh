#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export NO_MKDOCS_2_WARNING=1
uv run --active --no-default-groups --extra docs mkdocs build --strict
