#!/usr/bin/env bash
# Push the local editing mirror to v100.
#
# The authoritative git repo and the only place experiments run is
# v100:/storage/xuedh_workspace/mopar-impl. This directory is just an editing
# mirror, so the copy is one-way and deliberately has no --delete: a stale file
# on the server is recoverable, a wiped runs/ directory is not.
set -euo pipefail

HOST="${MOPAR_HOST:-v100}"
REMOTE="${MOPAR_REMOTE:-/storage/xuedh_workspace/mopar-impl}"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec rsync -a -e "ssh -o BatchMode=yes" \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.venv/' \
  --exclude '.uvcache/' \
  --exclude 'weights/' \
  --exclude 'runs/' \
  --exclude '.venv-install.log' \
  "$LOCAL/" "$HOST:$REMOTE/"
