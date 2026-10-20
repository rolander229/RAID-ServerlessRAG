#!/usr/bin/env bash
# Sync the repo to v100, then run a command there inside the project venv.
#
#   ./scripts/vrun.sh python -m mopar.faas.runtime --model vgg16 --slices 3
#   ./scripts/vrun.sh --no-sync nvidia-smi -L
set -euo pipefail

REMOTE="${MOPAR_REMOTE:-/storage/xuedh_workspace/mopar-impl}"
HOST="${MOPAR_HOST:-v100}"

if [[ "${1:-}" == "--no-sync" ]]; then
  shift
else
  "$(dirname "${BASH_SOURCE[0]}")/sync.sh"
fi

# CUDA 12.4 wheels need no toolkit on PATH; keep the environment explicit so a
# non-interactive ssh session (which never sources ~/.bashrc) behaves like an
# interactive one.
exec ssh -o BatchMode=yes "$HOST" \
  "cd $REMOTE && export PYTHONDONTWRITEBYTECODE=0 VIRTUAL_ENV=$REMOTE/.venv PATH=$REMOTE/.venv/bin:\$PATH && $*"
