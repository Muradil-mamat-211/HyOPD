#!/usr/bin/env bash
set -euo pipefail
example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$example_dir/../.." && pwd)"
cd "$repo_root"
exec "${OPD_PYTHON:-python}" -m verl.trainer.main_ppo \
  --config-path "$example_dir" --config-name full_vocab_adaptive_opd "$@"
