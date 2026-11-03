#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${VIRTUAL_ENV:?Activate the target virtual environment first}"
liger_source="${LIGER_OPD_SOURCE:-${repo_root}/.deps/liger-kernel-0.8.4-opd}"
if [[ ! -d "$liger_source/.git" && ! -f "$liger_source/.git" ]]; then
  git clone --depth=1 --branch v0.8.4 https://github.com/linkedin/Liger-Kernel.git "$liger_source"
  git -C "$liger_source" checkout --detach 43293d84d760e402e4fd974b12410a1d47c31761
fi
[[ "$(git -C "$liger_source" rev-parse HEAD)" == 43293d84d760e402e4fd974b12410a1d47c31761 ]]
patch_path="$repo_root/patches/liger/0.8.4-selected-logprobs.patch"
if git -C "$liger_source" apply --check "$patch_path" 2>/dev/null; then
  git -C "$liger_source" apply "$patch_path"
else
  git -C "$liger_source" apply --reverse --check "$patch_path"
fi
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --editable "$liger_source"
