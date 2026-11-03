# Reproducible Liger 0.8.4 selected-logprob backport

Base: `43293d84d760e402e4fd974b12410a1d47c31761` (v0.8.4).
Selected-logprob source: upstream main `8c7a5078318da1295dc3ca445f9d85e9e9e153f5`.

`0.8.4-selected-logprobs.patch` contains only four operator files:

- Device-dispatch frontend `ops/fused_linear_scaled_cross_entropy.py`.
- SM90 autograd wrapper `ops/cutedsl/ops/fused_scaled_cross_entropy_sm90.py`.
- SM90 forward fragment and backward fused implementation.

The patch backports the upstream differentiable selected outputs, full-vocabulary
normalizer, ignore-index masks, duplicate-ID scatter-add gradients and entropy
composition. It keeps the existing NLL/entropy API.

Two local correctness/compilation changes are included:

1. Always pass the optional selected-ID argument into the autograd Function,
   including `None`, so backward returns exactly as many gradients as inputs.
2. Detect `selected_token_ids == arange(V)` and dispatch it to Liger's existing
   portable GEMM/FP32-softmax autograd implementation. The upstream SM90 selected
   implementation statically unrolls K; compiling K=V would be impractical. This
   path still returns every normalized vocabulary entry and backpropagates exact
   log-softmax gradients. It is **not** a new native SM90 full-vocabulary fused
   implementation, and no native-fusion speedup is claimed for full-vocabulary OPD.
   Small selected subsets continue to use the backported native SM90 implementation
   where Liger's architecture dispatcher selects it. Non-BF16 inputs use the
   portable implementation because the native SM90 operator requires BF16. Full
   ranges do not import unused optional native SM90 dependencies.

Liger internally tiles tokens; the complete `[N_micro,V]` output is materialized.
The trainer does not stream chunks or truncate vocabulary.

Activate the intended environment, then run:

```bash
bash patches/liger/install.sh
```

The installer checks the exact base SHA, applies or verifies the patch, and uses
`uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --editable ...`.
It never patches global site-packages. Set `LIGER_OPD_SOURCE` to reuse a local
source checkout at the pinned base. Subsequent `uv sync` may reinstall the upstream
wheel; rerun this installer afterwards. The FSDP2 student requires this patched package. The native vLLM teacher uses
vLLM prompt scoring and does not use Liger. Student full-vocabulary requests reject
an unpatched package explicitly.

To review:

```bash
git -C "$LIGER_OPD_SOURCE" diff
```

To validate after installation:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/full_vocab_opd/test_kernel.py
```

The upstream Liger files retain their licensing; this patch is distributed under
the repository's Apache-2.0 license. The portable formulas also derive from veRL's
Apache-2.0 FusedLinearForPPO implementation, as acknowledged in Liger's frontend.
