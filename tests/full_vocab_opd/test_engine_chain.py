from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transformers import LlamaConfig, LlamaForCausalLM

from verl.models.transformers.dense_common import forward_with_torch_backend
from verl.trainer.distillation.full_vocab_cache import pad_teacher_logprobs
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationLossConfig, FullVocabularyOPDConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def make_batch():
    rows = [torch.tensor([1, 3, 4, 5, 6, 2]), torch.tensor([1, 7, 8, 2])]
    mask = nested([torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1])])
    data = TensorDict(
        {
            "input_ids": nested(rows),
            "position_ids": nested([torch.arange(len(r)) for r in rows]),
            "response_mask": mask,
            "loss_mask": mask,
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        use_remove_padding=False,
        use_fused_kernels=True,
        return_full_log_probs=True,
        temperature=0.7,
        calculate_entropy=False,
        distillation_only=True,
        max_response_len=4,
        dp_size=1,
        batch_num_tokens=5,
        global_batch_size=2,
    )
    return data


def make_engine(model):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = model
    engine.use_ulysses_sp = False
    engine.pad_to_length = False
    engine.model_config = SimpleNamespace(fused_kernel_options={"impl_backend": "liger"})
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False)
    engine._autocast_dtype = torch.float32
    engine.get_data_parallel_group = lambda: None
    return engine


def test_real_model_engine_cache_loss_backward_and_update(monkeypatch):
    torch.manual_seed(27)
    cfg = LlamaConfig(
        vocab_size=17,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    student = LlamaForCausalLM(cfg)
    teacher = LlamaForCausalLM(cfg).eval().requires_grad_(False)
    batch = make_batch()
    # A CPU HuggingFace reference supplies prompt scores in this unit test;
    # real vLLM server execution is covered separately on CUDA-capable hosts.
    lengths = batch["input_ids"].offsets().diff()
    dense_ids = batch["input_ids"].to_padded_tensor(0)
    attention_mask = torch.arange(dense_ids.shape[1]) < lengths[:, None]
    with torch.no_grad():
        teacher_raw = teacher(input_ids=dense_ids, attention_mask=attention_mask).logits.float()
        cached_rows = []
        for row, length in zip(teacher_raw, lengths, strict=True):
            log_probs = (row[:length] / 0.7).log_softmax(-1)
            log_probs[-1].zero_()
            cached_rows.append(pad_teacher_logprobs(log_probs, 8))
    batch["teacher_logprobs"] = torch.stack(cached_rows)
    monkeypatch.setattr(LlamaForCausalLM, "forward", forward_with_torch_backend)
    student._verl_fused_kernels_backend = "liger"
    se = make_engine(student)
    args, out_args = se.prepare_model_inputs(batch)
    raw = student(**args, use_cache=False)
    output = se.prepare_model_outputs(raw, out_args, batch, None)
    assert output["full_log_probs"].values().shape == (10, 17)
    assert output["full_log_probs"].requires_grad
    distill = SimpleNamespace(
        full_vocab_opd=FullVocabularyOPDConfig(temperature=0.7),
        distillation_loss=DistillationLossConfig(
            loss_mode="coverage_adaptive_full_vocab",
            use_task_rewards=False,
            use_policy_gradient=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
        ),
    )
    actor = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None, global_batch_info={})
    loss, metrics = distillation_ppo_loss(actor, distill, output, batch)
    assert loss.isfinite() and "coverage/p50" in metrics
    before = student.lm_head.weight.detach().clone()
    opt = torch.optim.SGD(student.parameters(), lr=0.1)
    loss.backward()
    assert student.lm_head.weight.grad.norm() > 0
    assert student.model.layers[0].self_attn.q_proj.weight.grad.norm() > 0
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    assert not torch.equal(before, student.lm_head.weight)
    assert all(p.grad is None for p in teacher.parameters())


@pytest.mark.parametrize("remove_padding", [False, True])
def test_engine_extracts_full_vocab_and_keeps_graph(remove_padding):
    batch = make_batch()
    tu.assign_non_tensor(batch, use_remove_padding=remove_padding)
    engine = make_engine(None)
    if remove_padding:
        values = torch.randn(1, 10, 17, requires_grad=True)
        out_args = {
            "pad_size": 0,
            "input_ids_rmpad_rolled": torch.zeros(10, dtype=torch.long),
            "temperature_rmpad": torch.ones(10),
        }
    else:
        values = torch.randn(2, 6, 17, requires_grad=True)
        out_args = {}
    output = SimpleNamespace(full_log_probs=values, log_probs=torch.zeros(values.shape[:-1]), entropy=None)
    result = engine.prepare_model_outputs(output, out_args, batch, None)["full_log_probs"]
    assert result.values().shape == (10, 17)
    result.values().square().sum().backward()
    assert values.grad is not None


def test_native_global_token_normalization_matches_uneven_dp_partition():
    from verl.trainer.distillation.coverage_adaptive_loss import coverage_adaptive_full_vocab
    from verl.trainer.ppo.core_algos import agg_loss

    torch.manual_seed(3)
    weights = torch.randn(4, 7, requires_grad=True)
    states = torch.randn(5, 4)
    teacher = torch.randn(5, 7).log_softmax(-1)
    settings = FullVocabularyOPDConfig()
    full, _ = coverage_adaptive_full_vocab(teacher, (states @ weights).log_softmax(-1), settings)
    baseline = torch.autograd.grad(full.mean(), weights, retain_graph=True)[0]
    grads = []
    # Each rank uses the native global denominator and DP compensation. FSDP
    # averages gradients over DP ranks after microbatch accumulation.
    for subset in (slice(0, 1), slice(1, 5)):
        local = full[subset][None]
        scaled = agg_loss(
            local, torch.ones_like(local), "token-mean", dp_size=2, batch_num_tokens=5, global_batch_size=2
        )
        grads.append(torch.autograd.grad(scaled, weights, retain_graph=True)[0])
    torch.testing.assert_close((grads[0] + grads[1]) / 2, baseline)
