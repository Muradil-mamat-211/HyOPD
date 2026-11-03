"""CPU protocol/transport tests. Real CUDA vLLM execution has an opt-in smoke test."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager, _get_teacher_sampling_params
from verl.workers.rollout.full_vocab_logprobs import extract_full_vocab_prompt_logprobs, full_vocab_prompt_fields
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput


class FlatRows:
    """vLLM 0.32 FlatLogprobs wire fields; prohibit rebuilding per-row dicts."""

    def __init__(self):
        self.start_indices = [0]
        self.end_indices = [0]
        self.token_ids, self.logprobs = [], []

    def append(self, ids, logprobs):
        self.start_indices.append(len(self.token_ids))
        self.token_ids.extend(ids)
        self.logprobs.extend(logprobs)
        self.end_indices.append(len(self.token_ids))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, item):
        raise AssertionError("Flat full-vocabulary scores must not rebuild Python dicts")


def scores(sequence, vocab_size=7):
    logits = torch.stack(
        [torch.arange(vocab_size).float() * (1 + sum(sequence[: i + 1]) % 5) / 9 for i in range(len(sequence) - 1)]
    )
    return logits.log_softmax(-1)


def flat_output(sequence, vocab_size=7):
    reference = scores(sequence, vocab_size)
    flat = FlatRows()
    for i, row in enumerate(reference):
        order = torch.argsort(row, descending=True).tolist()
        # vLLM includes the target prompt token first and may repeat it in all-V.
        ids = [sequence[i + 1]] + order
        flat.append(ids, row[ids].tolist())
    return flat, reference


def teacher_config(temperature=0.7, dtype="float32"):
    return OmegaConf.create(
        {
            "distillation": {
                "_target_": "verl.workers.config.DistillationConfig",
                "enabled": True,
                "execution_mode": "full_vocab_vllm",
                "nnodes": 1,
                "n_gpus_per_node": 1,
                "teacher_models": {
                    "teacher_model": {
                        "_target_": "verl.workers.config.DistillationTeacherModelConfig",
                        "model_path": "tiny-test-model",
                        "inference": {
                            "_target_": "verl.workers.config.RolloutConfig",
                            "name": "vllm",
                            "tensor_model_parallel_size": 1,
                            "prompt_length": 8,
                            "response_length": 8,
                            "temperature": 1.0,
                            "logprobs_mode": "raw_logprobs",
                            "engine_kwargs": {"vllm": {"max_logprobs": -1}},
                        },
                    }
                },
                "full_vocab_opd": {
                    "_target_": "verl.workers.config.FullVocabularyOPDConfig",
                    "temperature": temperature,
                },
                "teacher_cache": {"_target_": "verl.workers.config.TeacherRAMCacheConfig", "dtype": dtype},
                "distillation_loss": {
                    "_target_": "verl.workers.config.DistillationLossConfig",
                    "loss_mode": "coverage_adaptive_full_vocab",
                    "use_task_rewards": False,
                    "use_policy_gradient": False,
                    "loss_max_clamp": None,
                    "log_prob_min_clamp": None,
                },
            },
            "actor_rollout_ref": {"rollout": {"name": "vllm", "full_determinism": False}},
        }
    )


class AsyncRPC:
    def __init__(self, function):
        self.remote = function


class ScoringServer:
    def __init__(self):
        self.requests, self.completed = [], []
        self.generate = AsyncRPC(self.score)

    async def score(self, request_id, prompt_ids, sampling_params, session_id, **kwargs):
        assert sampling_params == {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": -1,
            "flat_logprobs": True,
            "detokenize": False,
        }
        self.requests.append((session_id, request_id, tuple(prompt_ids)))
        await asyncio.sleep(0.02 if prompt_ids[0] == 1 else 0)
        flat, _ = flat_output(prompt_ids)
        self.completed.append(tuple(prompt_ids))
        fields = full_vocab_prompt_fields(
            SimpleNamespace(request_id=request_id, prompt_logprobs=flat), len(prompt_ids), 7, request_id, session_id
        )
        return TokenOutput(token_ids=[0], extra_fields=fields)


class InProcessRPCClient(LLMServerClient):
    """Retain native LLMServerClient.generate; replace only the Ray RPC endpoints."""

    def __init__(self, config, server):
        super().__init__(config, None)
        self.server = server
        self.released = []

    async def _acquire_server(self, request_id, **kwargs):
        return "teacher-server", self.server

    def _release_server(self, server_id, request_id=None):
        self.released.append(request_id)


def test_full_vocabulary_flat_and_dict_decode_causal_shift():
    sequence = [1, 3, 5, 2]
    flat, reference = flat_output(sequence)
    result = extract_full_vocab_prompt_logprobs(flat, len(sequence), 7)
    assert result.shape == (4, 7) and result.device.type == "cpu" and not result.requires_grad
    torch.testing.assert_close(result[:-1], reference)
    assert not result[-1].any()
    dict_rows = [None]
    for row in reference:
        dict_rows.append({i: SimpleNamespace(logprob=float(row[i])) for i in range(7)})
    torch.testing.assert_close(extract_full_vocab_prompt_logprobs(dict_rows, 4, 7), result)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nan", "length"])
def test_incomplete_or_misordered_wire_data_is_validated(corruption):
    flat, _ = flat_output([1, 2, 3])
    if corruption == "length":
        flat.end_indices.pop()
        flat.start_indices.pop()
    elif corruption == "missing":
        flat.token_ids[flat.start_indices[1] : flat.end_indices[1]] = [0] * 8
    elif corruption == "duplicate":
        flat.logprobs[flat.start_indices[1]] += 1
    else:
        flat.logprobs[1] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        extract_full_vocab_prompt_logprobs(flat, 3, 7)


def test_async_native_manager_and_client_preserve_interleaved_request_identity():
    async def run():
        cfg = teacher_config()
        server = ScoringServer()
        client = InProcessRPCClient(cfg, server)
        manager = AsyncTeacherLLMServerManager(cfg, {"default": client})
        model = manager.teacher_model_configs["default"]
        assert model.inference.engine_kwargs["vllm"]["max_logprobs"] == -1
        assert _get_teacher_sampling_params(model, manager.distillation_loss_config)["flat_logprobs"]
        sequences = [[1, 3, 2], [2, 4, 1, 3, 2]]
        outputs = await asyncio.gather(*(manager.compute_teacher_logprobs_single(seq) for seq in sequences))
        assert server.completed == [tuple(s) for s in sequences[::-1]]
        assert len(set(item[0] for item in server.requests)) == 2
        assert len(client.released) == 2
        for sequence, (ids, tensor) in zip(sequences, outputs, strict=True):
            assert ids is None and tensor.shape == (len(sequence), 7) and tensor.device.type == "cpu"
            torch.testing.assert_close(tensor[:-1], (scores(sequence) / 0.7).log_softmax(-1))
            assert not tensor.requires_grad

    asyncio.run(run())


def test_request_identity_mismatch_is_rejected():
    flat, _ = flat_output([1, 2, 3])
    with pytest.raises(ValueError, match="request identity"):
        full_vocab_prompt_fields(SimpleNamespace(request_id="other", prompt_logprobs=flat), 3, 7, "expected")
