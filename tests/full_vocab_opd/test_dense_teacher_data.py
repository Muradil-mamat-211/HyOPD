import asyncio
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab_cache import (
    move_micro_batch_without_teacher,
    pad_teacher_logprobs,
    restore_dense_teacher_logprobs,
    teacher_action_logprobs,
)
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import prepare_micro_batches
from verl.workers.utils.padding import response_from_nested


def nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def test_dense_teacher_causal_alignment_masks_first_last_and_eos():
    lengths = [8, 3, 8]
    masks = nested([torch.tensor([1, 1, 0, 1, 1]), torch.tensor([1, 1]), torch.tensor([1, 0, 1, 1])])
    rows = [torch.arange(n).float()[:, None].expand(n, 7) + 20 * i for i, n in enumerate(lengths)]
    full = nested(rows).requires_grad_()
    data = TensorDict(
        {
            "input_ids": nested([torch.arange(n) for n in lengths]),
            "response_mask": masks,
            "teacher_logprobs": torch.stack([pad_teacher_logprobs(row.detach(), 10) for row in rows]),
        },
        [3],
    )
    student = response_from_nested(full, masks).values()[masks.values().bool()]
    teacher = teacher_action_logprobs(data, student)
    torch.testing.assert_close(teacher, student.detach())
    assert teacher[:, 0].tolist() == [2, 3, 5, 6, 20, 21, 43, 45, 46]
    assert teacher.shape == (9, 7) and not teacher.requires_grad
    student.sum().backward()
    assert full.grad is not None


def test_native_dynamic_microbatches_slice_dense_cpu_teacher_before_late_h2d(monkeypatch):
    lengths = [8, 4, 6, 5]
    rows = [torch.full((8, 7), float(i)) for i in range(4)]
    mask = nested([torch.tensor(row) for row in ([1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0])])
    teacher = torch.stack(rows).half()
    data = TensorDict(
        {
            "input_ids": nested([torch.arange(n) for n in lengths]),
            "response_mask": mask,
            "loss_mask": mask,
            "sample_id": torch.arange(4),
            "teacher_logprobs": teacher,
        },
        [4],
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=True, max_token_len_per_gpu=8, sp_size=1)
    microbatches, indexes = prepare_micro_batches(data, same_micro_num_in_dp=False)
    assert len(microbatches) > 1
    transfers = []
    expected_actions = []
    original_to = torch.Tensor.to

    def tracked_to(self, *args, **kwargs):
        if self.ndim >= 2 and self.shape[-1] == 7:
            # Teacher must already be selected on CPU; a dense microbatch transfer fails.
            assert self.ndim == 2 and self.device.type == "cpu"
            assert self.shape[0] == len(expected_actions)
            assert self[:, 0].tolist() == expected_actions
            assert kwargs["dtype"] == torch.float32
            transfers.append(tuple(self.shape))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    seen = []
    for micro in microbatches:
        before = len(transfers)
        moved = move_micro_batch_without_teacher(micro, "cpu")
        assert len(transfers) == before  # No teacher transfer before student forward.
        ids = moved["sample_id"].tolist()
        for row, identity in zip(moved["teacher_logprobs"], ids, strict=True):
            assert row[0, 0] == identity
        expected_actions[:] = [identity for identity in ids for _ in range(int(mask.unbind()[identity].sum()))]
        assert len(expected_actions) > len(ids)  # distinguish action count from microbatch size
        student_actions = torch.zeros(len(expected_actions), 7, requires_grad=True)  # current forward completed
        received = teacher_action_logprobs(moved, student_actions)
        assert len(transfers) == before + 1
        assert transfers[-1] == (len(expected_actions), 7)
        assert received[:, 0].tolist() == expected_actions
        assert received.dtype == torch.float32 and not received.requires_grad
        seen.extend(ids)
    assert sorted(seen) == [0, 1, 2, 3]
    assert teacher.device.type == "cpu" and not teacher.requires_grad


def test_rejects_non_cpu_or_gradient_bearing_teacher_and_bad_capacity():
    with pytest.raises(ValueError):
        pad_teacher_logprobs(torch.randn(4, 7, requires_grad=True), 8)
    with pytest.raises(ValueError, match="capacity"):
        pad_teacher_logprobs(torch.randn(9, 7), 8)


def test_transferqueue_nested_packing_restored_before_dense_reorder():
    rows = [torch.full((8, 7), float(i)) for i in range(3)]
    data = TensorDict({"teacher_logprobs": nested(rows), "sample_id": torch.arange(3)}, [3])
    restored = restore_dense_teacher_logprobs(data)
    assert restored is data and not data["teacher_logprobs"].is_nested
    reordered = data[torch.tensor([2, 0, 1])]
    assert reordered["teacher_logprobs"].shape == (3, 8, 7)
    assert reordered["teacher_logprobs"][:, 0, 0].tolist() == reordered["sample_id"].tolist()
    bad = TensorDict({"teacher_logprobs": nested([torch.zeros(8, 7), torch.zeros(9, 7)])}, [2])
    with pytest.raises(ValueError, match="capacity"):
        restore_dense_teacher_logprobs(bad)
    topk = TensorDict({"teacher_logprobs": nested(rows), "teacher_ids": nested([r.long() for r in rows])}, [3])
    assert restore_dense_teacher_logprobs(topk)["teacher_logprobs"].is_nested


@pytest.mark.parametrize("batch_size", [1, 3])
def test_native_tq_read_restores_actual_storage_packing_without_ray(monkeypatch, batch_size):
    from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager

    from verl.utils import transferqueue_utils as tqu

    rows = [torch.full((8, 7), float(i)) for i in range(batch_size)]
    # Run the installed TQ storage packer itself; only the remote read is replaced.
    packed = AsyncSimpleStorageManager._pack_field_values(rows)
    assert packed.is_nested
    stored = TensorDict({"teacher_logprobs": packed, "sample_id": torch.arange(batch_size)}, [batch_size])
    meta = SimpleNamespace(size=batch_size, extra_info={"return_full_log_probs": True})

    class Client:
        async def async_get_data(self, received_meta):
            assert received_meta is meta
            return stored

    monkeypatch.setattr(tqu.tq, "get_client", lambda: Client())
    restored = asyncio.run(tqu._async_meta_to_realdata(meta))
    assert restored["teacher_logprobs"].shape == (batch_size, 8, 7)
    assert not restored["teacher_logprobs"].is_nested
    assert restored["teacher_logprobs"].device.type == "cpu"
    torch.testing.assert_close(restored["teacher_logprobs"], torch.stack(rows))
    assert restored.get_non_tensor("return_full_log_probs")


def test_synthetic_padding_preserves_dense_capacity_with_zero_action_loss():
    padding, tag = construct_minimal_padding_template(
        {"teacher_logprobs": torch.randn(8, 7)}, {}, eos_token_id=2
    )
    assert padding["teacher_logprobs"].shape == (8, 7)
    assert padding["input_ids"].numel() == 128 and tag["is_padding"]
    data = TensorDict(
        {
            "input_ids": nested([padding["input_ids"]]),
            "response_mask": nested([padding["response_mask"]]),
            "teacher_logprobs": padding["teacher_logprobs"].unsqueeze(0),
        },
        [1],
    )
    selected = teacher_action_logprobs(data, torch.empty(0, 7, requires_grad=True))
    assert selected.shape == (0, 7) and not selected.requires_grad
