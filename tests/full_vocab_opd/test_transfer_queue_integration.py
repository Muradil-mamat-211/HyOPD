"""Optional real Ray transport check for dense CPU full-vocabulary trajectory fields."""

import os

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab_cache import CACHE_FIELD
from verl.utils.transferqueue_utils import _meta_to_realdata


@pytest.mark.skipif(os.getenv("OPD_RUN_RAY_TESTS") != "1", reason="Set OPD_RUN_RAY_TESTS=1 for local Ray transport")
def test_real_transferqueue_dense_cpu_put_get_order_slice_and_clear():
    import ray
    import transfer_queue as tq

    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=80 * 1024**2)
    tq.init(
        OmegaConf.create(
            {
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {"num_data_storage_units": 1, "ssd_offload": {"enabled": False}},
                }
            }
        )
    )
    try:
        keys = ["full-vocab-a", "full-vocab-b"]
        payload = torch.randn(2, 8, 7).half()
        fields = TensorDict({CACHE_FIELD: payload, "sample_id": torch.tensor([0, 1])}, batch_size=[2])
        meta = tq.kv_batch_put(keys=keys, partition_id="opd-test", fields=fields)
        meta.reorder([1, 0])
        # Exercise the same TQ read boundary as native actor worker dispatch.
        read = _meta_to_realdata(meta)
        tensor = read[CACHE_FIELD]
        assert not tensor.is_nested and tensor.shape == (2, 8, 7) and tensor.device.type == "cpu"
        torch.testing.assert_close(tensor[0], payload[1])
        torch.testing.assert_close(tensor[1], payload[0])
        micro = read[:1]
        assert micro["sample_id"].item() == 1
        torch.testing.assert_close(micro[CACHE_FIELD][0], payload[1])
        tq.kv_clear(keys=keys, partition_id="opd-test")
        assert not tq.kv_list("opd-test").get("opd-test", {})
    finally:
        tq.close()
        ray.shutdown()
