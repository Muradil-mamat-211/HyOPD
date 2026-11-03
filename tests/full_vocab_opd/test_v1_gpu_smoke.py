"""Opt-in real V1/AgentLoop/Ray/FSDP2 smoke test; requires the GPU extras.

Run with OPD_RUN_GPU_SMOKE=1 and the example YAML's resource/model variables.
The model paths must be small enough for the configured pools. No server,
AgentLoop, TransferQueue, forward, backward, or optimizer step is mocked.
"""

import hashlib
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU FSDP2/vLLM integration requires CUDA")


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        tensor = parameter.detach()
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        digest.update(name.encode())
        digest.update(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_real_v1_student_rollout_teacher_cache_backward_update(tmp_path, monkeypatch):
    if os.environ.get("OPD_RUN_GPU_SMOKE") != "1":
        pytest.skip("Set OPD_RUN_GPU_SMOKE=1 to launch the real worker and rollout resource pools")
    import cloudpickle
    import pandas as pd
    import ray
    import transfer_queue as tq
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from verl.single_controller.base.decorator import Dispatch, register
    from verl.trainer.ppo.utils import Role
    from verl.trainer.ppo.v1 import AgentLoopManagerTQ, PPOTrainerSync
    from verl.workers.engine_workers import ActorRolloutRefWorker

    class AuditedActor(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def prepare_smoke(self):
            # Perturb the student test policy, then resync its native rollout.
            # This permits identical initial model paths with a nonzero OPD signal.
            with torch.no_grad():
                for name, p in self.actor.engine.module.named_parameters():
                    if "lm_head" in name:
                        local = p.to_local() if hasattr(p, "to_local") else p
                        local.add_(
                            torch.linspace(-0.03, 0.03, local.numel(), device=local.device, dtype=local.dtype).reshape(
                                local.shape
                            )
                        )
            return self.audit_smoke()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def audit_smoke(self):
            return {"digest": parameter_digest(self.actor.engine.module)}

    class SmokeTrainer(PPOTrainerSync):
        def _init_resource_pool_mgr(self):
            super()._init_resource_pool_mgr()
            for role in list(self.role_worker_mapping):
                if role in (Role.Actor, Role.ActorRollout, Role.ActorRolloutRef):
                    self.role_worker_mapping[role] = ray.remote(AuditedActor)

        def _update_actor(self, batch, metrics):
            result = super()._update_actor(batch, metrics)
            self.smoke_grad_norm = metrics["actor/grad_norm"]
            self.smoke_loss = metrics["actor/distillation/loss"]
            return result

    # Cloudpickle the test-local instrumentation along with the native classes.
    cloudpickle.register_pickle_by_value(__import__(__name__, fromlist=[""]))
    batch_size = int(os.environ["OPD_TRAIN_BATCH_SIZE"])
    data_path = tmp_path / "smoke.parquet"
    pd.DataFrame(
        [
            {
                "data_source": "gsm8k",
                "prompt": [{"role": "user", "content": "What is 1 + 1?"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "2"},
                "extra_info": {"index": i},
            }
            for i in range(batch_size)
        ]
    ).to_parquet(data_path)
    monkeypatch.setenv("OPD_TRAIN_FILES", str(data_path))
    monkeypatch.setenv("OPD_VAL_FILES", str(data_path))
    config_dir = str(Path(__file__).parents[2] / "examples/on_policy_distillation_trainer")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="full_vocab_adaptive_opd",
            overrides=[
                "trainer.total_training_steps=1",
                "trainer.total_epochs=1",
                "trainer.val_before_train=false",
                "trainer.test_freq=-1",
                "trainer.save_freq=-1",
                "trainer.v1.trainer_mode=sync",
                "data.max_prompt_length=32",
                "data.max_response_length=16",
            ],
        )
    OmegaConf.resolve(cfg)
    ray.init(include_dashboard=False)
    tq.init(cfg.transfer_queue)
    try:
        trainer = SmokeTrainer(cfg)
        trainer.init()
        assert trainer.get_teacher_client() is not None
        assert not hasattr(trainer, "teacher_wg")
        assert all(model.inference.name == "vllm" for model in trainer.distillation_config.teacher_models.values())
        student_before = trainer.actor_rollout_wg.prepare_smoke()
        trainer.checkpoint_manager.update_weights(trainer.global_steps)
        manager = AgentLoopManagerTQ.create(
            config=cfg,
            llm_client=trainer.get_llm_client(),
            teacher_client=trainer.get_teacher_client(),
            reward_loop_worker_handles=trainer.get_reward_handles(),
        )
        trainer.fit(manager)
        student_after = trainer.actor_rollout_wg.audit_smoke()
        assert [x["digest"] for x in student_before] != [x["digest"] for x in student_after]
        assert trainer.smoke_grad_norm > 0
        assert torch.isfinite(torch.tensor(trainer.smoke_loss))
    finally:
        tq.close()
        ray.shutdown()
