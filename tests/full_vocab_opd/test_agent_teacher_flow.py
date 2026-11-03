import asyncio
from types import SimpleNamespace

from omegaconf import OmegaConf
from test_vllm_teacher import AsyncTeacherLLMServerManager, InProcessRPCClient, ScoringServer, teacher_config

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.teacher_loop.teacher_model import MultiTeacherModelManager
from verl.trainer.ppo.v1 import agent_loop_tq
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ

WorkerTQ = AgentLoopWorkerTQ.__ray_metadata__.modified_class


def test_completed_trajectories_score_before_reward_and_store_dense_cpu_fields(monkeypatch):
    async def run():
        cfg = teacher_config()
        server = ScoringServer()
        client = InProcessRPCClient(cfg, server)
        manager = AsyncTeacherLLMServerManager(cfg, {"default": client})
        scoring_started = asyncio.Event()
        requests = []
        original = manager.compute_teacher_logprobs_single

        async def observed(sequence_ids, **kwargs):
            requests.append(tuple(sequence_ids))
            if len(requests) == 2:
                scoring_started.set()
            return await original(sequence_ids, **kwargs)

        manager.compute_teacher_logprobs_single = observed

        class Worker:
            _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
            _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
            _compute_position_ids = AgentLoopWorker._compute_position_ids
            distillation_enabled = True
            teacher_key = "data_source"
            teacher_server_manager = manager
            rollout_config = OmegaConf.create({"prompt_length": 8, "response_length": 8})
            processor = None

            async def _compute_score(self, outputs, kwargs):
                # A slow reward cannot prevent either completed trajectory from scoring.
                await asyncio.wait_for(scoring_started.wait(), 1.0)

        outputs = [
            AgentLoopOutput(
                prompt_ids=[1, 3],
                response_ids=[4, 5, 2],
                response_mask=[1, 0, 1],
                num_turns=2,
                metrics=AgentLoopMetrics(),
            ),
            AgentLoopOutput(
                prompt_ids=[2], response_ids=[3, 2], response_mask=[1, 1], num_turns=1, metrics=AgentLoopMetrics()
            ),
        ]
        captured = {}

        async def put(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", put)
        await WorkerTQ._agent_loop_postprocess(
            Worker(), outputs, validate=False, uid="sample", session_id=0, global_steps=1, data_source="math"
        )
        fields = captured["fields"]
        assert requests == [(1, 3, 4, 5, 2), (2, 3, 2)]
        assert captured["keys"] == ["sample_0_0", "sample_0_1"]
        assert "teacher_ids" not in fields
        teacher = fields["teacher_logprobs"]
        assert not teacher.is_nested and teacher.shape == (2, 16, 7)
        assert teacher.device.type == "cpu" and not teacher.requires_grad
        assert fields["response_mask"][0].tolist() == [1, 0, 1]
        assert teacher[0, 5:].count_nonzero() == 0
        assert teacher[1, 3:].count_nonzero() == 0

    asyncio.run(run())


def test_teacher_client_uses_teacher_backend_when_student_rollout_is_sglang():
    config = teacher_config()
    config.actor_rollout_ref.rollout.name = "sglang"
    manager = MultiTeacherModelManager.__new__(MultiTeacherModelManager)
    manager.config = config
    manager.teacher_model_managers = {
        "default": SimpleNamespace(
            load_balancer_handle=None, teacher_model_config=SimpleNamespace(inference=SimpleNamespace(name="vllm"))
        )
    }
    client = manager.get_client()["default"]
    assert client.config.actor_rollout_ref.rollout.name == "vllm"
    assert config.actor_rollout_ref.rollout.name == "sglang"
