# Copyright 2026 OpenAI
# Licensed under the Apache License, Version 2.0 (the "License").
"""Full-vocabulary mode contract: native vLLM teacher and existing FSDP2 student."""


def is_full_vocab_mode(config) -> bool:
    return bool(
        config is not None
        and config.get("enabled", False)
        and config.get("execution_mode", "teacher_server") == "full_vocab_vllm"
    )


def validate_student(config, tokenizer):
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local

    actor = config.actor_rollout_ref
    if not config.trainer.use_v1:
        raise ValueError("full_vocab_vllm requires trainer.use_v1=true")
    if actor.actor.strategy != "fsdp2" or actor.actor.fsdp_config.strategy != "fsdp2":
        raise ValueError("Full-vocabulary student requires FSDP2")
    if not actor.model.use_fused_kernels or actor.model.fused_kernel_options.get("impl_backend") != "liger":
        raise ValueError("Full-vocabulary student requires use_fused_kernels=true and impl_backend=liger")
    if actor.actor.policy_loss.get("rollout_correction", {}).get("score_centering", False):
        raise ValueError("Full-vocabulary OPD does not support score centering")
    distill = config.distillation
    if (
        distill.distillation_loss.use_task_rewards
        and float(actor.rollout.temperature) != distill.full_vocab_opd.temperature
    ):
        raise ValueError("Mixed task-reward OPD requires matching rollout and distillation temperatures")
    for teacher in distill.teacher_models.values():
        teacher_tokenizer = hf_tokenizer(copy_to_local(teacher.model_path))
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError("Teacher and student tokenizers must have identical token-to-ID mappings")
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(tokenizer, key) != getattr(teacher_tokenizer, key):
                raise ValueError(f"Teacher/student tokenizer {key} differs")
    if config.transfer_queue.backend.storage_backend != "SimpleStorage":
        raise ValueError("Full-vocabulary teacher CPU RAM storage requires TransferQueue SimpleStorage")
    if config.transfer_queue.backend.SimpleStorage.ssd_offload.enabled:
        raise ValueError("Full-vocabulary teacher storage requires RAM, with SSD offload disabled")
