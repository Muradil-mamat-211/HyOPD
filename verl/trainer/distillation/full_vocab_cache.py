# Copyright 2026 OpenAI
# Licensed under the Apache License, Version 2.0 (the "License").
"""Dense CPU teacher trajectories, native microbatch slicing and action-only H2D."""

import torch
from torch.nn import functional as F

CACHE_FIELD = "teacher_logprobs"


def restore_dense_teacher_logprobs(data):
    """Undo TransferQueue's automatic nested packing of equal-capacity CPU rows."""
    if CACHE_FIELD not in data or "teacher_ids" in data:
        return data
    teacher = data[CACHE_FIELD]
    if teacher.device.type != "cpu" or teacher.requires_grad or teacher.ndim != 3:
        raise ValueError("Full-vocabulary teacher storage must be detached CPU [B,T,V]")
    if teacher.is_nested:
        rows = teacher.unbind()
        if not rows or any(row.shape != rows[0].shape for row in rows):
            raise ValueError("Full-vocabulary teacher trajectories must have the same padded capacity")
        data[CACHE_FIELD] = torch.stack(rows)
    return data


def pad_teacher_logprobs(log_probs, capacity):
    if log_probs.device.type != "cpu" or log_probs.requires_grad or log_probs.ndim != 2:
        raise ValueError("Teacher logprobs must be detached CPU [sequence,vocabulary]")
    if log_probs.shape[0] > capacity:
        raise ValueError(f"Teacher trajectory length {log_probs.shape[0]} exceeds configured capacity {capacity}")
    return F.pad(log_probs, (0, 0, 0, capacity - log_probs.shape[0]))


def move_micro_batch_without_teacher(micro_batch, device):
    """Move model inputs only; teacher rows remain on CPU until after student forward."""
    if CACHE_FIELD not in micro_batch or "teacher_ids" in micro_batch:
        return micro_batch.to(device)
    teacher = micro_batch[CACHE_FIELD]
    if teacher.device.type != "cpu" or teacher.requires_grad or teacher.is_nested or teacher.ndim != 3:
        raise ValueError("Full-vocabulary teacher TensorDict field must be dense detached CPU [B_micro,T,V]")
    model_data = micro_batch.exclude(CACHE_FIELD).to(device)
    model_data.clear_device_()
    model_data[CACHE_FIELD] = teacher
    return model_data


def teacher_action_logprobs(data, student_actions):
    """Select identical prediction rows on CPU, then transfer only action distributions.

    Server extraction has already mapped prompt score[j+1] to prediction row[j].
    Student response_from_nested performs the same single causal selection.
    """
    teacher_cpu = data[CACHE_FIELD]
    if teacher_cpu.device.type != "cpu" or teacher_cpu.requires_grad or teacher_cpu.is_nested or teacher_cpu.ndim != 3:
        raise ValueError("Expected detached dense CPU teacher_logprobs [B_micro,T,V] before loss H2D")
    if teacher_cpu.shape[0] != len(data) or teacher_cpu.shape[-1] != student_actions.shape[-1]:
        raise ValueError("Teacher/student microbatch or full vocabulary dimensions differ")
    # Model inputs/masks may already be on the Student device after its forward.
    # Copy only the small alignment metadata back; never transfer dense Teacher rows.
    response_mask = data["response_mask"].detach().cpu()
    sequence_lengths = data["input_ids"].offsets().detach().cpu().diff()
    response_lengths = response_mask.offsets().diff()
    rows = []
    for i, mask in enumerate(response_mask.unbind()):
        action_positions = mask.bool().nonzero().flatten()
        prediction_rows = sequence_lengths[i] - response_lengths[i] - 1 + action_positions
        if prediction_rows.numel() and (prediction_rows.min() < 0 or prediction_rows.max() >= teacher_cpu.shape[1] - 1):
            raise ValueError("Teacher action prediction rows are out of range or lack a scored next token")
        rows.append(teacher_cpu[i].index_select(0, prediction_rows))
    selected = torch.cat(rows, dim=0) if rows else teacher_cpu.new_empty((0, teacher_cpu.shape[-1]))
    if selected.shape != student_actions.shape:
        raise ValueError(f"Teacher/student action distributions differ: {selected.shape} vs {student_actions.shape}")
    return selected.to(device=student_actions.device, dtype=torch.float32, non_blocking=True)
