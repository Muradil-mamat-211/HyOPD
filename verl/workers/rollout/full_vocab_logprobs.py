# Copyright 2026 OpenAI
# Licensed under the Apache License, Version 2.0 (the "License").
"""Decode complete vLLM prompt distributions without materializing Python dicts."""

import numpy as np
import torch


def extract_full_vocab_prompt_logprobs(prompt_logprobs, prompt_length, vocab_size):
    """Return CPU [L,V], indexed by prediction state, with one final dummy row.

    vLLM position j scores token j given tokens[:j]. Shift once to row j-1 so
    native response_from_nested selects the same causal states as student forward.
    No probability-rank ordering or selected-token IDs are returned to the trainer.
    """
    if prompt_logprobs is None or len(prompt_logprobs) != prompt_length or vocab_size <= 0:
        raise ValueError("vLLM did not return the complete trajectory prompt-logprob collection")
    result = torch.zeros(prompt_length, vocab_size, dtype=torch.float32, device="cpu")
    flat = all(hasattr(prompt_logprobs, key) for key in ("start_indices", "end_indices", "token_ids", "logprobs"))
    output = result.numpy()
    for position in range(1, prompt_length):
        if flat:
            start, end = prompt_logprobs.start_indices[position], prompt_logprobs.end_indices[position]
            ids = np.asarray(prompt_logprobs.token_ids[start:end], dtype=np.int64)
            values = np.asarray(prompt_logprobs.logprobs[start:end], dtype=np.float32)
        else:
            row = prompt_logprobs[position]
            if row is None:
                raise ValueError(f"Missing full-vocabulary prompt score at position {position}")
            ids = np.fromiter((int(key) for key in row), dtype=np.int64)
            values = np.fromiter((value.logprob for value in row.values()), dtype=np.float32)
        if len(ids) not in (vocab_size, vocab_size + 1) or len(values) != len(ids):
            raise ValueError(f"Incomplete vocabulary at prompt position {position}: {len(ids)} entries, V={vocab_size}")
        if np.any(ids < 0) or np.any(ids >= vocab_size):
            raise ValueError("vLLM returned an out-of-vocabulary token ID")
        counts = np.bincount(ids, minlength=vocab_size)
        if not np.all(counts):
            raise ValueError(f"Full vocabulary is missing token IDs at prompt position {position}")
        for duplicate in np.flatnonzero(counts > 1):
            copies = values[ids == duplicate]
            if not np.all(copies == copies[0]):
                raise ValueError("vLLM returned inconsistent duplicate full-vocabulary scores")
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite raw vLLM full-vocabulary prompt logprobs")
        output[position - 1, ids] = values
    return result


def full_vocab_prompt_fields(output, prompt_length, vocab_size, request_id, session_id=None):
    """Native Ray server transport payload with explicit request and causal identity."""
    if output.request_id != request_id:
        raise ValueError("Full-vocabulary teacher result request identity mismatch")
    full = extract_full_vocab_prompt_logprobs(output.prompt_logprobs, prompt_length, vocab_size)
    return {
        "prompt_logprobs": full,
        "prompt_vocab_size": vocab_size,
        "prompt_logprob_layout": "causal_prediction",
        "prompt_scoring_request_id": session_id or request_id,
    }
