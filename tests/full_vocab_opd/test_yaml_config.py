from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, PreTrainedTokenizerFast

from verl.trainer.distillation.full_vocab import validate_student
from verl.workers.config import DistillationLossConfig, FullVocabularyOPDConfig


def configure(monkeypatch, tmp_path):
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "a": 4, "b": 5}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    LlamaConfig(
        vocab_size=6,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        architectures=["LlamaForCausalLM"],
    ).save_pretrained(tmp_path)
    for name, val in {
        "STUDENT_MODEL": str(tmp_path),
        "TEACHER_MODEL": str(tmp_path),
        "STUDENT_NNODES": "1",
        "STUDENT_GPUS_PER_NODE": "3",
        "TEACHER_NNODES": "2",
        "TEACHER_GPUS_PER_NODE": "5",
        "MINI_BATCH_SIZE": "30",
        "TRAIN_BATCH_SIZE": "30",
        "STUDENT_MAX_TOKENS": "64",
        "TEACHER_MAX_BATCHED_TOKENS": "128",
        "TEACHER_TP_SIZE": "1",
        "OLD_LOGPROB_MAX_TOKENS": "64",
        "TRAIN_FILES": "train.parquet",
        "VAL_FILES": "val.parquet",
    }.items():
        monkeypatch.setenv("OPD_" + name, val)
    config_dir = str(Path(__file__).parents[2] / "examples/on_policy_distillation_trainer")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="full_vocab_adaptive_opd")
    return cfg, tokenizer


def test_example_composes_and_instantiates_typed_configs(monkeypatch, tmp_path):
    cfg, tokenizer = configure(monkeypatch, tmp_path)
    typed = instantiate(cfg.distillation, _convert_="partial")
    teacher = typed.teacher_models["default"]
    assert teacher.inference.name == "vllm"
    assert teacher.inference.engine_kwargs["vllm"]["max_logprobs"] == -1
    assert teacher.inference.logprobs_mode == "raw_logprobs"
    assert teacher.inference.max_num_batched_tokens == 128
    assert typed.nnodes == 2 and typed.n_gpus_per_node == 5
    assert typed.distillation_loss.loss_settings.use_full_vocab
    assert cfg.distillation.full_vocab_opd.alpha_max == typed.full_vocab_opd.alpha_max == 0.8
    assert not hasattr(typed, "teacher_engine") and not hasattr(typed, "teacher_model_config")
    validate_student(cfg, tokenizer)
    cfg.distillation.distillation_loss.use_task_rewards = True
    cfg.actor_rollout_ref.rollout.temperature = 2.0
    with pytest.raises(ValueError, match="matching.*temperatures"):
        validate_student(cfg, tokenizer)


def test_full_vocab_rejects_policy_gradient():
    # Loss config may be created before enclosing DistillationConfig; the latter
    # validates the execution mode and direct-autograd requirement.
    cfg = DistillationLossConfig(
        loss_mode="coverage_adaptive_full_vocab",
        use_policy_gradient=False,
        log_prob_min_clamp=None,
        loss_max_clamp=None,
    )
    assert cfg.loss_settings.use_full_vocab and not cfg.loss_settings.use_topk


@pytest.mark.parametrize("alpha_max", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_full_vocab_rejects_invalid_alpha_max(alpha_max):
    with pytest.raises(ValueError, match="alpha_max"):
        FullVocabularyOPDConfig(alpha_max=alpha_max)
