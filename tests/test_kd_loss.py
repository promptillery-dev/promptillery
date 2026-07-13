"""Token-level forward-KL distillation loss: invariants before wiring."""
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from promptillery.trainers.causal_lm_sft_trainer import (
    _assert_kd_tokenizer_match,
    _assert_kd_vocab_size_match,
    kd_loss,
)


def _batch(seed=0, batch=2, seq=6, vocab=11):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=g)
    labels = torch.randint(0, vocab, (batch, seq), generator=g)
    labels[:, :3] = -100  # prompt mask
    return logits, labels


def test_kl_of_identical_logits_is_zero():
    logits, labels = _batch()
    loss = kd_loss(logits, logits.clone(), labels)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_ce_weight_one_equals_cross_entropy():
    student, labels = _batch(seed=1)
    teacher, _ = _batch(seed=2)
    loss = kd_loss(student, teacher, labels, ce_weight=1.0)
    expected = F.cross_entropy(
        student[:, :-1].reshape(-1, student.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.isclose(loss, expected, atol=1e-6)


def test_masked_positions_do_not_contribute():
    student, labels = _batch(seed=3)
    teacher, _ = _batch(seed=4)
    base = kd_loss(student, teacher, labels)
    corrupted = teacher.clone()
    corrupted[:, :2, :] += 100.0  # positions whose TARGETS (shift) are masked
    assert torch.isclose(base, kd_loss(student, corrupted, labels), atol=1e-5)


def test_kd_loss_matches_hand_computed_forward_kl():
    """Pins the KL direction: sum p_teacher * (log p_teacher - log p_student)."""
    student, labels = _batch(seed=7)
    teacher, _ = _batch(seed=8)
    loss = kd_loss(student, teacher, labels)

    log_ps = F.log_softmax(student[:, :-1], dim=-1)
    log_pt = F.log_softmax(teacher[:, :-1], dim=-1)
    target = labels[:, 1:]
    mask = target.ne(-100)
    per_token = (log_pt.exp() * (log_pt - log_ps)).sum(-1)
    expected = (per_token * mask).sum() / mask.sum()
    assert torch.isclose(loss, expected, atol=1e-6)


def test_temperature_scales_the_kl_term():
    student, labels = _batch(seed=5)
    teacher, _ = _batch(seed=6)
    assert not torch.isclose(
        kd_loss(student, teacher, labels, temperature=1.0),
        kd_loss(student, teacher, labels, temperature=2.0),
    )


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_tokenizer_match_passes_on_identical_vocab():
    _assert_kd_tokenizer_match(_FakeTokenizer({"a": 0}), _FakeTokenizer({"a": 0}))


def test_tokenizer_mismatch_raises():
    with pytest.raises(ValueError, match="tokenizer"):
        _assert_kd_tokenizer_match(_FakeTokenizer({"a": 0}), _FakeTokenizer({"b": 0}))


def _fake_model(vocab_size):
    return types.SimpleNamespace(config=SimpleNamespace(vocab_size=vocab_size))


def test_vocab_size_match_passes_on_equal_sizes():
    _assert_kd_vocab_size_match(_fake_model(32000), _fake_model(32000))


def test_vocab_size_mismatch_raises():
    with pytest.raises(ValueError, match="vocab_size"):
        _assert_kd_vocab_size_match(_fake_model(32000), _fake_model(32064))
