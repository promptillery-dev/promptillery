"""Shared GSM8K-style final-answer extraction (last number, comma-stripped)."""
import pytest

from promptillery.utils import extract_final_number


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is 42. #### 72", "72"),
        ("#### 1,234", "1234"),
        ("She pays $-3.50 so #### -3.5", "-3.5"),
        ("no numbers here", None),
        ("", None),
        (72, "72"),  # non-str input is coerced
    ],
)
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


def test_trainer_number_mode_uses_shared_extraction():
    """_normalize_answer('number') must agree with extract_final_number."""
    from promptillery.trainers.causal_lm_sft_trainer import CausalLMSFTTrainer

    trainer = CausalLMSFTTrainer.__new__(CausalLMSFTTrainer)
    trainer.trainer_config = {"answer_extraction": "number"}
    assert trainer._normalize_answer("Total: 1,234 so #### 72") == "72"
    # fallback when no number: unchanged legacy behavior (lowercased text)
    assert trainer._normalize_answer("No Idea") == "no idea"
