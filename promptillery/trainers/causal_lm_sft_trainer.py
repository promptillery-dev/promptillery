"""Causal language model SFT trainer for pre-materialized instruction data."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .base import BaseTrainer

logger = logging.getLogger(__name__)


class CausalLMSFTTrainer(BaseTrainer):
    """Trainer for small causal LMs on SFT-style text records.

    The trainer expects either a preformatted text field or prompt/response
    fields. This keeps teacher-data generation auditable and separate from
    student fine-tuning for the first LLM-to-LLM experiments.
    """

    def __init__(self, config, dataset, out_dir):
        super().__init__(config, dataset, out_dir)
        self.trainer_config = self.cfg.trainer_config or {}
        self.max_seq_length = int(self.trainer_config.get("max_seq_length", 512))
        self.text_field = self.trainer_config.get("text_field", self.cfg.text_field)
        self.prompt_field = self.trainer_config.get("prompt_field", "student_prompt")
        self.response_field = self.trainer_config.get(
            "response_field", "teacher_response"
        )
        self.gold_answer_field = self.trainer_config.get("gold_answer_field", "gold_answer")
        self.add_eos_token = bool(self.trainer_config.get("add_eos_token", True))
        self.trust_remote_code = bool(
            self.trainer_config.get("trust_remote_code", False)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.student,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = self.trainer_config.get("padding_side", "right")

        if self.trainer_config.get("model_from_config", False):
            model_config = self._build_model_config()
            self.model = AutoModelForCausalLM.from_config(model_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.student,
                trust_remote_code=self.trust_remote_code,
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.trainer_config.get("use_lora", False):
            self.model = self._apply_lora(self.model)

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def _build_model_config(self):
        """Build a tiny/random model config for offline smoke tests."""
        model_type = self.trainer_config.get("model_type", "gpt2")
        config_kwargs = dict(self.trainer_config.get("model_config", {}))
        config_kwargs.setdefault("vocab_size", len(self.tokenizer))
        config_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            config_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        return AutoConfig.for_model(model_type, **config_kwargs)

    def _apply_lora(self, model):
        """Wrap the base model with PEFT LoRA adapters."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "trainer_config.use_lora=true requires the `peft` package. "
                "Install promptillery with updated dependencies or set use_lora=false."
            ) from exc

        target_modules = self.trainer_config.get("lora_target_modules", "all-linear")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(self.trainer_config.get("lora_r", 8)),
            lora_alpha=int(self.trainer_config.get("lora_alpha", 16)),
            lora_dropout=float(self.trainer_config.get("lora_dropout", 0.05)),
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def _format_prompt_response(self, prompt: str, response: str) -> str:
        """Format one prompt/response pair as supervised text."""
        template = self.trainer_config.get("format_template")
        if template:
            text = template.format(prompt=prompt, response=response)
        else:
            text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        if self.add_eos_token and self.tokenizer.eos_token:
            text += self.tokenizer.eos_token
        return text

    def _build_texts(self, examples: Dict[str, List[Any]]) -> List[str]:
        """Build SFT text strings from a batched dataset map call."""
        if self.text_field in examples:
            texts = [str(value) for value in examples[self.text_field]]
            if self.add_eos_token and self.tokenizer.eos_token:
                return [
                    text
                    if text.endswith(self.tokenizer.eos_token)
                    else text + self.tokenizer.eos_token
                    for text in texts
                ]
            return texts

        if self.prompt_field in examples and self.response_field in examples:
            return [
                self._format_prompt_response(str(prompt), str(response))
                for prompt, response in zip(
                    examples[self.prompt_field],
                    examples[self.response_field],
                )
            ]

        if "instruction" in examples and "output" in examples:
            inputs = examples.get("input") or [""] * len(examples["instruction"])
            return [
                self._format_prompt_response(
                    "\n".join(part for part in [str(instruction), str(input_)] if part),
                    str(output),
                )
                for instruction, input_, output in zip(
                    examples["instruction"],
                    inputs,
                    examples["output"],
                )
            ]

        available = ", ".join(sorted(examples.keys()))
        raise ValueError(
            "CausalLMSFTTrainer requires either a text field, prompt/response "
            f"fields, or instruction/output fields. Available columns: {available}"
        )

    def prepare_data(self, split: str) -> Dataset:
        """Tokenize SFT data for causal language modeling."""
        ds = self.dataset[split]

        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
            texts = self._build_texts(examples)
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )

        return ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    def _eval_split(self) -> str | None:
        """Return the preferred eval split for SFT training."""
        if "validation" in self.dataset:
            return "validation"
        if "test" in self.dataset:
            return "test"
        return None

    def train(self) -> Trainer:
        """Fine-tune the causal language model."""
        eval_split = self._eval_split()
        has_eval = eval_split is not None
        args = TrainingArguments(
            output_dir=str(self.out_dir / "training"),
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            warmup_steps=self.cfg.warmup_steps,
            weight_decay=self.cfg.weight_decay,
            learning_rate=self.cfg.learning_rate,
            logging_dir=str(self.out_dir / "logs"),
            logging_steps=int(self.trainer_config.get("logging_steps", 25)),
            eval_strategy="epoch" if has_eval else "no",
            save_strategy="epoch" if has_eval else "no",
            load_best_model_at_end=has_eval,
            metric_for_best_model="eval_loss" if has_eval else None,
            greater_is_better=False if has_eval else None,
            report_to=self.trainer_config.get("report_to", ["tensorboard"]),
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.prepare_data("train"),
            eval_dataset=self.prepare_data(eval_split) if has_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        trainer.train()
        return trainer

    def evaluate(self, trainer: Trainer, split: str = "test") -> Dict[str, Any]:
        """Evaluate SFT loss/perplexity on the requested split."""
        if split not in self.dataset:
            if self.cfg.paper_mode or self.trainer_config.get(
                "require_requested_eval_split", False
            ):
                available = ", ".join(self.dataset.keys())
                raise ValueError(
                    f"Requested split {split} is unavailable. Available: {available}"
                )
            fallback = self._eval_split() or "train"
            logger.warning(
                "Requested split %s is unavailable; evaluating on %s",
                split,
                fallback,
            )
            split = fallback

        metrics = trainer.evaluate(eval_dataset=self.prepare_data(split))
        eval_loss = float(metrics.get("eval_loss", metrics.get("loss", 0.0)))
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        return {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            **self._evaluate_generation(trainer, split),
        }

    def _normalize_answer(self, value: str) -> str:
        """Normalize generated and gold answers for deterministic metrics."""
        value = str(value).strip()
        mode = self.trainer_config.get("answer_extraction", "text")
        if mode == "number":
            matches = re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
            return matches[-1] if matches else value.lower()
        if mode == "choice":
            match = re.search(r"\b([A-E])\b", value.upper())
            return match.group(1) if match else value.strip().upper()
        return re.sub(r"\s+", " ", value.lower()).strip(" .,:;")

    def _evaluate_generation(self, trainer: Trainer, split: str) -> Dict[str, Any]:
        """Run deterministic generation metrics when prompt/gold fields exist."""
        ds = self.dataset[split]
        if (
            self.prompt_field not in ds.column_names
            or self.gold_answer_field not in ds.column_names
        ):
            return {}

        max_samples = self.trainer_config.get("max_eval_generation_samples")
        if max_samples is not None:
            ds = ds.select(range(min(int(max_samples), len(ds))))

        max_new_tokens = int(self.trainer_config.get("generation_max_new_tokens", 32))
        batch_size = int(self.trainer_config.get("generation_batch_size", self.cfg.batch_size))
        predictions_path = self.out_dir / "eval_predictions.jsonl"
        summary_path = self.out_dir / "exact_metric_summary.json"

        records: List[Dict[str, Any]] = []
        normalized_preds: List[str] = []
        normalized_golds: List[str] = []
        model = trainer.model
        model.eval()

        for start in range(0, len(ds), batch_size):
            batch = ds.select(range(start, min(start + batch_size, len(ds))))
            prompts = [str(value) for value in batch[self.prompt_field]]
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            ).to(model.device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for offset, output_ids in enumerate(generated):
                prompt_len = int(encoded["attention_mask"][offset].sum().item())
                completion_ids = output_ids[prompt_len:]
                completion = self.tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True,
                ).strip()
                gold = str(batch[self.gold_answer_field][offset])
                pred_norm = self._normalize_answer(completion)
                gold_norm = self._normalize_answer(gold)
                normalized_preds.append(pred_norm)
                normalized_golds.append(gold_norm)
                records.append(
                    {
                        "split": split,
                        "index": start + offset,
                        "prompt": prompts[offset],
                        "completion": completion,
                        "gold_answer": gold,
                        "normalized_prediction": pred_norm,
                        "normalized_gold": gold_norm,
                        "exact_match": pred_norm == gold_norm,
                    }
                )

        exact_match = (
            sum(record["exact_match"] for record in records) / len(records)
            if records
            else 0.0
        )
        summary: Dict[str, Any] = {
            "split": split,
            "examples": len(records),
            "exact_match": exact_match,
            "answer_extraction": self.trainer_config.get("answer_extraction", "text"),
        }
        if self.trainer_config.get("task_metric") == "macro_f1" and records:
            labels = sorted(set(normalized_golds) | set(normalized_preds))
            summary["macro_f1"] = f1_score(
                normalized_golds,
                normalized_preds,
                labels=labels,
                average="macro",
                zero_division=0,
            )

        with open(predictions_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        return {
            key: value
            for key, value in summary.items()
            if isinstance(value, (int, float))
        }

    def predict_for_augmentation(self, model: Trainer, split: str = "train") -> List[int]:
        """Causal-LM SFT does not expose classifier errors for augmentation."""
        return []

    def save_model(self, trainer: Trainer) -> None:
        """Save the SFT model and tokenizer."""
        trainer.save_model(str(self.out_dir / "model"))
        self.tokenizer.save_pretrained(str(self.out_dir / "model"))

    def load_model(self, model_path) -> Trainer:
        """Load a saved SFT model and return an eval-only Trainer."""
        model_path = Path(model_path)
        model_dir = model_path / "model" if (model_path / "model").exists() else model_path
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=self.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=self.trust_remote_code,
        )
        args = TrainingArguments(
            output_dir=str(model_path / "eval_output"),
            per_device_eval_batch_size=self.cfg.batch_size,
            report_to=[],
        )
        return Trainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            ),
        )

    def push_to_hub(self, model: Any, repo_name: str) -> None:
        """Push model to HuggingFace Hub if supported."""
        self.model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)
