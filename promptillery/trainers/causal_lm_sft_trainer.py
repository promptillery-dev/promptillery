"""Causal language model SFT trainer for pre-materialized instruction data."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
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

from ..reproducibility import model_revision_kwargs, tokenizer_revision_kwargs
from .base import BaseTrainer, PredictionResult

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
        self.explicit_text_field = "text_field" in self.trainer_config
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
        self._canonical_labels_cache: List[str] | None = None
        self._generation_eval_cache: Dict[tuple[int, int, str, int | None], Any] = {}
        self._generation_eval_version = 0
        tokenizer_kwargs = tokenizer_revision_kwargs(self.cfg)
        model_kwargs = model_revision_kwargs(self.cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.student,
            trust_remote_code=self.trust_remote_code,
            **tokenizer_kwargs,
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
                **model_kwargs,
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
        has_prompt_response = (
            self.prompt_field in examples and self.response_field in examples
        )
        if has_prompt_response and (
            not self.explicit_text_field or self.text_field == self.prompt_field
        ):
            return [
                self._format_prompt_response(str(prompt), str(response))
                for prompt, response in zip(
                    examples[self.prompt_field],
                    examples[self.response_field],
                )
            ]

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

        if has_prompt_response:
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

    def _format_generation_prompt(self, prompt: str) -> str:
        """Format an inference prompt with the same boundary used for SFT."""
        template = self.trainer_config.get("generation_prompt_template")
        if template:
            return str(template).format(prompt=prompt)
        train_template = self.trainer_config.get("format_template")
        if train_template:
            return str(train_template).format(prompt=prompt, response="")
        return f"### Instruction:\n{prompt}\n\n### Response:\n"

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
        self._generation_eval_version += 1
        self._generation_eval_cache = {}
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
        if mode == "canonical_label":
            value = re.sub(r"\s+", " ", value.lower()).strip(" .,:;")
            return re.sub(r"[^a-z0-9]+", "_", value).strip("_")
        return re.sub(r"\s+", " ", value.lower()).strip(" .,:;")

    def _normalized_unique_labels(self, labels: List[Any]) -> List[str]:
        """Normalize labels while preserving first-seen order."""
        seen = set()
        normalized = []
        for label in labels:
            label_text = self._normalize_answer(str(label))
            if label_text and label_text not in seen:
                seen.add(label_text)
                normalized.append(label_text)
        return normalized

    def _resolve_canonical_labels_path(self) -> Path | None:
        """Resolve an optional canonical label artifact path."""
        configured = self.trainer_config.get("canonical_labels_path")
        if not configured:
            return None
        path = Path(str(configured)).expanduser()
        if path.is_absolute():
            return path

        candidates = [Path.cwd() / path, self.out_dir / path]
        base_output_dir = getattr(self.cfg, "base_output_dir", None)
        if base_output_dir:
            candidates.append(Path(str(base_output_dir)) / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _canonical_labels(self) -> List[str]:
        """Return normalized task labels for generative classification metrics."""
        cached = getattr(self, "_canonical_labels_cache", None)
        if cached is not None:
            return list(cached)

        configured_labels = self.trainer_config.get("canonical_labels")
        if configured_labels:
            labels = self._normalized_unique_labels(list(configured_labels))
            self._canonical_labels_cache = labels
            return list(labels)

        labels_path = self._resolve_canonical_labels_path()
        if labels_path is not None:
            if not labels_path.exists():
                raise FileNotFoundError(
                    f"canonical_labels_path does not exist: {labels_path}"
                )
            with labels_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            labels = payload.get("normalized_canonical_labels") or payload.get(
                "canonical_labels"
            )
            if not labels:
                raise ValueError(
                    "canonical_labels_path must contain canonical_labels or "
                    "normalized_canonical_labels"
                )
            labels = self._normalized_unique_labels(list(labels))
            self._canonical_labels_cache = labels
            return list(labels)

        observed_labels: List[Any] = []
        for split in self.dataset.values():
            if self.gold_answer_field in split.column_names:
                observed_labels.extend(split[self.gold_answer_field])
        labels = sorted(self._normalized_unique_labels(observed_labels))
        if labels and len(labels) < getattr(self.cfg, "num_classes", 0):
            logger.warning(
                "Inferred only %s canonical labels from loaded splits; expected %s",
                len(labels),
                self.cfg.num_classes,
            )
        self._canonical_labels_cache = labels
        return list(labels)

    def _generation_sample_limit(self, *, detailed: bool = False) -> int | None:
        """Return the configured cap for generation-based evaluation."""
        if detailed:
            limit = self.trainer_config.get("max_detailed_prediction_samples")
            if limit is not None:
                return int(limit)
        limit = self.trainer_config.get("max_eval_generation_samples")
        return int(limit) if limit is not None else None

    def _generation_cache(self) -> Dict[tuple[int, int, str, int | None], Any]:
        """Return a lazily initialized generation-eval cache."""
        if not hasattr(self, "_generation_eval_cache"):
            self._generation_eval_cache = {}
        return self._generation_eval_cache

    def _limited_generation_dataset(
        self,
        split: str,
        sample_limit: int | None,
    ) -> tuple[Dataset, List[int]]:
        """Return a capped eval dataset plus original row indices."""
        ds = self.dataset[split]
        indices = list(range(len(ds)))
        if sample_limit is None or sample_limit >= len(ds):
            return ds, indices
        if sample_limit <= 0:
            return ds.select([]), []

        canonical_labels = (
            self._canonical_labels()
            if self.trainer_config.get("answer_extraction") == "canonical_label"
            else []
        )
        if not canonical_labels or self.gold_answer_field not in ds.column_names:
            selected = indices[:sample_limit]
            return ds.select(selected), selected

        label_order = {label: position for position, label in enumerate(canonical_labels)}
        grouped: Dict[str, List[int]] = {}
        for index, gold in enumerate(ds[self.gold_answer_field]):
            normalized = self._normalize_answer(str(gold))
            grouped.setdefault(normalized, []).append(index)

        selected: List[int] = []
        labels = sorted(
            grouped,
            key=lambda label: (label_order.get(label, len(label_order)), label),
        )
        offset = 0
        while len(selected) < sample_limit:
            progressed = False
            for label in labels:
                label_indices = grouped[label]
                if offset < len(label_indices):
                    selected.append(label_indices[offset])
                    progressed = True
                    if len(selected) >= sample_limit:
                        break
            if not progressed:
                break
            offset += 1

        return ds.select(selected), selected

    def _generation_token_stats(
        self,
        generated: Any,
        output_ids: torch.Tensor,
        row_offset: int,
        input_width: int,
    ) -> tuple[float, float]:
        """Estimate completion confidence and entropy from generated-token scores."""
        scores = getattr(generated, "scores", None) or []
        token_log_probs = []
        token_entropies = []
        for step, logits in enumerate(scores):
            token_pos = input_width + step
            if token_pos >= output_ids.shape[0]:
                break
            token_id = int(output_ids[token_pos].item())
            probs = torch.softmax(logits[row_offset].float(), dim=-1)
            token_prob = float(probs[token_id].clamp_min(1e-12).item())
            token_log_probs.append(math.log(token_prob))
            token_entropies.append(
                float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
            )
            if token_id == self.tokenizer.eos_token_id:
                break

        if not token_log_probs:
            return 1.0, 0.0
        confidence = math.exp(sum(token_log_probs) / len(token_log_probs))
        entropy = sum(token_entropies) / len(token_entropies)
        return float(confidence), float(entropy)

    def _score_canonical_label_records(
        self,
        trainer: Trainer,
        split: str,
        sample_limit: int | None,
    ) -> List[Dict[str, Any]]:
        """Classify by scoring each canonical label under the causal LM."""
        if split not in self.dataset:
            return []
        ds = self.dataset[split]
        if (
            self.prompt_field not in ds.column_names
            or self.gold_answer_field not in ds.column_names
        ):
            return []
        ds, original_indices = self._limited_generation_dataset(split, sample_limit)

        canonical_labels = self._canonical_labels()
        if not canonical_labels:
            return []

        label_batch_size = int(
            self.trainer_config.get("canonical_label_score_batch_size", 16)
        )
        records: List[Dict[str, Any]] = []
        model = trainer.model
        model.eval()
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device

        for index, row in zip(original_indices, ds):
            prompt = self._format_generation_prompt(str(row[self.prompt_field]))
            prompt_token_count = len(
                self.tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )["input_ids"]
            )
            label_losses: List[float] = []
            label_token_counts: List[int] = []
            original_padding_side = self.tokenizer.padding_side
            try:
                self.tokenizer.padding_side = "right"
                for start in range(0, len(canonical_labels), label_batch_size):
                    labels = canonical_labels[start : start + label_batch_size]
                    texts = [prompt + label for label in labels]
                    encoded = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_seq_length,
                    ).to(device)
                    with torch.no_grad():
                        logits = model(**encoded).logits.float()
                    input_ids = encoded["input_ids"]
                    attention_mask = encoded["attention_mask"]
                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]
                    shift_attention = attention_mask[:, 1:].bool()
                    positions = torch.arange(
                        shift_labels.shape[1], device=shift_labels.device
                    ).unsqueeze(0)
                    label_mask = positions >= max(0, prompt_token_count - 1)
                    label_mask = label_mask & shift_attention
                    token_losses = torch.nn.functional.cross_entropy(
                        shift_logits.transpose(1, 2),
                        shift_labels,
                        reduction="none",
                    )
                    counts = label_mask.sum(dim=1).clamp_min(1)
                    losses = (token_losses * label_mask).sum(dim=1) / counts
                    label_losses.extend(float(value) for value in losses.cpu())
                    label_token_counts.extend(int(value) for value in counts.cpu())
            finally:
                self.tokenizer.padding_side = original_padding_side

            scores = torch.tensor([-loss for loss in label_losses], dtype=torch.float)
            probabilities = torch.softmax(scores, dim=0)
            best_index = int(torch.argmax(probabilities).item())
            completion = canonical_labels[best_index]
            confidence = float(probabilities[best_index].item())
            entropy = float(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
            )
            gold = str(row[self.gold_answer_field])
            pred_norm = self._normalize_answer(completion)
            gold_norm = self._normalize_answer(gold)
            records.append(
                {
                    "split": split,
                    "index": index,
                    "prompt": prompt,
                    "completion": completion,
                    "gold_answer": gold,
                    "normalized_prediction": pred_norm,
                    "normalized_gold": gold_norm,
                    "exact_match": pred_norm == gold_norm,
                    "generation_confidence": confidence,
                    "generation_entropy": entropy,
                    "canonical_label_decoding": "score",
                    "canonical_label_token_count": label_token_counts[best_index],
                }
            )
        return records

    def _generate_eval_records(
        self,
        trainer: Trainer,
        split: str,
        sample_limit: int | None,
    ) -> List[Dict[str, Any]]:
        """Generate deterministic completions and return structured records."""
        if (
            self.trainer_config.get("answer_extraction") == "canonical_label"
            and self.trainer_config.get("canonical_label_decoding", "score")
            == "score"
        ):
            return self._score_canonical_label_records(trainer, split, sample_limit)

        if split not in self.dataset:
            return []
        ds = self.dataset[split]
        if (
            self.prompt_field not in ds.column_names
            or self.gold_answer_field not in ds.column_names
        ):
            return []
        ds, original_indices = self._limited_generation_dataset(split, sample_limit)

        max_new_tokens = int(self.trainer_config.get("generation_max_new_tokens", 32))
        batch_size = int(
            self.trainer_config.get("generation_batch_size", self.cfg.batch_size)
        )
        records: List[Dict[str, Any]] = []
        model = trainer.model
        model.eval()

        for start in range(0, len(ds), batch_size):
            batch = ds.select(range(start, min(start + batch_size, len(ds))))
            raw_prompts = [str(value) for value in batch[self.prompt_field]]
            prompts = [self._format_generation_prompt(prompt) for prompt in raw_prompts]
            original_padding_side = self.tokenizer.padding_side
            try:
                self.tokenizer.padding_side = self.trainer_config.get(
                    "generation_padding_side", "left"
                )
                encoded = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                device = getattr(model, "device", None)
                if device is None:
                    device = next(model.parameters()).device
                encoded = encoded.to(device)
                input_width = int(encoded["input_ids"].shape[1])
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
            finally:
                self.tokenizer.padding_side = original_padding_side

            for offset, output_ids in enumerate(generated.sequences):
                original_index = original_indices[start + offset]
                completion_ids = output_ids[input_width:]
                completion = self.tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True,
                ).strip()
                gold = str(batch[self.gold_answer_field][offset])
                pred_norm = self._normalize_answer(completion)
                gold_norm = self._normalize_answer(gold)
                confidence, entropy = self._generation_token_stats(
                    generated,
                    output_ids,
                    offset,
                    input_width,
                )
                records.append(
                    {
                        "split": split,
                        "index": original_index,
                        "prompt": prompts[offset],
                        "completion": completion,
                        "gold_answer": gold,
                        "normalized_prediction": pred_norm,
                        "normalized_gold": gold_norm,
                        "exact_match": pred_norm == gold_norm,
                        "generation_confidence": confidence,
                        "generation_entropy": entropy,
                    }
                )
        return records

    def _generation_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize generation records with canonical-label validity metrics."""
        if not records:
            return {}

        normalized_preds = [
            str(record["normalized_prediction"]) for record in records
        ]
        normalized_golds = [str(record["normalized_gold"]) for record in records]
        canonical_labels = self._canonical_labels()
        canonical_set = set(canonical_labels)
        observed_gold_labels = sorted(set(normalized_golds))
        exact_match = (
            sum(record["exact_match"] for record in records) / len(records)
        )
        summary: Dict[str, Any] = {
            "split": records[0]["split"],
            "examples": len(records),
            "exact_match": float(exact_match),
            "answer_extraction": self.trainer_config.get(
                "answer_extraction", "text"
            ),
        }

        metric_labels = sorted(set(normalized_golds) | set(normalized_preds))
        if canonical_labels:
            invalid = [
                prediction not in canonical_set for prediction in normalized_preds
            ]
            for record, is_invalid in zip(records, invalid):
                record["prediction_is_valid_label"] = not is_invalid
            observed_canonical_golds = [
                label for label in canonical_labels if label in set(normalized_golds)
            ]
            metric_labels = observed_canonical_golds or canonical_labels
            summary.update(
                {
                    "canonical_label_count": len(canonical_labels),
                    "observed_gold_label_count": len(observed_canonical_golds),
                    "invalid_label_rate": float(sum(invalid) / len(invalid)),
                    "normalized_canonical_labels": canonical_labels,
                }
            )
        elif observed_gold_labels:
            summary["observed_gold_label_count"] = len(observed_gold_labels)

        if self.trainer_config.get("task_metric") == "macro_f1":
            summary["macro_f1"] = float(
                f1_score(
                    normalized_golds,
                    normalized_preds,
                    labels=metric_labels,
                    average="macro",
                    zero_division=0,
                )
            )
            if canonical_labels:
                summary["macro_f1_full_canonical"] = float(
                    f1_score(
                        normalized_golds,
                        normalized_preds,
                        labels=canonical_labels,
                        average="macro",
                        zero_division=0,
                    )
                )

        return summary

    def _write_generation_eval(
        self,
        records: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> None:
        """Persist generation predictions and metric summary."""
        split = str(summary.get("split") or "eval")
        predictions_path = self.out_dir / "eval_predictions.jsonl"
        summary_path = self.out_dir / "exact_metric_summary.json"
        split_predictions_path = self.out_dir / f"eval_predictions_{split}.jsonl"
        split_summary_path = self.out_dir / f"exact_metric_summary_{split}.json"
        prediction_paths = {split_predictions_path}
        summary_paths = {split_summary_path}
        if split == (self._eval_split() or split):
            prediction_paths.add(predictions_path)
            summary_paths.add(summary_path)
        for path in prediction_paths:
            with path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
        for path in summary_paths:
            with path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

    def _run_generation_eval(
        self,
        trainer: Trainer,
        split: str,
        sample_limit: int | None,
        *,
        write_artifacts: bool,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run or reuse generation evaluation for a model/split/sample cap."""
        model = getattr(trainer, "model", trainer)
        version = int(getattr(self, "_generation_eval_version", 0))
        key = (version, id(model), split, sample_limit)
        cache = self._generation_cache()
        if key not in cache:
            records = self._generate_eval_records(trainer, split, sample_limit)
            summary = self._generation_summary(records)
            cache[key] = (records, summary)
        records, summary = cache[key]
        if write_artifacts and records:
            self._write_generation_eval(records, summary)
        return records, summary

    def _evaluate_generation(self, trainer: Trainer, split: str) -> Dict[str, Any]:
        """Run deterministic generation metrics when prompt/gold fields exist."""
        _, summary = self._run_generation_eval(
            trainer,
            split,
            self._generation_sample_limit(detailed=False),
            write_artifacts=True,
        )
        return {
            key: value
            for key, value in summary.items()
            if isinstance(value, (int, float))
        }

    def _prediction_result_from_generation(
        self,
        records: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> PredictionResult:
        """Convert generation records into policy-compatible predictions."""
        canonical_labels = list(summary.get("normalized_canonical_labels") or [])
        observed_labels = sorted(
            {
                str(record["normalized_gold"])
                for record in records
            }
            | {
                str(record["normalized_prediction"])
                for record in records
            }
        )
        label_names = list(canonical_labels or observed_labels)
        label_to_id = {label: index for index, label in enumerate(label_names)}

        def encode_true(label: str) -> int:
            if label not in label_to_id:
                label_to_id[label] = len(label_names)
                label_names.append(label)
            return label_to_id[label]

        def encode_predicted(label: str) -> int:
            if label in label_to_id:
                return label_to_id[label]
            if canonical_labels:
                return len(label_names)
            label_to_id[label] = len(label_names)
            label_names.append(label)
            return label_to_id[label]

        predicted_texts = [
            str(record["normalized_prediction"]) for record in records
        ]
        true_texts = [str(record["normalized_gold"]) for record in records]
        true_label_ids = [encode_true(label) for label in true_texts]
        predicted_label_ids = [encode_predicted(label) for label in predicted_texts]
        metadata = dict(summary)
        metadata["canonical_labels"] = canonical_labels
        return PredictionResult(
            indices=[int(record["index"]) for record in records],
            predicted_labels=predicted_label_ids,
            true_labels=true_label_ids,
            confidences=[
                float(record.get("generation_confidence", 0.0))
                for record in records
            ],
            entropies=[
                float(record.get("generation_entropy", 0.0)) for record in records
            ],
            label_names=label_names,
            predicted_texts=predicted_texts,
            true_texts=true_texts,
            metadata=metadata,
        )

    def get_detailed_predictions(
        self,
        trainer: Trainer,
        split: str = "train",
    ) -> PredictionResult:
        """Get generation-based predictions for policy prompts and features."""
        records, summary = self._run_generation_eval(
            trainer,
            split,
            self._generation_sample_limit(detailed=True),
            write_artifacts=False,
        )
        if not records:
            return PredictionResult(
                indices=[],
                predicted_labels=[],
                true_labels=[],
                confidences=[],
                entropies=None,
            )
        return self._prediction_result_from_generation(records, summary)

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
