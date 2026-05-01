"""Co-training cycle orchestrator (design §2).

Loop per cycle t:
  Train  : LoRA_A on D_A^t, LoRA_B on D_B^t
  Eval   : both on V → metrics + features
  Probe  : run B on bootstrap_A, A on bootstrap_B → fault seeds
  Generate: A produces variants from B-fault seeds; symmetric for B
  Validate: peer agreement → strong-teacher arbitration
  Merge  : accepted A→B variants enter D_B^{t+1}; B→A → D_A^{t+1}
  Control: features → controller → next action or STOP
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from datasets import Dataset, concatenate_datasets

from ..config import ExperimentConfig
from ..token_tracker import TokenTracker
from .actions import CoTrainAction, enumerate_cotrain_actions
from .arbitration import ArbitrationResult, StrongTeacherArbiter
from .bootstrap import BootstrapPartition, partition_bootstrap
from .controller_linearbai import LinearBAIController
from .controller_p import FrugalKDCoTrainP
from .dual_trainer import DualStudentTrainer
from .features import StudentEvalSummary, build_cotrain_features
from .flow import allocate_volumes
from .provenance import AuditLedger
from .validation import ValidationPipeline, VariantUnderReview
from .variant_generator import GenerationRequest, VariantGenerator

logger = logging.getLogger(__name__)


def _budget_state_of(tracker: TokenTracker) -> Dict[str, Any]:
    """Expose the minimum surface build_cotrain_features needs.

    When no token_budget is configured, report a fully-unused budget so the
    controller's stop-on-budget-exhaustion check stays inert.
    """
    if tracker.token_budget is None:
        return {"token_budget": 1, "tokens_remaining": 1}
    used = tracker.summary.grand_total.total_tokens
    return {
        "token_budget": tracker.token_budget,
        "tokens_remaining": max(0, tracker.token_budget - used),
    }


class _LiteLLMChatClient:
    """Adapter from a LiteLLM-shaped completion callable to ChatClient."""

    def __init__(self, complete_fn: Callable[..., Awaitable[Dict[str, Any]]]):
        self._complete = complete_fn

    async def complete(self, *, model, messages, temperature, max_tokens):
        return await self._complete(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )


class _StudentPeerLabeller:
    """Wrap a DualStudentTrainer + side identity to expose the PeerLabeller API."""

    def __init__(self, dual: DualStudentTrainer, side: str, label_options: List[str]):
        self.dual = dual
        self.side = side
        self.label_options = label_options

    async def label(self, text: str) -> tuple[str, float]:
        trainer = self.dual.trainer_a if self.side == "a" else self.dual.trainer_b
        model = self.dual.model_a if self.side == "a" else self.dual.model_b
        placeholder_label = self.label_options[0] if self.label_options else "label"
        rows = [{"text": text, "label": placeholder_label}]
        ds_split = Dataset.from_list(rows)
        original = trainer.dataset.get("__cotrain_peer__")
        trainer.dataset["__cotrain_peer__"] = ds_split
        try:
            preds = trainer.get_detailed_predictions(model, split="__cotrain_peer__")
        finally:
            if original is None:
                trainer.dataset.pop("__cotrain_peer__", None)
            else:
                trainer.dataset["__cotrain_peer__"] = original
        if not preds.predicted_labels:
            return placeholder_label, 0.0
        return str(preds.predicted_labels[0]), float(preds.confidences[0] or 0.0)


class CoTrainEngine:
    def __init__(
        self,
        config: ExperimentConfig,
        dataset: Dict[str, Dataset],
        out_dir: Path,
        *,
        generate_fn_a: Callable[..., List[str]],
        generate_fn_b: Callable[..., List[str]],
        arbiter_complete: Callable[..., Awaitable[Dict[str, Any]]],
        force_stop_after_cycle: Optional[int] = None,
    ) -> None:
        if config.cotrain is None:
            raise ValueError("CoTrainEngine requires acquisition_mode='cotrain'")
        self.config = config
        self.cot = config.cotrain
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = f"{config.name}-{int(time.time())}"
        self.force_stop_after_cycle = force_stop_after_cycle

        ds_cfg = config.get_dataset_config_obj()
        self.text_field = ds_cfg.text_field if ds_cfg else "text"
        if self.cot.task_kind == "classification":
            self.label_field = ds_cfg.label_field if ds_cfg else "label"
        else:
            self.label_field = None

        train_split = dataset["train"]
        if "id" not in train_split.column_names:
            train_split = train_split.add_column(
                "id", [f"row{i}" for i in range(len(train_split))]
            )
            dataset = dict(dataset)
            dataset["train"] = train_split
        self.id_field = "id"

        self.partition: BootstrapPartition = partition_bootstrap(
            train_split,
            label_field=self.label_field,
            id_field=self.id_field,
            target_size=self.cot.bootstrap_size,
            seed=int(config.seed) if isinstance(config.seed, int) else 0,
        )
        if self.label_field and self.label_field in train_split.column_names:
            self.label_options = sorted({r[self.label_field] for r in train_split})
        else:
            self.label_options = []

        validation = dataset.get("validation", dataset.get("test"))
        self.dataset_a: Dict[str, Dataset] = {
            "train": self.partition.bootstrap_a,
            "validation": validation,
        }
        self.dataset_b: Dict[str, Dataset] = {
            "train": self.partition.bootstrap_b,
            "validation": validation,
        }

        self.dual = DualStudentTrainer(config, self.dataset_a, self.out_dir)
        self.dual.dataset_a = self.dataset_a
        self.dual.dataset_b = self.dataset_b

        self.actions = enumerate_cotrain_actions(
            operators=self.cot.operator_choices,
            volumes=self.cot.volume_choices,
            taus=self.cot.tau_choices,
            include_stop=self.cot.include_stop,
        )
        self.controller_p = FrugalKDCoTrainP(
            stop_accept_rate_threshold=self.cot.stop_accept_rate_threshold,
        )
        self.controller_bai: Optional[LinearBAIController] = (
            LinearBAIController(actions=self.actions, feature_dim=8, seed=0)
            if self.cot.controller == "frugalkd_cotrain_linearbai" else None
        )

        self.audit_ledger = AuditLedger(
            path=self.out_dir / "cotrain_ledger.jsonl", run_id=self.run_id
        )
        token_budget = (
            config.token_budget
            if isinstance(config.token_budget, int)
            else None
        )
        self.token_tracker = TokenTracker(
            experiment_name=config.name,
            teacher_model=self.cot.strong_teacher,
            token_budget=token_budget,
            budget_warning=config.budget_warning,
            quiet=True,
        )

        self.generator_a = VariantGenerator(
            task_kind=self.cot.task_kind,
            text_field=self.text_field,
            label_field=self.label_field or "label",
            generate_fn=generate_fn_a,
        )
        self.generator_b = VariantGenerator(
            task_kind=self.cot.task_kind,
            text_field=self.text_field,
            label_field=self.label_field or "label",
            generate_fn=generate_fn_b,
        )
        self.arbiter = StrongTeacherArbiter(
            model=self.cot.strong_teacher,
            client=_LiteLLMChatClient(arbiter_complete),
            task_kind=self.cot.task_kind,
            self_consistency_n=self.cot.self_consistency_n,
            self_consistency_temperature=self.cot.self_consistency_temperature,
        )
        self.results: Dict[str, Any] = {"cycles": [], "run_id": self.run_id}

    async def run(self) -> None:
        cycles = (
            self.config.cycles if isinstance(self.config.cycles, int)
            else self.config.cycles[0]
        )
        prev_metric: Optional[float] = None
        accepted_counts: Dict[str, int] = {}
        for t in range(cycles):
            with self.token_tracker.cycle(t):
                self.dual.train_a()
                self.dual.train_b()
                metrics = self.dual.evaluate(split="validation")
                preds = self.dual.detailed_predictions(split="validation")

                a_errs = [
                    p != y for p, y in zip(
                        preds["a"].predicted_labels, preds["a"].true_labels
                    )
                ]
                b_errs = [
                    p != y for p, y in zip(
                        preds["b"].predicted_labels, preds["b"].true_labels
                    )
                ]
                a_summary = StudentEvalSummary(
                    error_rate=sum(a_errs) / max(len(a_errs), 1),
                    errors_aligned=a_errs,
                )
                b_summary = StudentEvalSummary(
                    error_rate=sum(b_errs) / max(len(b_errs), 1),
                    errors_aligned=b_errs,
                )
                current_metric = metrics["a"].get(self.cot.stop_metric)

                features = build_cotrain_features(
                    cycle=t, cycles=cycles,
                    student_a=a_summary, student_b=b_summary,
                    prev_validation_metric=prev_metric,
                    current_validation_metric=current_metric,
                    accepted_counts=accepted_counts,
                    budget=_budget_state_of(self.token_tracker),
                    predicted_cost_next_cycle=0.0,
                )

                if self.controller_bai is not None and t > 0:
                    arm = self.controller_bai.recommend()
                    self.controller_bai.update(
                        arm=arm, reward=current_metric or 0.0
                    )
                    choice_action = arm
                else:
                    choice = self.controller_p.select(features, self.actions)
                    choice_action = choice.action

                self._log_cycle(t, choice_action, features, metrics)
                if choice_action.is_stop or (
                    self.force_stop_after_cycle is not None
                    and t >= self.force_stop_after_cycle
                ):
                    self.results["stopped_at_cycle"] = t
                    break

                fault_seeds_a = self._fault_seeds(
                    self._predict_other_on(self.partition.bootstrap_a, other="b")
                )
                fault_seeds_b = self._fault_seeds(
                    self._predict_other_on(self.partition.bootstrap_b, other="a")
                )

                allocation = allocate_volumes(
                    total_volume=choice_action.volume,
                    error_ratio=features["error_ratio"],
                    low=self.cot.asymmetric_flow_low,
                    high=self.cot.asymmetric_flow_high,
                )

                a_to_b = await self._generate_and_validate(
                    proposer="a", receiver="b",
                    seeds=fault_seeds_a[: allocation.a_to_b],
                    cycle=t, action=choice_action,
                )
                b_to_a = await self._generate_and_validate(
                    proposer="b", receiver="a",
                    seeds=fault_seeds_b[: allocation.b_to_a],
                    cycle=t, action=choice_action,
                )

                self._merge_into("b", a_to_b)
                self._merge_into("a", b_to_a)

                accepted_counts = self.audit_ledger.summary_counts()
                prev_metric = current_metric
        self._save_results()

    def _fault_seeds(
        self, preds_other_on_my_bootstrap: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [r for r in preds_other_on_my_bootstrap if r["is_error"]]

    def _predict_other_on(
        self, owner_bootstrap: Dataset, other: str
    ) -> List[Dict[str, Any]]:
        trainer = self.dual.trainer_a if other == "a" else self.dual.trainer_b
        model = self.dual.model_a if other == "a" else self.dual.model_b
        trainer.dataset["__cotrain_probe__"] = owner_bootstrap
        try:
            preds = trainer.get_detailed_predictions(model, split="__cotrain_probe__")
        finally:
            trainer.dataset.pop("__cotrain_probe__", None)
        out = []
        for i, (p_label, t_label) in enumerate(
            zip(preds.predicted_labels, preds.true_labels)
        ):
            row = owner_bootstrap[i]
            out.append({
                "seed_id": row[self.id_field],
                "seed_text": row[self.text_field],
                "seed_label": str(t_label),
                "predicted_by_other": str(p_label),
                "is_error": p_label != t_label,
            })
        return out

    async def _generate_and_validate(
        self, *, proposer: str, receiver: str,
        seeds: List[Dict[str, Any]], cycle: int, action: CoTrainAction,
    ) -> List[Dict[str, Any]]:
        gen = self.generator_a if proposer == "a" else self.generator_b
        peer = _StudentPeerLabeller(
            self.dual, side=receiver, label_options=self.label_options
        )
        accepted: List[Dict[str, Any]] = []
        for seed in seeds:
            req = GenerationRequest(
                seed_id=seed["seed_id"], seed_text=seed["seed_text"],
                seed_label=seed["seed_label"], operator=action.operator,
                confused_with=seed.get("predicted_by_other"),
                k=self.cot.variants_per_seed, temperature=0.8,
            )
            out = gen.generate(req)
            for variant in out.variants:
                pipe = ValidationPipeline(
                    peer_labeller=peer, arbiter=self.arbiter,
                    proposer_confidence=0.95,
                    ledger=self.audit_ledger,
                )
                review = VariantUnderReview(
                    cycle=cycle, variant=variant,
                    proposer=proposer, receiver=receiver, action=action,
                    label_options=self.label_options or None,
                )
                outcome = await pipe.validate(review)
                if outcome.accepted_label is not None:
                    accepted.append({
                        self.text_field: variant.text,
                        (self.label_field or "label"): outcome.accepted_label,
                        self.id_field: outcome.record.variant_id,
                    })
        return accepted

    def _merge_into(self, side: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        ds_pair = self.dual.dataset_a if side == "a" else self.dual.dataset_b
        new_train = concatenate_datasets([
            ds_pair["train"], Dataset.from_list(rows)
        ])
        ds_pair["train"] = new_train
        self.dual.set_train_split(side, new_train)

    def _log_cycle(
        self, cycle: int, action: CoTrainAction,
        features: Dict[str, Any], metrics: Dict[str, Any],
    ) -> None:
        self.results["cycles"].append({
            "cycle": cycle,
            "action": action.model_dump(),
            "features": features,
            "metrics": metrics,
        })

    def _save_results(self) -> None:
        (self.out_dir / "cotrain_results.json").write_text(
            json.dumps(self.results, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def from_config(
        cls, config: ExperimentConfig, *, out_dir: Optional[Path] = None
    ) -> "CoTrainEngine":
        """Build a CoTrainEngine from an ExperimentConfig for CLI runs.

        Loads the dataset via the same helpers DistillationEngine uses, wires
        the arbiter to LiteLLM acompletion, and binds each student's variant
        generation to its trainer's generate_text() method.
        """
        from datasets import load_dataset

        from ..engine import (
            ensure_class_label,
            ensure_validation_split,
            prepare_dataset,
        )
        from ..reproducibility import dataset_load_kwargs

        out_path = Path(out_dir) if out_dir else (
            Path(config.base_output_dir) / config.name
        )

        async def _arbiter_complete(*, model, messages, temperature, max_tokens):
            from litellm import acompletion
            resp = await acompletion(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return {
                "text": resp.choices[0].message.content,
                "usage": {
                    "input": resp.usage.prompt_tokens,
                    "output": resp.usage.completion_tokens,
                },
            }

        # Lazy student-as-generator: at construction time the dual trainer
        # has not yet been built. We close over a holder that gets the
        # finished engine after cls(...) returns.
        engine_holder: Dict[str, "CoTrainEngine"] = {}

        def _student_generate(student_side: str):
            def _gen(prompt: str, *, n: int, temperature: float):
                engine = engine_holder.get("engine")
                if engine is None:
                    raise RuntimeError(
                        "student generator invoked before engine construction"
                    )
                trainer = (
                    engine.dual.trainer_a if student_side == "a"
                    else engine.dual.trainer_b
                )
                if trainer is None:
                    raise RuntimeError(
                        "student trainer not yet built; cotrain run must train "
                        "before generating variants"
                    )
                if not hasattr(trainer, "generate_text"):
                    raise RuntimeError(
                        "trainer does not implement generate_text(); cotrain "
                        "mode currently requires student_type='causal_lm_sft'"
                    )
                return trainer.generate_text(
                    prompt, n=n, temperature=temperature, max_new_tokens=512,
                )
            return _gen

        dataset_kwargs = dataset_load_kwargs(config)
        if config.dataset_subset:
            dataset = load_dataset(config.dataset, config.dataset_subset, **dataset_kwargs)
        else:
            dataset = load_dataset(config.dataset, **dataset_kwargs)
        if config.sampling.enabled:
            dataset = ensure_class_label(dataset, config.sampling.stratify_column)
        dataset = prepare_dataset(dataset, config.sampling)
        dataset = ensure_validation_split(dataset, config)

        engine = cls(
            config, dataset=dataset, out_dir=out_path,
            generate_fn_a=_student_generate("a"),
            generate_fn_b=_student_generate("b"),
            arbiter_complete=_arbiter_complete,
        )
        engine_holder["engine"] = engine
        return engine
