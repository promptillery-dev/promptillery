"""Ablation study utilities for running multiple experiment configurations."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset

from .config import TIMESTAMP_FORMAT, ExperimentConfig
from .engine import DistillationEngine, ensure_class_label, prepare_dataset


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


logger = logging.getLogger(__name__)


class AblationStudyRunner:
    """Run ablation studies with multiple configurations."""

    def __init__(
        self, base_config: ExperimentConfig, cleanup_metric: Optional[str] = None
    ):
        """Initialize ablation study runner.

        Args:
            base_config: Base experiment configuration with list parameters for ablation
            cleanup_metric: Metric to use for selecting best config during cleanup.
                          If None, uses the first metric from config.metrics.
                          After each augmentation_batch_size group completes, only
                          the best-performing config (by this metric) is kept.
        """
        self.base_config = base_config
        self.results = []
        self.ablation_dir = None
        self.cleanup_metric = cleanup_metric or base_config.metrics[0]
        # Track output directories for cleanup
        self._current_group_dirs: Dict[str, Path] = {}  # config_name -> output_dir
        self._current_group_results: List[Dict[str, Any]] = []
        self._current_batch_size: Optional[int] = None

    async def run_single_config(
        self, config: ExperimentConfig, config_name: str, dataset: DatasetDict = None
    ) -> Dict[str, Any]:
        """Run a single configuration and return results."""
        logger.info(f"Running configuration: {config_name}")

        # Run the experiment with pre-loaded dataset if provided
        engine = DistillationEngine(config, dataset=dataset)
        results = await engine.run()

        # Add configuration metadata to results
        results["config_name"] = config_name
        # Store the full config dump - let the config itself determine what's important
        results["config"] = config.model_dump(
            exclude={"base_output_dir", "output_repo"}
        )
        # Store output directory for potential cleanup
        results["_output_dir"] = str(engine.out_dir)

        return results

    def _get_best_metric_value(self, result: Dict[str, Any]) -> Optional[float]:
        """Extract the best metric value from a result dict."""
        if "error" in result:
            return None

        # Check early stopping info for best cycle
        best_cycle = "0"
        if "early_stopping" in result:
            best_cycle = str(result["early_stopping"].get("best_cycle", 0))

        # Get metric from best cycle
        if best_cycle in result and self.cleanup_metric in result[best_cycle]:
            return result[best_cycle][self.cleanup_metric]

        # Fallback: find last cycle with the metric
        cycles = sorted([k for k in result.keys() if k.isdigit()], key=int)
        for cycle in reversed(cycles):
            if self.cleanup_metric in result[cycle]:
                return result[cycle][self.cleanup_metric]

        return None

    def _cleanup_group(self, keep_best: bool = True):
        """Clean up current group, keeping only the best performer.

        Args:
            keep_best: If True, keep the best config. If False, delete all.
        """
        if not self._current_group_results:
            return

        batch_size = self._current_batch_size
        logger.info(
            f"Cleaning up augmentation_batch_size={batch_size} group "
            f"({len(self._current_group_results)} configs)"
        )

        # Find best result by metric
        best_result = None
        best_value = float("-inf")

        for result in self._current_group_results:
            value = self._get_best_metric_value(result)
            if value is not None and value > best_value:
                best_value = value
                best_result = result

        if best_result is None:
            logger.warning("No valid results in group, skipping cleanup")
            return

        best_config_name = best_result["config_name"]
        logger.info(
            f"Best config for batch_size={batch_size}: {best_config_name} "
            f"({self.cleanup_metric}={best_value:.4f})"
        )

        # Delete all directories except the best one
        deleted_count = 0
        freed_space = 0

        for result in self._current_group_results:
            config_name = result["config_name"]
            output_dir = result.get("_output_dir")

            if output_dir is None:
                continue

            output_path = Path(output_dir)

            if keep_best and config_name == best_config_name:
                logger.debug(f"Keeping best: {output_path.name}")
                continue

            if output_path.exists():
                # Calculate size before deletion
                try:
                    dir_size = sum(
                        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
                    )
                    freed_space += dir_size
                except OSError:
                    pass

                # Delete the directory
                try:
                    shutil.rmtree(output_path)
                    deleted_count += 1
                    logger.debug(f"Deleted: {output_path.name}")
                except OSError as e:
                    logger.warning(f"Failed to delete {output_path}: {e}")

        freed_gb = freed_space / (1024**3)
        logger.info(
            f"Cleanup complete: deleted {deleted_count} directories, "
            f"freed {freed_gb:.2f} GB"
        )

        # Clear group tracking
        self._current_group_results = []
        self._current_group_dirs = {}

    def _load_and_prepare_dataset(self) -> DatasetDict:
        """Load and sample dataset once for all ablation configurations.

        Returns:
            Prepared dataset with sampling applied and tracking columns added.
        """
        cfg = self.base_config

        # Load dataset
        dataset_subset = cfg.dataset_subset
        if dataset_subset:
            dataset = load_dataset(cfg.dataset, dataset_subset)
        else:
            dataset = load_dataset(cfg.dataset)

        # Ensure label column is ClassLabel type for stratified sampling
        if cfg.sampling.enabled:
            dataset = ensure_class_label(dataset, cfg.sampling.stratify_column)

        # Apply stratified sampling if configured
        dataset = prepare_dataset(dataset, cfg.sampling)

        # Add tracking columns for origin of each row
        for split, ds in dataset.items():
            ds = ds.add_column("source_split", [split] * len(ds))
            ds = ds.add_column("source_idx", [-1] * len(ds))
            ds = ds.add_column("origin_cycle", [0] * len(ds))
            dataset[split] = ds

        logger.info(
            f"Dataset loaded and sampled once: "
            f"{', '.join(f'{k}={len(v)}' for k, v in dataset.items())}"
        )

        return dataset

    async def run(self, cleanup_after_group: bool = True) -> List[Dict[str, Any]]:
        """Run all configurations generated from list parameters.

        Args:
            cleanup_after_group: If True, after each augmentation_batch_size group
                completes, delete all experiment directories except the best one
                (by cleanup_metric). This dramatically reduces disk usage for
                large ablation studies.

        Returns:
            List of result dictionaries for all configurations.
        """
        # Generate all configurations
        configs = self.base_config.generate_ablation_configs()
        logger.info(f"Generated {len(configs)} configurations for ablation study")

        if cleanup_after_group:
            logger.info(
                f"Cleanup enabled: keeping only best config per augmentation_batch_size "
                f"(metric: {self.cleanup_metric})"
            )

        # Load and sample dataset ONCE for all configurations
        dataset = self._load_and_prepare_dataset()

        # Create a dedicated ablation run directory that contains all experiments and summary
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        ablation_name = f"ablation_{self.base_config.name}_{timestamp}"
        self.ablation_dir = Path(self.base_config.base_output_dir) / ablation_name
        self.ablation_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Ablation study directory: {self.ablation_dir}")

        # Run each configuration with base_output_dir set to the ablation directory
        for config in configs:
            # Check if we're starting a new augmentation_batch_size group
            current_batch_size = config.augmentation_batch_size
            if cleanup_after_group and current_batch_size != self._current_batch_size:
                # Cleanup previous group before starting new one
                if self._current_batch_size is not None:
                    self._cleanup_group(keep_best=True)
                self._current_batch_size = current_batch_size
                logger.info(
                    f"Starting augmentation_batch_size={current_batch_size} group"
                )

            try:
                # Override base_output_dir so experiments are created inside the ablation folder
                config.base_output_dir = str(self.ablation_dir)
                result = await self.run_single_config(
                    config, config.name, dataset=dataset
                )
                self.results.append(result)

                # Track for group cleanup
                if cleanup_after_group:
                    self._current_group_results.append(result)

                # Check if this config stopped due to budget - stop entire ablation
                budget_info = result.get("budget_control", {})
                if budget_info.get("stopped_for_budget", False):
                    logger.warning(
                        f"Configuration {config.name} stopped due to budget limit. "
                        f"Stopping ablation study."
                    )
                    break

            except Exception as e:
                logger.error(f"Failed to run configuration {config.name}: {e}")
                error_result = {
                    "config_name": config.name,
                    "error": str(e),
                    "config": config.model_dump(),
                }
                self.results.append(error_result)

                if cleanup_after_group:
                    self._current_group_results.append(error_result)

        # Cleanup the final group
        if cleanup_after_group and self._current_group_results:
            self._cleanup_group(keep_best=True)

        # Save results
        self._save_results()
        self._generate_summary()

        return self.results

    def _save_results(self):
        """Save all results to a JSON file."""
        # Save raw results in the ablation directory
        with open(self.ablation_dir / "all_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=_json_serializer)

        logger.info(f"Saved ablation study results to {self.ablation_dir}")

    def _generate_summary(self):
        """Generate a summary of the ablation study results."""
        # Extract metrics for each configuration
        summary_data = []
        for result in self.results:
            if "error" in result:
                continue

            # Start with just the config name
            config_summary = {"config_name": result["config_name"]}

            # Add all config parameters (flattened)
            # This is generic - we don't know what parameters exist
            for key, value in result["config"].items():
                if isinstance(value, (str, int, float, bool)):
                    config_summary[key] = value
                elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                    config_summary[key] = "_".join(value) if value else "none"
                elif isinstance(value, dict):
                    # Flatten nested dicts with dot notation
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (str, int, float, bool)):
                            config_summary[f"{key}.{sub_key}"] = sub_value

            # Add early stopping info if available
            early_stopping_triggered = False
            if "early_stopping" in result:
                early_stopping_triggered = result["early_stopping"]["triggered"]
                config_summary["early_stopped"] = early_stopping_triggered
                config_summary["best_cycle"] = result["early_stopping"]["best_cycle"]
                config_summary["total_cycles"] = result["early_stopping"][
                    "total_cycles"
                ]

            # Add metrics from the best cycle (not final cycle if early stopping restored earlier model)
            cycles = [k for k in result.keys() if k.isdigit()]
            if cycles:
                if early_stopping_triggered:
                    # Use metrics from the best cycle when early stopping triggered
                    metrics_cycle = str(result["early_stopping"]["best_cycle"])
                else:
                    # Use final cycle metrics when no early stopping
                    metrics_cycle = str(max(int(c) for c in cycles))

                cycle_metrics = result[metrics_cycle]
                for metric, value in cycle_metrics.items():
                    if isinstance(value, (int, float)):
                        config_summary[f"best_{metric}"] = value

            summary_data.append(config_summary)

        # Create DataFrame and save as CSV
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(self.ablation_dir / "ablation_summary.csv", index=False)

            # Also save a markdown summary
            with open(self.ablation_dir / "ablation_summary.md", "w") as f:
                f.write("# Ablation Study Summary\n\n")
                f.write(f"Total configurations tested: {len(self.results)}\n")
                f.write(f"Successful runs: {len(summary_data)}\n")
                f.write(f"Failed runs: {len(self.results) - len(summary_data)}\n\n")

                f.write("## Results Table\n\n")
                f.write(df.to_markdown(index=False))

                # Find best configuration based on the primary metric from config
                metric_cols = [col for col in df.columns if col.startswith("best_")]
                if metric_cols:
                    # Use the first metric from config's metrics list for deterministic selection
                    primary_metric = f"best_{self.base_config.metrics[0]}"
                    if primary_metric not in df.columns:
                        # Fallback to first available metric if configured one not found
                        primary_metric = metric_cols[0]
                    best_idx = df[primary_metric].dropna().idxmax()

                    f.write(f"\n\n## Best Configuration (by {primary_metric})\n\n")
                    f.write(
                        f"{primary_metric}: {df.loc[best_idx, primary_metric]:.4f}\n"
                    )
                    f.write(f"Configuration: {df.loc[best_idx, 'config_name']}\n")
                    f.write("\nAll metrics:\n")
                    for metric_col in metric_cols:
                        f.write(f"- {metric_col}: {df.loc[best_idx, metric_col]:.4f}\n")

                    f.write("\nConfiguration details:\n")
                    # Show non-metric columns
                    for col in df.columns:
                        if not col.startswith("best_") and col not in [
                            "config_name",
                            "early_stopped",
                            "best_cycle",
                            "total_cycles",
                        ]:
                            f.write(f"- {col}: {df.loc[best_idx, col]}\n")
