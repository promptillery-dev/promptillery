"""Token usage tracking for LLM API calls."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

# Try to import completion_cost at module level for efficiency
try:
    from litellm import completion_cost as _completion_cost

    _HAS_COMPLETION_COST = True
except ImportError:
    _HAS_COMPLETION_COST = False
    _completion_cost = None  # type: ignore


class OperationType(str, Enum):
    """Types of token-consuming operations."""

    AUGMENTATION = "augmentation"
    PSEUDO_LABELING = "pseudo_labeling"


class TokenUsage(BaseModel):
    """Token usage for a single or aggregated set of API calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: Optional[float] = None

    def add(self, other: "TokenUsage") -> None:
        """Add another TokenUsage to this one (in-place)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        if other.estimated_cost is not None:
            if self.estimated_cost is None:
                self.estimated_cost = 0.0
            self.estimated_cost += other.estimated_cost


class CycleTokenUsage(BaseModel):
    """Token usage breakdown for a single cycle."""

    cycle: int
    operations: Dict[str, TokenUsage] = Field(default_factory=dict)
    cycle_total: TokenUsage = Field(default_factory=TokenUsage)


class TokenUsageSummary(BaseModel):
    """Complete token usage summary for an experiment."""

    experiment_name: str
    teacher_model: str
    cycles_completed: int = 0
    per_cycle: List[CycleTokenUsage] = Field(default_factory=list)
    totals: Dict[str, TokenUsage] = Field(default_factory=dict)
    grand_total: TokenUsage = Field(default_factory=TokenUsage)


def _extract_usage_from_response(response: Any) -> TokenUsage:
    """Extract token usage from a litellm response.

    Handles both dict responses and ModelResponse objects from litellm.
    """
    usage_data: dict = {}

    # Handle ModelResponse objects (have .usage attribute)
    if hasattr(response, "usage") and response.usage is not None:
        usage_obj = response.usage
        if hasattr(usage_obj, "model_dump"):
            usage_data = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage_data = usage_obj
        else:
            # Fallback: try to access as attributes
            usage_data = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }
    # Handle plain dict responses
    elif isinstance(response, dict):
        usage_data = response.get("usage", {})

    input_tokens = usage_data.get("prompt_tokens", 0) or 0
    output_tokens = usage_data.get("completion_tokens", 0) or 0
    total_tokens = usage_data.get("total_tokens", 0) or 0

    # Calculate cost using litellm's completion_cost
    estimated_cost = None
    if _HAS_COMPLETION_COST and _completion_cost is not None:
        try:
            estimated_cost = _completion_cost(completion_response=response)
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Could not calculate cost: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error calculating cost: {e}")

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
    )


class TokenTracker:
    """Tracks token usage across cycles and operations."""

    def __init__(
        self,
        experiment_name: str,
        teacher_model: str,
        console: Optional[Console] = None,
        quiet: bool = False,
        budget_warning: Optional[float] = None,
        budget_stop: bool = False,
    ) -> None:
        self.summary = TokenUsageSummary(
            experiment_name=experiment_name,
            teacher_model=teacher_model,
        )
        self._current_cycle: Optional[CycleTokenUsage] = None
        self._quiet = quiet
        self.console = (
            console if console is not None else (None if quiet else Console())
        )
        self.budget_warning = budget_warning
        self.budget_stop = budget_stop
        self._budget_warning_shown = False
        self._budget_exceeded = False

    def start_cycle(self, cycle_num: int) -> None:
        """Initialize tracking for a new cycle."""
        self._current_cycle = CycleTokenUsage(cycle=cycle_num)

    @contextmanager
    def cycle(self, cycle_num: int) -> Generator[None, None, None]:
        """Context manager for cycle token tracking.

        Usage:
            with tracker.cycle(0):
                # ... do work that uses tokens ...
        """
        self.start_cycle(cycle_num)
        try:
            yield
        finally:
            self.end_cycle()

    def record_usage(self, response: Any, operation: OperationType) -> TokenUsage:
        """Extract and record token usage from a litellm response.

        Raises:
            RuntimeError: If called without an active cycle.
        """
        usage = _extract_usage_from_response(response)

        if self._current_cycle is None:
            raise RuntimeError(
                "record_usage called without active cycle. "
                "Call start_cycle() first or use the cycle() context manager."
            )

        # Add to operation-specific tracking (dict-based for extensibility)
        op_key = operation.value
        if op_key not in self._current_cycle.operations:
            self._current_cycle.operations[op_key] = TokenUsage()
        self._current_cycle.operations[op_key].add(usage)

        # Add to cycle total
        self._current_cycle.cycle_total.add(usage)

        return usage

    def end_cycle(self) -> None:
        """Finalize current cycle and update totals."""
        if self._current_cycle is None:
            return

        # Add cycle to list
        self.summary.per_cycle.append(self._current_cycle)
        self.summary.cycles_completed += 1

        # Update operation totals (dict-based for extensibility)
        for op_key, op_usage in self._current_cycle.operations.items():
            if op_key not in self.summary.totals:
                self.summary.totals[op_key] = TokenUsage()
            self.summary.totals[op_key].add(op_usage)

        # Update grand total
        self.summary.grand_total.add(self._current_cycle.cycle_total)

        # Check budget warning
        self._check_budget_warning()

        self._current_cycle = None

    def _check_budget_warning(self) -> None:
        """Check if budget warning threshold has been exceeded."""
        if (
            self.budget_warning is not None
            and not self._budget_warning_shown
            and self.summary.grand_total.estimated_cost is not None
            and self.summary.grand_total.estimated_cost >= self.budget_warning
        ):
            self._budget_warning_shown = True
            self._budget_exceeded = True

            if not self._quiet and self.console is not None:
                warning_msg = (
                    f"⚠️  Budget Warning: Total cost ${self.summary.grand_total.estimated_cost:.4f} "
                    f"has reached or exceeded budget limit of ${self.budget_warning:.4f}"
                )
                self.console.print(f"[bold yellow]{warning_msg}[/bold yellow]")

                if self.budget_stop:
                    stop_msg = "🛑 Budget Stop: Experiment will stop after this cycle"
                    self.console.print(f"[bold red]{stop_msg}[/bold red]")

            logger.warning(
                f"Budget warning triggered: ${self.summary.grand_total.estimated_cost:.4f} >= ${self.budget_warning:.4f}"
            )
            if self.budget_stop:
                logger.warning(
                    "Budget stop enabled - experiment will halt after current cycle"
                )

    def should_stop_for_budget(self) -> bool:
        """Check if experiment should stop due to budget constraints.

        Returns:
            True if budget_stop is enabled and budget has been exceeded.
        """
        return self.budget_stop and self._budget_exceeded

    def print_cycle_summary(self) -> None:
        """Display cycle summary using Rich table."""
        if self._quiet or self.console is None:
            return

        if not self.summary.per_cycle:
            return

        cycle_data = self.summary.per_cycle[-1]

        # Skip printing if no tokens were used this cycle
        if cycle_data.cycle_total.total_tokens == 0:
            return

        table = Table(title=f"Token Usage - Cycle {cycle_data.cycle}")
        table.add_column("Operation", style="cyan")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Cost", justify="right", style="green")

        def format_cost(cost: Optional[float]) -> str:
            return f"${cost:.4f}" if cost is not None else "-"

        def add_row(name: str, usage: TokenUsage) -> None:
            table.add_row(
                name,
                f"{usage.input_tokens:,}",
                f"{usage.output_tokens:,}",
                f"{usage.total_tokens:,}",
                format_cost(usage.estimated_cost),
            )

        # Print all operations dynamically
        for op_key, op_usage in cycle_data.operations.items():
            display_name = op_key.replace("_", " ").title()
            add_row(display_name, op_usage)

        table.add_section()
        add_row("Cycle Total", cycle_data.cycle_total)

        self.console.print(table)

    def print_final_summary(self) -> None:
        """Display experiment summary using Rich panel."""
        if self._quiet or self.console is None:
            return

        if self.summary.grand_total.total_tokens == 0:
            return

        lines = []
        lines.append(f"Model: {self.summary.teacher_model}")
        lines.append(f"Cycles: {self.summary.cycles_completed}")
        lines.append("")
        lines.append(f"Input Tokens: {self.summary.grand_total.input_tokens:,}")
        lines.append(f"Output Tokens: {self.summary.grand_total.output_tokens:,}")
        lines.append(f"Total Tokens: {self.summary.grand_total.total_tokens:,}")

        if self.summary.grand_total.estimated_cost is not None:
            lines.append(
                f"Estimated Cost: ${self.summary.grand_total.estimated_cost:.4f}"
            )

            # Show budget information if configured
            if self.budget_warning is not None:
                lines.append(f"Budget Warning Limit: ${self.budget_warning:.4f}")
                remaining = (
                    self.budget_warning - self.summary.grand_total.estimated_cost
                )
                if remaining >= 0:
                    lines.append(f"Budget Remaining: ${remaining:.4f}")
                else:
                    lines.append(f"Budget Exceeded By: ${abs(remaining):.4f}")

        panel = Panel(
            "\n".join(lines),
            title="Token Usage Summary",
            border_style="blue",
        )
        self.console.print(panel)

    def save(self, output_dir: Path, filename: str = "token_usage.json") -> Path:
        """Save token usage to JSON file.

        Args:
            output_dir: Directory to save the file in.
            filename: Name of the output file (default: token_usage.json).

        Returns:
            Path to the saved file.
        """
        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.summary.model_dump(), f, indent=2)
        logger.info(f"Token usage saved to: {output_path}")
        return output_path
