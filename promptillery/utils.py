"""Utility functions."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .trainers.base import PredictionResult

import numpy as np
import torch
from jinja2 import Environment, Template
from rich.logging import RichHandler
from sklearn.metrics import classification_report as sklearn_classification_report


def setup_logging(level: int = logging.INFO) -> None:
    """Configure rich logging."""
    logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler()])


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Default Jinja2 templates for formatting samples
DEFAULT_SAMPLE_TEMPLATE = """\
{% for sample in samples %}
Example {{ loop.index }}:
Text: {{ sample.text }}
Label: {{ sample.label }}
{% if include_prediction %}
Predicted: {{ sample.predicted_label }} (confidence: {{ "%.2f"|format(sample.confidence) }})
{% endif %}
{% endfor %}
"""

DEFAULT_SAMPLE_TEMPLATE_WITH_ENTROPY = """\
{% for sample in samples %}
Example {{ loop.index }}:
Text: {{ sample.text }}
Label: {{ sample.label }}
{% if include_prediction %}
Predicted: {{ sample.predicted_label }} (confidence: {{ "%.2f"|format(sample.confidence) }}{% if sample.entropy is defined %}, entropy: {{ "%.3f"|format(sample.entropy) }}{% endif %})
{% endif %}
{% endfor %}
"""


def format_samples_for_prompt(
    samples: List[Dict[str, Any]],
    include_prediction: bool = False,
    template: Optional[str] = None,
    label_map: Optional[Dict[int, str]] = None,
) -> str:
    """Format samples for inclusion in a teacher model prompt.

    Supports different sample types for knowledge distillation:
    - Few-shot examples: Clean exemplar samples (include_prediction=False)
    - High-entropy samples: Uncertain predictions (include_prediction=True, samples contain entropy)
    - Hard negatives: Misclassifications with high confidence (include_prediction=True)

    Args:
        samples: List of sample dictionaries. Each sample should contain:
            - text: The sample text
            - label: The ground truth label
            For prediction-included samples:
            - predicted_label: The model's prediction
            - confidence: Prediction confidence score (0-1)
            - entropy (optional): Prediction entropy for high-uncertainty samples
        include_prediction: Whether to include model predictions and confidence.
            Set to False for few-shot examples, True for hard negatives/high-entropy.
        template: Optional custom Jinja2 template string. If not provided, uses
            default template. Template has access to:
            - samples: The list of samples
            - include_prediction: Boolean flag
            - loop.index: Current sample number (1-indexed)
        label_map: Optional mapping from label indices to human-readable names.
            If provided, labels will be converted using this mapping.

    Returns:
        Formatted string suitable for prompt injection.

    Examples:
        >>> # Few-shot examples (no predictions)
        >>> few_shot = [
        ...     {"text": "Great product!", "label": 1},
        ...     {"text": "Terrible service", "label": 0},
        ... ]
        >>> format_samples_for_prompt(few_shot, include_prediction=False)

        >>> # Hard negatives (high confidence misclassifications)
        >>> hard_negs = [
        ...     {"text": "Not bad actually", "label": 1, "predicted_label": 0, "confidence": 0.95},
        ... ]
        >>> format_samples_for_prompt(hard_negs, include_prediction=True)

        >>> # High entropy samples (uncertain predictions)
        >>> uncertain = [
        ...     {"text": "It's okay I guess", "label": 1, "predicted_label": 0,
        ...      "confidence": 0.52, "entropy": 0.98},
        ... ]
        >>> format_samples_for_prompt(uncertain, include_prediction=True)
    """
    if not samples:
        return ""

    # Apply label mapping if provided
    if label_map:
        samples = _apply_label_map(samples, label_map, include_prediction)

    # Select appropriate default template
    if template is None:
        has_entropy = any("entropy" in s for s in samples)
        template = (
            DEFAULT_SAMPLE_TEMPLATE_WITH_ENTROPY
            if has_entropy
            else DEFAULT_SAMPLE_TEMPLATE
        )

    jinja_template = Template(template)
    return jinja_template.render(
        samples=samples,
        include_prediction=include_prediction,
    ).strip()


def _apply_label_map(
    samples: List[Dict[str, Any]],
    label_map: Dict[int, str],
    include_prediction: bool,
) -> List[Dict[str, Any]]:
    """Apply label mapping to samples, converting indices to names."""
    mapped_samples = []
    for sample in samples:
        mapped = sample.copy()
        if "label" in mapped and mapped["label"] in label_map:
            mapped["label"] = label_map[mapped["label"]]
        if include_prediction and "predicted_label" in mapped:
            if mapped["predicted_label"] in label_map:
                mapped["predicted_label"] = label_map[mapped["predicted_label"]]
        mapped_samples.append(mapped)
    return mapped_samples


def create_prompt_environment() -> Environment:
    """Create a Jinja2 environment with prompt utility functions available.

    Returns an Environment with prompt utilities registered as globals,
    allowing them to be called directly in prompt templates.

    Available functions:
        - format_samples_for_prompt: Format sample lists for prompts
        - format_classification_report: Generate classification report from predictions

    Example usage in YAML config:
        prompt: |
            Here are some examples:
            {{ format_samples_for_prompt(few_shot_examples, include_prediction=False) }}

            Hard negatives the model struggled with:
            {{ format_samples_for_prompt(hard_negatives, include_prediction=True) }}

            Current model performance:
            {{ classification_report }}

    Returns:
        Jinja2 Environment with registered prompt utilities.
    """
    env = Environment()
    env.globals["format_samples_for_prompt"] = format_samples_for_prompt
    env.globals["format_classification_report"] = format_classification_report
    return env


def extract_few_shot_samples(
    dataset,
    n_per_class: int = 2,
    text_column: str = "text",
    label_column: str = "label",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Extract balanced few-shot examples from a dataset.

    Selects n_per_class correctly classified examples per label for use as
    few-shot demonstrations in prompts.

    Args:
        dataset: HuggingFace Dataset or similar with text and label columns
        n_per_class: Number of examples to select per class
        text_column: Name of the text column
        label_column: Name of the label column
        seed: Random seed for reproducibility

    Returns:
        List of sample dictionaries with 'text' and 'label' keys
    """
    random.seed(seed)

    # Group samples by label
    samples_by_label: Dict[Any, List[Dict[str, Any]]] = {}
    for i in range(len(dataset)):
        label = dataset[i][label_column]
        if label not in samples_by_label:
            samples_by_label[label] = []
        samples_by_label[label].append(
            {
                "text": dataset[i][text_column],
                "label": label,
            }
        )

    # Select n_per_class from each label
    few_shot = []
    for label, samples in sorted(samples_by_label.items()):
        selected = random.sample(samples, min(n_per_class, len(samples)))
        few_shot.extend(selected)

    return few_shot


def extract_high_entropy_samples(
    dataset,
    predictions: "PredictionResult",
    top_k: int = 10,
    text_column: str = "text",
    label_column: str = "label",
) -> List[Dict[str, Any]]:
    """Extract samples with highest prediction entropy (most uncertain).

    These are samples where the model is least confident, useful for
    identifying decision boundaries the model struggles with.

    Args:
        dataset: HuggingFace Dataset or similar
        predictions: PredictionResult from trainer.get_detailed_predictions()
        top_k: Number of high-entropy samples to return
        text_column: Name of the text column
        label_column: Name of the label column

    Returns:
        List of sample dictionaries with text, label, predicted_label,
        confidence, and entropy keys
    """
    if predictions.entropies is None:
        return []

    # Sort by entropy (descending)
    sorted_indices = sorted(
        range(len(predictions.entropies)),
        key=lambda i: predictions.entropies[i],
        reverse=True,
    )[:top_k]

    samples = []
    for i in sorted_indices:
        idx = predictions.indices[i]
        samples.append(
            {
                "text": dataset[idx][text_column],
                "label": predictions.true_labels[i],
                "predicted_label": predictions.predicted_labels[i],
                "confidence": predictions.confidences[i],
                "entropy": predictions.entropies[i],
            }
        )

    return samples


def extract_hard_negatives(
    dataset,
    predictions: "PredictionResult",
    top_k: int = 10,
    confidence_threshold: Optional[float] = None,
    text_column: str = "text",
    label_column: str = "label",
) -> List[Dict[str, Any]]:
    """Extract misclassified samples with highest confidence (hard negatives).

    These are samples the model got wrong despite being confident,
    indicating systematic errors or confusing patterns.

    Args:
        dataset: HuggingFace Dataset or similar
        predictions: PredictionResult from trainer.get_detailed_predictions()
        top_k: Maximum number of hard negatives to return
        confidence_threshold: Optional minimum confidence threshold. If None,
            returns top_k misclassified samples by confidence regardless of threshold.
        text_column: Name of the text column
        label_column: Name of the label column

    Returns:
        List of sample dictionaries with text, label, predicted_label,
        and confidence keys, sorted by confidence (highest first)
    """
    # Find all misclassified samples
    hard_negs = []
    for i in range(len(predictions.indices)):
        pred = predictions.predicted_labels[i]
        true = predictions.true_labels[i]
        conf = predictions.confidences[i]

        # Only include misclassified samples
        if pred != true:
            # Apply threshold if specified
            if confidence_threshold is not None and conf < confidence_threshold:
                continue
            idx = predictions.indices[i]
            hard_negs.append(
                {
                    "text": dataset[idx][text_column],
                    "label": true,
                    "predicted_label": pred,
                    "confidence": conf,
                }
            )

    # Sort by confidence (highest first) and take top_k
    hard_negs.sort(key=lambda x: x["confidence"], reverse=True)
    return hard_negs[:top_k]


def format_classification_report(
    predictions: "PredictionResult",
    label_map: Optional[Dict[int, str]] = None,
) -> str:
    """Generate a classification report from predictions.

    Creates a text-based classification report showing per-class precision,
    recall, F1-score, and support. This gives the teacher model insight into
    which classes the student model struggles with.

    Args:
        predictions: PredictionResult from trainer.get_detailed_predictions()
        label_map: Optional mapping from label indices to human-readable names.
            If provided, class names will be displayed instead of indices.

    Returns:
        Formatted classification report string.

    Example output:
                      precision    recall  f1-score   support

            negative       0.65      0.72      0.68       150
             neutral       0.45      0.38      0.41       120
            positive       0.71      0.69      0.70       180

            accuracy                           0.61       450
           macro avg       0.60      0.60      0.60       450
        weighted avg       0.61      0.61      0.61       450
    """
    if not predictions.predicted_labels or not predictions.true_labels:
        return "No predictions available for classification report."

    # Prepare target names if label_map provided
    target_names = None
    if label_map:
        # Get unique labels and sort them
        unique_labels = sorted(
            set(predictions.true_labels + predictions.predicted_labels)
        )
        target_names = [label_map.get(label, str(label)) for label in unique_labels]

    report = sklearn_classification_report(
        y_true=predictions.true_labels,
        y_pred=predictions.predicted_labels,
        target_names=target_names,
        zero_division=0,
    )

    return report
