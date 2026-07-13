"""Policy feature extraction for label-free generative tasks."""

from promptillery.policy_features import build_policy_features
from promptillery.trainers.base import PredictionResult


def test_label_free_predictions_do_not_require_num_classes():
    predictions = PredictionResult(
        indices=[0],
        predicted_labels=["42"],
        true_labels=["42"],
        confidences=[1.0],
        entropies=[0.0],
    )

    features = build_policy_features(
        cycle=0,
        cycles=5,
        metrics={"exact_match": 1.0},
        previous_metrics=None,
        train_predictions=predictions,
        eval_predictions=predictions,
        train_size=1,
        synthetic_count=None,
        budget={},
        num_classes=None,
    )

    assert features["train_entropy_mean"] == 0.0
    assert features["train_entropy_normalized_mean"] == 0.0
