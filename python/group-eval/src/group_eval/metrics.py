# Extracted from the private AI Camera Coach repository and adapted to run standalone.
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Sequence

RELATIONS = ("BEST>GOOD", "BEST>OTHER", "GOOD>OTHER")

def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def prediction_metrics(predictions: Sequence[dict[str, Any]], *, ece_bins: int, high_confidence: float) -> dict[str, Any]:
    """Summarize pairwise accuracy, group-macro accuracy, loss, and calibration."""
    if not predictions:
        raise ValueError("metrics require predictions")
    epsilon = 1e-7
    targets = [int(float(row["target"]) >= 0.5) for row in predictions]
    probabilities = [float(row["probability_a"]) for row in predictions]
    predicted = [int(value >= 0.5) for value in probabilities]
    correct = [int(value == target) for value, target in zip(predicted, targets, strict=True)]
    positives = [index for index, target in enumerate(targets) if target == 1]
    negatives = [index for index, target in enumerate(targets) if target == 0]
    tpr = statistics.fmean(correct[index] for index in positives) if positives else 0.0
    tnr = statistics.fmean(correct[index] for index in negatives) if negatives else 0.0
    bce = statistics.fmean(
        -target * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        - (1 - target) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for target, probability in zip(targets, probabilities, strict=True)
    )
    brier = statistics.fmean(
        (probability - target) ** 2
        for target, probability in zip(targets, probabilities, strict=True)
    )
    confidences = [abs(probability - 0.5) * 2.0 for probability in probabilities]
    high_indices = [index for index, value in enumerate(confidences) if value >= high_confidence]
    calibration_bins = []
    ece = 0.0
    for index in range(ece_bins):
        lower = index / ece_bins
        upper = (index + 1) / ece_bins
        members = [
            item
            for item, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == ece_bins - 1 and probability == 1.0)
        ]
        if members:
            mean_probability = statistics.fmean(probabilities[item] for item in members)
            positive_rate = statistics.fmean(targets[item] for item in members)
            contribution = len(members) / len(predictions) * abs(mean_probability - positive_rate)
            ece += contribution
        else:
            mean_probability = None
            positive_rate = None
            contribution = 0.0
        calibration_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_probability_a": mean_probability,
                "observed_a_rate": positive_rate,
                "ece_contribution": contribution,
            }
        )
    confidence_histogram = []
    for index in range(5):
        lower = index / 5.0
        upper = (index + 1) / 5.0
        members = [
            item
            for item, value in enumerate(confidences)
            if lower <= value < upper or (index == 4 and value == 1.0)
        ]
        confidence_histogram.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "accuracy": statistics.fmean(correct[item] for item in members) if members else None,
            }
        )
    per_group = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(predictions):
        grouped[row["group_id"]].append(index)
    for group_id, indices in sorted(grouped.items()):
        per_group.append(
            {
                "group_id": group_id,
                "pairs": len(indices),
                "accuracy": statistics.fmean(correct[index] for index in indices),
                "bce": statistics.fmean(
                    -targets[index]
                    * math.log(min(1.0 - epsilon, max(epsilon, probabilities[index])))
                    - (1 - targets[index])
                    * math.log(
                        min(1.0 - epsilon, max(epsilon, 1.0 - probabilities[index]))
                    )
                    for index in indices
                ),
            }
        )
    relation_metrics: dict[str, Any] = {}
    for relation in RELATIONS:
        rows = [row for row in predictions if row["relation"] == relation]
        if rows:
            relation_metrics[relation] = {
                "pairs": len(rows),
                "accuracy": statistics.fmean(1.0 if row["correct"] else 0.0 for row in rows),
                "mean_confidence": statistics.fmean(float(row["confidence"]) for row in rows),
            }
    return {
        "pairs": len(predictions),
        "pairwise_accuracy": statistics.fmean(correct),
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "bce_log_loss": bce,
        "brier_score": brier,
        "ece": ece,
        "calibration_bins": calibration_bins,
        "confidence": {
            "mean": statistics.fmean(confidences),
            "median": statistics.median(confidences),
            "p90": _percentile(confidences, 0.90),
            "histogram": confidence_histogram,
            "high_confidence_threshold": high_confidence,
            "high_confidence_pairs": len(high_indices),
            "high_confidence_error_rate": (
                statistics.fmean(1 - correct[index] for index in high_indices)
                if high_indices
                else None
            ),
        },
        "orientation": {
            "a_targets": len(positives),
            "b_targets": len(negatives),
            "true_positive_rate": tpr,
            "true_negative_rate": tnr,
        },
        "group_macro_pair_accuracy": statistics.fmean(row["accuracy"] for row in per_group),
        "group_macro_bce": statistics.fmean(row["bce"] for row in per_group),
        "per_group": per_group,
        "relation": relation_metrics,
    }


def group_bootstrap_interval(
    predictions: Sequence[dict[str, Any]],
    *,
    metric: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    """Return a seeded percentile 95% interval by resampling whole-group metric values."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["group_id"]].append(row)
    groups = sorted(grouped)
    if not groups:
        raise ValueError("bootstrap requires groups")
    values_by_group: dict[str, float] = {}
    for group_id, rows in grouped.items():
        if metric == "accuracy":
            values_by_group[group_id] = statistics.fmean(
                1.0 if row["correct"] else 0.0 for row in rows
            )
        elif metric == "bce":
            values_by_group[group_id] = prediction_metrics(
                rows, ece_bins=10, high_confidence=0.8
            )["bce_log_loss"]
        else:
            raise ValueError(f"unsupported bootstrap metric: {metric}")
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(values_by_group[rng.choice(groups)] for _ in groups)
        for _ in range(samples)
    ]
    return {
        "unit": "group",
        "metric": metric,
        "groups": len(groups),
        "samples": samples,
        "seed": seed,
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
    }
