# Extracted from the private AI Camera Coach repository and adapted to run standalone.
from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any

RELATIONS = ("BEST>GOOD", "BEST>OTHER", "GOOD>OTHER")
LABELS = ("A", "B")
SOURCE_BOOTSTRAP_SAMPLES = 10_000

def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _confidence_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("confidence summary requires values")
    return {
        "min": min(values),
        "p10": _quantile(values, 0.10),
        "p25": _quantile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "max": max(values),
    }

def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def prediction_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize RGB pair accuracy, group-macro accuracy, loss, and swap consistency."""
    if not predictions:
        raise ValueError("metrics require predictions")
    losses = []
    confusion: Counter[str] = Counter()
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        probability = min(max(float(prediction["probability_a"]), 1e-12), 1.0 - 1e-12)
        target = float(prediction["target_probability_a"])
        losses.append(-target * math.log(probability) - (1.0 - target) * math.log(1.0 - probability))
        confusion[
            f"human_{prediction['human_label']}_model_{prediction['predicted_label']}"
        ] += 1
        group_rows[prediction["group_id"]].append(prediction)
        relation_rows[prediction["relation"]].append(prediction)
    recalls = {}
    for label in LABELS:
        total = sum(
            count for key, count in confusion.items() if key.startswith(f"human_{label}_")
        )
        recalls[label] = confusion[f"human_{label}_model_{label}"] / total if total else None
    correct = sum(bool(prediction["correct"]) for prediction in predictions)
    per_group = [
        {
            "group_id": group_id,
            "pairs": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "bce_log_loss": statistics.fmean(
                -float(row["target_probability_a"])
                * math.log(min(max(float(row["probability_a"]), 1e-12), 1.0 - 1e-12))
                - (1.0 - float(row["target_probability_a"]))
                * math.log(
                    1.0 - min(max(float(row["probability_a"]), 1e-12), 1.0 - 1e-12)
                )
                for row in rows
            ),
        }
        for group_id, rows in sorted(group_rows.items())
    ]
    relation_metrics = {}
    for relation in RELATIONS:
        rows = relation_rows.get(relation, [])
        if not rows:
            relation_metrics[relation] = {
                "pairs": 0,
                "groups": 0,
                "pairwise_accuracy": None,
                "group_macro_accuracy": None,
            }
            continue
        relation_groups: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            relation_groups[row["group_id"]].append(bool(row["correct"]))
        relation_metrics[relation] = {
            "pairs": len(rows),
            "groups": len(relation_groups),
            "pairwise_accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "group_macro_accuracy": statistics.fmean(
                sum(outcomes) / len(outcomes) for outcomes in relation_groups.values()
            ),
        }
    confidences = [float(prediction["confidence"]) for prediction in predictions]
    return {
        "pairs": len(predictions),
        "bce_log_loss": statistics.fmean(losses),
        "pairwise_accuracy": correct / len(predictions),
        "balanced_accuracy": statistics.fmean(
            value for value in recalls.values() if value is not None
        )
        if all(value is not None for value in recalls.values())
        else None,
        "recall_a": recalls["A"],
        "recall_b": recalls["B"],
        "confusion": dict(sorted(confusion.items())),
        "group_level": {
            "groups": len(per_group),
            "macro_accuracy": statistics.fmean(row["accuracy"] for row in per_group),
            "macro_bce_log_loss": statistics.fmean(row["bce_log_loss"] for row in per_group),
            "min_accuracy": min(row["accuracy"] for row in per_group),
            "max_accuracy": max(row["accuracy"] for row in per_group),
            "per_group": per_group,
        },
        "relation": relation_metrics,
        "confidence": _confidence_summary(confidences),
        "swap_consistency": {
            "pairs": len(predictions),
            "max_abs_logit_sum_error": max(
                float(row["swap_logit_sum_abs_error"]) for row in predictions
            ),
            "max_abs_probability_complement_error": max(
                float(row["swap_probability_complement_abs_error"])
                for row in predictions
            ),
            "label_flip_consistency": sum(
                bool(row["swap_label_flip_consistent"]) for row in predictions
            )
            / len(predictions),
            "correctness_consistency": sum(
                bool(row["correct"]) == bool(row["swapped_correct"])
                for row in predictions
            )
            / len(predictions),
        },
    }

def source_bootstrap_accuracy_interval(
    predictions: list[dict[str, Any]], *, seed: int, samples: int = SOURCE_BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    """Resamples shooting groups (source = shooting group in this dataset).
    Return a seeded percentile 95% interval for non-tie group-macro accuracy.
    """
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    source_outcomes: dict[str, list[bool]] = defaultdict(list)
    for prediction in predictions:
        if prediction["human_label"] != "TIE":
            source_outcomes[prediction["group_id"]].append(bool(prediction["correct"]))
    source_accuracies = [
        sum(outcomes) / len(outcomes)
        for _, outcomes in sorted(source_outcomes.items())
        if outcomes
    ]
    if not source_accuracies:
        raise ValueError("source bootstrap requires non-tie predictions")
    generator = random.Random(seed)
    estimates = [
        statistics.fmean(generator.choice(source_accuracies) for _ in source_accuracies)
        for _ in range(samples)
    ]
    return {
        "method": "percentile-source-resampling-macro-accuracy-v1",
        "confidence": 0.95,
        "seed": seed,
        "samples": samples,
        "sources": len(source_accuracies),
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
    }
