# Extracted from the private AI Camera Coach repository and adapted to run standalone.
"""Synthetic-only tests for the extracted evaluation code."""
from collections import Counter

import numpy as np
import pytest

from group_eval import (
    PermutationData,
    aggregate_oof,
    build_group_folds,
    group_bootstrap_interval,
    permutation_summary,
    permute_group_tiers,
    prediction_metrics,
    rgb_prediction_metrics,
    source_bootstrap_accuracy_interval,
    split_records_for_fold,
    validate_oof_test_sets,
)


def profiles_and_records():
    profiles = {
        f"synthetic-{index}": {
            "pairs": 3,
            "images": index + 3,
            "usable_images": index + 3,
            "relations": {"BEST>GOOD": 1, "BEST>OTHER": 1, "GOOD>OTHER": 1},
            "labels": {"A": 2, "B": 1},
        }
        for index in range(10)
    }
    records = [
        {"pair_id": f"{group}-{index}", "source": {"group_id": group}}
        for group in profiles
        for index in range(3)
    ]
    return profiles, records


def prediction(group, index, correct=True):
    human = "A" if index % 2 == 0 else "B"
    predicted = human if correct else ("B" if human == "A" else "A")
    return {
        "pair_id": f"{group}-{index}",
        "group_id": group,
        "relation": "BEST>GOOD",
        "target": 1.0 if human == "A" else 0.0,
        "target_probability_a": 1.0 if human == "A" else 0.0,
        "probability_a": 0.9 if predicted == "A" else 0.1,
        "human_label": human,
        "predicted_label": predicted,
        "correct": correct,
        "confidence": 0.8,
        "swap_logit_sum_abs_error": 0.0,
        "swap_probability_complement_abs_error": 0.0,
        "swap_label_flip_consistent": True,
        "swapped_correct": correct,
    }


def unequal_groups():
    return [prediction("large", index) for index in range(9)] + [
        prediction("small", 0, correct=False)
    ]


def test_seeded_folds_are_deterministic_and_order_independent():
    profiles, _ = profiles_and_records()
    first = build_group_folds(profiles, folds=5, seed=17)
    second = build_group_folds(dict(reversed(list(profiles.items()))), folds=5, seed=17)
    assert first == second
    assert Counter(first.values()) == {fold: 2 for fold in range(5)}


def test_train_validation_and_test_groups_do_not_overlap():
    profiles, records = profiles_and_records()
    assignment = build_group_folds(profiles, folds=5, seed=17)
    fold_splits = []
    for fold in range(5):
        train, validation, test, _ = split_records_for_fold(
            records, assignment, profiles, fold=fold, validation_groups=1, seed=17
        )
        groups = [{row["source"]["group_id"] for row in part}
                  for part in (train, validation, test)]
        assert all(groups)
        assert groups[0].isdisjoint(groups[1])
        assert groups[0].isdisjoint(groups[2])
        assert groups[1].isdisjoint(groups[2])
        fold_splits.append((train, validation, test))
    validate_oof_test_sets(fold_splits, records)


@pytest.mark.parametrize("scorer", [
    lambda rows: prediction_metrics(rows, ece_bins=10, high_confidence=0.8),
    rgb_prediction_metrics,
])
def test_pair_accuracy_and_group_macro_differ_on_hand_worked_example(scorer):
    metrics = scorer(unequal_groups())
    assert metrics["pairwise_accuracy"] == pytest.approx(9 / 10)
    macro = metrics.get("group_macro_pair_accuracy", metrics.get("group_level", {}).get("macro_accuracy"))
    assert macro == pytest.approx(np.mean([1.0, 0.0]))
    assert macro != metrics["pairwise_accuracy"]


@pytest.mark.parametrize("bootstrap", [
    lambda rows: group_bootstrap_interval(rows, metric="accuracy", seed=17, samples=400),
    lambda rows: source_bootstrap_accuracy_interval(rows, seed=17, samples=400),
])
def test_group_ci_is_deterministic_and_contains_this_point_estimate(bootstrap):
    rows = unequal_groups()
    first = bootstrap(rows)
    assert first == bootstrap(rows)
    assert first["lower"] == 0.0
    assert first["upper"] == 1.0
    assert first["lower"] <= 0.5 <= first["upper"]


def test_rgb_bootstrap_excludes_ties():
    rows = unequal_groups()
    tie = {"human_label": "TIE", "group_id": "tie-only", "correct": False}
    assert source_bootstrap_accuracy_interval(rows, seed=17, samples=40) == (
        source_bootstrap_accuracy_interval(rows + [tie], seed=17, samples=40)
    )


def test_oof_aggregation_sorts_complete_predictions():
    rows = [prediction("a", 0), prediction("b", 0), prediction("c", 0)]
    assert aggregate_oof([[rows[2]], [rows[1], rows[0]]], rows) == rows


@pytest.mark.parametrize("folds", [
    [[prediction("a", 0)]],
    [[prediction("a", 0)], [prediction("a", 0)]],
])
def test_oof_aggregation_rejects_missing_or_duplicate_predictions(folds):
    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        aggregate_oof(folds, [prediction("a", 0), prediction("b", 0)])


def test_oof_test_coverage_rejects_repeated_pair():
    row = {"pair_id": "synthetic-pair"}
    with pytest.raises(RuntimeError, match="exactly once"):
        validate_oof_test_sets([([], [], [row]), ([], [], [row])], [row])


def test_tier_permutation_preserves_groups_tier_counts_and_relations():
    relations = ("BEST>GOOD", "BEST>OTHER", "GOOD>OTHER")
    original_pairs = tuple(
        {"group_id": group, "relation": relation}
        for group in ("first", "second")
        for relation in relations
    )
    data = PermutationData(
        image_ids=tuple(f"synthetic-image-{index}" for index in range(6)),
        group_ids=("first",) * 3 + ("second",) * 3,
        ranking_labels=("BEST", "GOOD", "OTHER") * 2,
        statuses=("EXACT_ONE",) * 6,
        pairs=original_pairs,
    )
    first = permute_group_tiers(data, seed=17)
    assert first == permute_group_tiers(data, seed=17)
    assert data.ranking_labels == ("BEST", "GOOD", "OTHER") * 2
    for group in set(data.group_ids):
        assert Counter(label for label, group_id in zip(first.ranking_labels, first.group_ids)
                       if group_id == group) == {"BEST": 1, "GOOD": 1, "OTHER": 1}
    assert Counter(row["relation"] for row in first.pairs) == Counter(
        row["relation"] for row in original_pairs
    )
    for row in first.pairs:
        assert data.group_ids[row["a_index"]] == data.group_ids[row["b_index"]]


def test_permutation_summary_uses_original_plus_one_correction():
    null = {index: {"group_macro_pair_accuracy": score}
            for index, score in enumerate([0.2, 0.4, 0.6, 0.8])}
    summary = permutation_summary(null, target=4, observed=0.6)
    assert summary["greater_or_equal_null_trials"] == 2
    assert summary["empirical_p_value_one_sided"] == pytest.approx(3 / 5)
    assert summary["observed_percentile"] == 50.0


def test_empty_metrics_and_bootstrap_fail_explicitly():
    with pytest.raises(ValueError, match="metrics require predictions"):
        prediction_metrics([], ece_bins=10, high_confidence=0.8)
    with pytest.raises(ValueError, match="bootstrap requires groups"):
        group_bootstrap_interval([], metric="accuracy", seed=17, samples=40)
    with pytest.raises(ValueError, match="bootstrap samples must be positive"):
        source_bootstrap_accuracy_interval(unequal_groups(), seed=17, samples=0)
