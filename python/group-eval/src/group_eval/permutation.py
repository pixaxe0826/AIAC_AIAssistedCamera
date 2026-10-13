# Extracted from the private AI Camera Coach repository and adapted to run standalone.
from __future__ import annotations

import hashlib
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any

RANK_ORDER = {"BEST": 3, "GOOD": 2, "OTHER": 1, "EXCLUDE": 0}

def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class PermutationData:
    image_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    ranking_labels: tuple[str, ...]
    statuses: tuple[str, ...]
    pairs: tuple[dict[str, Any], ...]

def permute_group_tiers(data: PermutationData, *, seed: int) -> PermutationData:
    eligible_groups = {pair["group_id"] for pair in data.pairs}
    labels = list(data.ranking_labels)
    rng = random.Random(seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (group_id, status, label) in enumerate(
        zip(data.group_ids, data.statuses, data.ranking_labels, strict=True)
    ):
        if group_id in eligible_groups and status == "EXACT_ONE" and label in {"BEST", "GOOD", "OTHER"}:
            grouped[group_id].append(index)
    before_counts = {
        group_id: Counter(labels[index] for index in indices)
        for group_id, indices in grouped.items()
    }
    for group_id in sorted(grouped):
        indices = sorted(grouped[group_id], key=lambda index: data.image_ids[index])
        shuffled = [labels[index] for index in indices]
        rng.shuffle(shuffled)
        for index, label in zip(indices, shuffled, strict=True):
            labels[index] = label
    after_counts = {
        group_id: Counter(labels[index] for index in indices)
        for group_id, indices in grouped.items()
    }
    if before_counts != after_counts:
        raise RuntimeError("tier permutation did not preserve per-Group tier counts")
    pairs: list[dict[str, Any]] = []
    for group_id, indices in sorted(grouped.items()):
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                left_label = labels[left]
                right_label = labels[right]
                if left_label == right_label:
                    continue
                orientation = int(
                    _stable_digest(
                        "permutation-orientation-v1",
                        data.image_ids[left],
                        data.image_ids[right],
                    )[:8],
                    16,
                ) % 2
                a_index, b_index = (left, right) if orientation == 0 else (right, left)
                label_a = labels[a_index]
                label_b = labels[b_index]
                target = 1.0 if RANK_ORDER[label_a] > RANK_ORDER[label_b] else 0.0
                relation = ">".join(
                    sorted(
                        (label_a, label_b),
                        key=lambda value: RANK_ORDER[value],
                        reverse=True,
                    )
                )
                pair_id = "geometry-permutation:" + _stable_digest(
                    str(seed), data.image_ids[a_index], data.image_ids[b_index]
                )[:32]
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "group_id": group_id,
                        "a_index": a_index,
                        "b_index": b_index,
                        "target": target,
                        "human_label": "A" if target == 1.0 else "B",
                        "relation": relation,
                    }
                )
    if len(pairs) != len(data.pairs):
        raise RuntimeError(
            f"tier permutation changed pair count: {len(data.pairs)} -> {len(pairs)}"
        )
    if Counter(pair["relation"] for pair in pairs) != Counter(
        pair["relation"] for pair in data.pairs
    ):
        raise RuntimeError("tier permutation changed relation counts")
    return replace(data, ranking_labels=tuple(labels), pairs=tuple(pairs))


def permutation_summary(
    completed_by_index: dict[int, dict[str, Any]],
    *,
    target: int,
    observed: float,
    metric_name: str = "group_macro_pair_accuracy",
) -> dict[str, Any]:
    rows = [completed_by_index[index] for index in range(target)]
    null_values = [float(row[metric_name]) for row in rows]
    greater_or_equal = sum(value >= observed for value in null_values)
    p_value = (greater_or_equal + 1) / (target + 1)
    percentile = 100.0 * sum(value < observed for value in null_values) / target
    return {
        "stage": "Group-internal tier-label permutation test",
        "primary_metric": metric_name,
        "observed": observed,
        "permutations": target,
        "null_mean": statistics.fmean(null_values),
        "null_std_population": statistics.pstdev(null_values),
        "null_min": min(null_values),
        "null_max": max(null_values),
        "empirical_p_value_one_sided": p_value,
        "observed_percentile": percentile,
        "greater_or_equal_null_trials": greater_or_equal,
        "permutation_definition": (
            "BEST/GOOD/OTHER labels are shuffled within each Group while each Group's tier "
            "counts, image features, fold assignment, and total relation counts are preserved."
        ),
    }
