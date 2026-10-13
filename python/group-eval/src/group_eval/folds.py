# Extracted from the private AI Camera Coach repository and adapted to run standalone.
from __future__ import annotations

import hashlib
from typing import Any

RELATIONS = ("BEST>GOOD", "BEST>OTHER", "GOOD>OTHER")
LABELS = ("A", "B")


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

def build_group_folds(
    profiles: dict[str, dict[str, Any]], *, folds: int, seed: int
) -> dict[str, int]:
    if folds < 2 or len(profiles) < folds:
        raise ValueError("fold assignment requires at least two folds and enough groups")
    count = len(profiles)
    capacities = [count // folds + int(index < count % folds) for index in range(folds)]
    dimensions = [
        "pairs",
        "images",
        "usable_images",
        *(f"relation:{relation}" for relation in RELATIONS),
        *(f"label:{label}" for label in LABELS),
    ]

    def vector(profile: dict[str, Any]) -> dict[str, float]:
        values = {
            "pairs": float(profile["pairs"]),
            "images": float(profile["images"]),
            "usable_images": float(profile["usable_images"]),
        }
        values.update(
            {f"relation:{relation}": float(profile["relations"][relation]) for relation in RELATIONS}
        )
        values.update(
            {f"label:{label}": float(profile["labels"][label]) for label in LABELS}
        )
        return values

    vectors = {group_id: vector(profile) for group_id, profile in profiles.items()}
    totals = {
        dimension: sum(values[dimension] for values in vectors.values())
        for dimension in dimensions
    }
    loads = [{dimension: 0.0 for dimension in dimensions} for _ in range(folds)]
    group_counts = [0] * folds
    assignment: dict[str, int] = {}
    ordered = sorted(
        profiles,
        key=lambda group_id: (
            -profiles[group_id]["pairs"],
            -max(profiles[group_id]["relations"].values()),
            _stable_digest("fold-order-v1", str(seed), group_id),
        ),
    )
    for group_id in ordered:
        candidates: list[tuple[float, str, int]] = []
        for fold in range(folds):
            if group_counts[fold] >= capacities[fold]:
                continue
            proportion = capacities[fold] / count
            score = ((group_counts[fold] + 1) / capacities[fold]) ** 2
            for dimension in dimensions:
                target = totals[dimension] * proportion
                after = loads[fold][dimension] + vectors[group_id][dimension]
                score += (after / max(target, 1.0)) ** 2
            candidates.append(
                (
                    score,
                    _stable_digest("fold-choice-v1", str(seed), group_id, str(fold)),
                    fold,
                )
            )
        if not candidates:
            raise RuntimeError("fold assignment exhausted all capacities")
        fold = min(candidates)[2]
        assignment[group_id] = fold
        group_counts[fold] += 1
        for dimension in dimensions:
            loads[fold][dimension] += vectors[group_id][dimension]
    if sorted(group_counts, reverse=True) != sorted(capacities, reverse=True):
        raise RuntimeError("fold assignment did not meet group capacities")
    return assignment


def _profile_vector(profile: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(profile["pairs"]),
        float(profile["images"]),
        float(profile["usable_images"]),
        *(float(profile["relations"][relation]) for relation in RELATIONS),
        *(float(profile["labels"][label]) for label in LABELS),
    )


def select_validation_groups(
    group_ids: list[str],
    profiles: dict[str, dict[str, Any]],
    *,
    count: int,
    seed: int,
    test_fold: int,
) -> set[str]:
    if not 0 < count < len(group_ids):
        raise ValueError("validation group count must leave training groups")
    ordered = sorted(
        group_ids,
        key=lambda group_id: _stable_digest(
            "validation-start-v1", str(seed), str(test_fold), group_id
        ),
    )
    selected = set(ordered[:count])
    vectors = {group_id: _profile_vector(profiles[group_id]) for group_id in group_ids}
    totals = tuple(sum(vector[index] for vector in vectors.values()) for index in range(8))
    target = tuple(value * count / len(group_ids) for value in totals)

    def score(items: set[str]) -> float:
        observed = tuple(
            sum(vectors[group_id][index] for group_id in items) for index in range(8)
        )
        return sum(
            ((actual - expected) / max(expected, 1.0)) ** 2
            for actual, expected in zip(observed, target)
        )

    current = score(selected)
    for _ in range(1_000):
        best: tuple[float, str, str] | None = None
        unselected = set(group_ids) - selected
        for outgoing in sorted(selected):
            for incoming in sorted(unselected):
                candidate = set(selected)
                candidate.remove(outgoing)
                candidate.add(incoming)
                candidate_score = score(candidate)
                key = (candidate_score, outgoing, incoming)
                if best is None or key < best:
                    best = key
        if best is None or best[0] >= current - 1e-12:
            break
        selected.remove(best[1])
        selected.add(best[2])
        current = best[0]
    return selected


def split_records_for_fold(
    records: list[dict[str, Any]],
    assignment: dict[str, int],
    profiles: dict[str, dict[str, Any]],
    *,
    fold: int,
    validation_groups: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    test_groups = {group_id for group_id, value in assignment.items() if value == fold}
    remaining = sorted(set(assignment) - test_groups)
    val_groups = select_validation_groups(
        remaining,
        profiles,
        count=validation_groups,
        seed=seed,
        test_fold=fold,
    )
    train_groups = set(assignment) - test_groups - val_groups
    if test_groups & val_groups or test_groups & train_groups or val_groups & train_groups:
        raise RuntimeError("group leakage occurred while constructing a CV fold")
    by_partition = {"train": train_groups, "val": val_groups, "test": test_groups}
    output = []
    for partition in ("train", "val", "test"):
        output.append(
            [
                record
                for record in records
                if record["source"]["group_id"] in by_partition[partition]
            ]
        )
    return output[0], output[1], output[2], val_groups
