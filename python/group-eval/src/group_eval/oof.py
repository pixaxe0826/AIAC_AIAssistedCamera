# Extracted from the private AI Camera Coach repository and adapted to run standalone.
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


def validate_oof_test_sets(
    fold_splits: Sequence[tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
    records: Sequence[dict[str, Any]],
) -> None:
    test_occurrences = Counter(
        record["pair_id"] for _, _, test_records in fold_splits for record in test_records
    )
    if set(test_occurrences) != {record["pair_id"] for record in records} or any(
        count != 1 for count in test_occurrences.values()
    ):
        raise RuntimeError("each pair must occur exactly once in an out-of-fold test set")


def aggregate_oof(
    fold_predictions_by_fold: Sequence[list[dict[str, Any]]],
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    oof_predictions = []
    for fold_predictions in fold_predictions_by_fold:
        oof_predictions.extend(fold_predictions)
    if len(oof_predictions) != len(records) or len(
        {prediction["pair_id"] for prediction in oof_predictions}
    ) != len(records):
        raise RuntimeError("OOF predictions are incomplete or duplicated")
    oof_predictions.sort(key=lambda row: row["pair_id"])
    return oof_predictions
