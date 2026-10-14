# Extracted from the private AI Camera Coach repository and adapted to run standalone.
from .folds import build_group_folds, select_validation_groups, split_records_for_fold
from .metrics import group_bootstrap_interval, prediction_metrics
from .oof import aggregate_oof, validate_oof_test_sets
from .permutation import PermutationData, permutation_summary, permute_group_tiers
from .rgb import prediction_metrics as rgb_prediction_metrics
from .rgb import source_bootstrap_accuracy_interval

__all__ = [
    "PermutationData",
    "aggregate_oof",
    "build_group_folds",
    "group_bootstrap_interval",
    "permutation_summary",
    "permute_group_tiers",
    "prediction_metrics",
    "rgb_prediction_metrics",
    "select_validation_groups",
    "source_bootstrap_accuracy_interval",
    "split_records_for_fold",
    "validate_oof_test_sets",
]
