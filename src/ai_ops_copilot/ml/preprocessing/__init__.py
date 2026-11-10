"""Submódulo de pré-processamento e particionamento sem data leakage."""

from ai_ops_copilot.ml.preprocessing.pipeline import (
    CATEGORICAL_ML_FEATURES,
    NUMERIC_ML_FEATURES,
    DatasetSplits,
    build_preprocessor,
    extract_features_and_target,
    split_dataset,
)

__all__ = [
    "CATEGORICAL_ML_FEATURES",
    "NUMERIC_ML_FEATURES",
    "DatasetSplits",
    "build_preprocessor",
    "extract_features_and_target",
    "split_dataset",
]

