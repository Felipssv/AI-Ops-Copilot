"""Submódulo de treinamento comparativo entre baseline e modelos candidatos."""

from ai_ops_copilot.ml.training.trainer import (
    TrainingResult,
    run_training_experiment,
)

__all__ = [
    "TrainingResult",
    "run_training_experiment",
]
