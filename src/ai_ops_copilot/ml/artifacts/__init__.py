"""Submódulo de serialização e carregamento de artefatos de modelos em disco."""

from ai_ops_copilot.ml.artifacts.registry import (
    load_model_pipeline,
    save_model_pipeline,
)

__all__ = [
    "load_model_pipeline",
    "save_model_pipeline",
]
