"""Submódulo de geração determinística de dados sintéticos."""

from ai_ops_copilot.data.generator.synthetic_generator import (
    generate_synthetic_dataset,
    generate_synthetic_orders_df,
)

__all__ = [
    "generate_synthetic_dataset",
    "generate_synthetic_orders_df",
]

