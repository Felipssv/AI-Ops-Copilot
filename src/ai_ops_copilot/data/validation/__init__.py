"""Submódulo de validação de qualidade de dados de pedidos."""

from ai_ops_copilot.data.validation.validator import (
    MAX_INVALID_RATE_THRESHOLD,
    MIN_VALID_RECORDS_THRESHOLD,
    ValidationOutcome,
    validate_orders_dataframe,
)

__all__ = [
    "MAX_INVALID_RATE_THRESHOLD",
    "MIN_VALID_RECORDS_THRESHOLD",
    "ValidationOutcome",
    "validate_orders_dataframe",
]

