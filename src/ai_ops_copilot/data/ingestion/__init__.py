"""Submódulo de ingestão de arquivos CSV."""

from ai_ops_copilot.data.ingestion.csv_loader import (
    compute_file_sha256,
    load_orders_csv,
)

__all__ = [
    "compute_file_sha256",
    "load_orders_csv",
]
