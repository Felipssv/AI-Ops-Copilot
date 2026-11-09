"""Submódulo de métricas operacionais determinísticas e enriquecimento de pedidos."""

from ai_ops_copilot.data.analytics.metrics import (
    compare_groups,
    compute_late_rate_summary,
    compute_overview_metrics,
    enrich_orders_features,
)

__all__ = [
    "compare_groups",
    "compute_late_rate_summary",
    "compute_overview_metrics",
    "enrich_orders_features",
]

