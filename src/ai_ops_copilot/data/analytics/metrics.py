import pandas as pd

from ai_ops_copilot.domain.datasets import (
    AnalyticsMetric,
    GroupByDimension,
    GroupComparisonResponse,
    GroupMetricItem,
    LateRateSummary,
    OverviewMetrics,
)

def enrich_orders_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    dt = pd.to_datetime(enriched["created_at"])

    enriched["hour_of_day"] = dt.dt.hour
    enriched["day_of_week"] = dt.dt.dayofweek
    enriched["is_weekend"] = enriched["day_of_week"].isin([5,6])

    enriched["promised_total_minutes"] = (
        enriched["estimated_prep_minutes"] + 
        enriched["estimated_delivery_minutes"]
    )

    enriched["actual_total_minutes"] = (
        enriched["actual_prep_minutes"] +
        enriched["actual_delivery_minutes"]
    )

    enriched["delay_minutes"] = (
        enriched["actual_total_minutes"] -
        enriched ["promised_total_minutes"]
    )

    return enriched

def compute_overview_metrics(df: pd.DataFrame, dataset_id: str) -> OverviewMetrics:
    enriched = enrich_orders_features(df)
    total_orders = len(enriched)
    is_late_series = enriched["is_late"].astype(bool)
    late_orders = int(is_late_series.sum())
    on_time_orders = total_orders - late_orders

    return OverviewMetrics(
        dataset_id=dataset_id,
        total_orders=total_orders,
        late_orders=late_orders,
        on_time_orders=on_time_orders,
        late_rate=round(float(is_late_series.mean()), 4),
        avg_order_value=round(float(enriched["order_value"].mean()), 4),
        avg_estimated_prep_minutes=round(float(enriched["estimated_prep_minutes"].mean()), 4),
        avg_actual_prep_minutes=round(float(enriched["actual_prep_minutes"].mean()), 4),
        avg_estimated_delivery_minutes=round(float(enriched["estimated_delivery_minutes"].mean()), 4),
        avg_actual_delivery_minutes=round(float(enriched["actual_delivery_minutes"].mean()), 4),
        avg_promised_total_minutes=round(float(enriched["promised_total_minutes"].mean()), 4),
        avg_actual_total_minutes=round(float(enriched["actual_total_minutes"].mean()), 4),
        avg_delay_minutes=round(float(enriched["delay_minutes"].mean()), 4),
    )

def compute_late_rate_summary(df: pd.DataFrame, dataset_id: str) -> LateRateSummary:
    enriched = enrich_orders_features(df)
    total_orders = len(enriched)
    is_late_series = enriched["is_late"].astype(bool)
    late_orders = int(is_late_series.sum())
    on_time_orders = total_orders - late_orders

    late_df = enriched.loc[is_late_series]
    avg_delay_when_late = (
        round(float(late_df["delay_minutes"].mean()), 4) if late_orders > 0 else 0.0
    )
    max_delay = round(float(enriched["delay_minutes"].max()), 4) if total_orders > 0 else 0.0
    return LateRateSummary(
        dataset_id=dataset_id,
        total_orders=total_orders,
        late_orders=late_orders,
        on_time_orders=on_time_orders,
        late_rate=round(float(is_late_series.mean()), 4),
        avg_delay_when_late_minutes=avg_delay_when_late,
        max_delay_minutes=max_delay,
    )

def compare_groups(
    df: pd.DataFrame,
    dataset_id: str,
    group_by: GroupByDimension,
    metric: AnalyticsMetric,
) -> GroupComparisonResponse:
    enriched = enrich_orders_features(df)
    metric_column_map: dict[str, str] = {
        "late_rate": "is_late",
        "avg_delivery_minutes": "actual_delivery_minutes",
        "avg_prep_minutes": "actual_prep_minutes",
        "avg_order_value": "order_value",
        "avg_delay_minutes": "delay_minutes",
    }

    items: list[GroupMetricItem] = []
    for group_key, group_df in enriched.groupby(group_by, dropna=False):
        sample_count = len(group_df)
        if metric == "order_count":
            val = float(sample_count)
        else:
            target_col = metric_column_map[metric]
            val = round(float(group_df[target_col].mean()), 4)

        items.append(
            GroupMetricItem(
                group=str(group_key),
                value=val,
                sample_count=sample_count,
            )
        )

    # Ordena do maior valor para o menor (ex.: região com maior atraso no topo!)
    items.sort(key=lambda item: item.value, reverse=True)

    return GroupComparisonResponse(
        dataset_id=dataset_id,
        group_by=group_by,
        metric=metric,
        results=items,
    )
