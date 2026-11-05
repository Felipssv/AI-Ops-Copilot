from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ai_ops_copilot.data.ingestion import compute_file_sha256
from ai_ops_copilot.domain.datasets import DatasetGenerationResult, SyntheticDatasetConfig

# Pesos relativos de volume de pedidos para cada hora do dia (0h às 23h)
# Picos claros no almoço (11h-14h) e no jantar (18h-22h)
_HOURLY_WEIGHTS: tuple[float, ...] = (
    1.0, 0.3, 0.2, 0.1, 0.1, 0.2,  # 00h - 05h
    0.3, 1.0, 2.1, 2.5, 3.5, 8.5,  # 06h - 11h
    9.5, 9.0, 6.5, 4.0, 4.2, 4.5,  # 12h - 17h
    8.5, 11.0, 10.5, 8.0, 4.5, 1.5 # 18h - 23h
)

# Mapa: categoria -> (tempo_medio_preparo_min, ticket_medio_reais)
_CATEGORY_PREP_BASE: dict[str, tuple[float, float]] = {
    "restaurant": (26.0, 78.0),
    "fast_food": (15.0, 42.0),
    "market": (30.0, 145.0),
    "pharmacy": (10.0, 55.0),
    "bakery": (12.0, 34.0),
}

# Mapa: regiao -> tempo_medio_entrega_min
_REGION_DELIVERY_BASE: dict[str, float] = {
    "central": 22.0,
    "north": 28.0,
    "south": 24.0,
    "east": 31.0,
    "west": 26.0,
}

def generate_synthetic_orders_df(config: SyntheticDatasetConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    n = config.number_of_orders
    order_ids = [f"ORD-{i:06d}" for i in range(1, n + 1)]

    base_date = datetime(2026, 1, 1, 0, 0, 0)


    days_offset = rng.integers(0, 30, size=n)
    hour_probs = np.array(_HOURLY_WEIGHTS, dtype=float) / np.sum(_HOURLY_WEIGHTS)
    hours = rng.choice(np.arange(24), size=n, p=hour_probs)
    minutes = rng.integers(0, 60, size=n)

    timestamps = [
        base_date + timedelta(days=int(d), hours=int(h), minutes=int(m))
        for d, h, m in zip(days_offset, hours, minutes)
    ]
    created_at = [ts.strftime("%Y-%m-%dT%H:%M:%S") for ts in timestamps]
    weekdays = np.array([ts.weekday() for ts in timestamps])
    is_weekend = weekdays >= 5

    regions = rng.choice(
    ["north", "south", "east", "west", "central"],
    size=n,
    p=[0.22, 0.20, 0.18, 0.15, 0.25],
    )
    categories = rng.choice(
        ["restaurant", "fast_food", "market", "pharmacy", "bakery"],
        size=n,
        p=[0.35, 0.25, 0.18, 0.12, 0.10],
    )

    base_prep = np.array([_CATEGORY_PREP_BASE[c][0] for c in categories])
    base_val = np.array([_CATEGORY_PREP_BASE[c][1] for c in categories])
    base_deliv = np.array([_REGION_DELIVERY_BASE[r] for r in regions])

    order_value = np.round(np.clip(base_val + rng.normal(0.0, 18.0, size=n), 8.0, 600.0), 2)
    estimated_prep = np.clip(np.round(base_prep + rng.normal(0.0, 4.0, size=n)), 5, 90).astype(int)
    estimated_delivery = np.clip(np.round(base_deliv + rng.normal(0.0, 5.0, size=n)), 8, 90).astype(int)



    is_lunch_peak = (hours >= 11) & (hours <= 14)
    is_dinner_peak = (hours >= 18) & (hours <= 22)
    is_peak = is_lunch_peak | is_dinner_peak
    is_complex_cat = np.isin(categories, ["restaurant", "market"])
    is_congested_reg = np.isin(regions, ["central", "east"])

    prep_shift = (
        np.where(is_peak, 2.5, -1.0)
        + np.where(is_complex_cat & is_peak, 3.0, 0.0)
        + np.where(order_value > 130.0, 2.0, 0.0)
        + rng.normal(0.0, 3.5 * config.noise_level, size=n)
    )
    deliv_shift = (
        np.where(is_dinner_peak, 3.5, np.where(is_lunch_peak, 2.0, -1.5))
        + np.where(is_weekend, 2.0, 0.0)
        + np.where(is_congested_reg & is_peak, 3.5, 0.0)
        + rng.normal(0.0, 4.0 * config.noise_level, size=n)
    )

    actual_prep = np.clip(np.round(estimated_prep + prep_shift), 1, 360).astype(int)
    actual_delivery = np.clip(np.round(estimated_delivery + deliv_shift), 1, 360).astype(int)

    promised_total = estimated_prep + estimated_delivery
    actual_total = actual_prep + actual_delivery
    is_late = (actual_total > (promised_total + config.late_tolerance_minutes)).astype(bool)

    df = pd.DataFrame({
        "order_id": order_ids,
        "created_at": created_at,
        "region": regions,
        "order_category": categories,
        "order_value": order_value,
        "estimated_prep_minutes": estimated_prep,
        "estimated_delivery_minutes": estimated_delivery,
        "actual_prep_minutes": actual_prep,
        "actual_delivery_minutes": actual_delivery,
        "is_late": is_late,
        })

    if config.inject_quality_issues_rate > 0:
        num_issues = max(1, int(n * config.inject_quality_issues_rate))
        issue_indices = rng.choice(n, size=num_issues, replace=False)
        df.loc[issue_indices[0::2], "order_value"] = -15.0
        df.loc[issue_indices[1::2], "region"] = "invalid_region"

    return df

def generate_synthetic_dataset(
        config: SyntheticDatasetConfig,
        output_dir: Path | str,
        dataset_id: str | None = None,
) -> DatasetGenerationResult:
    df = generate_synthetic_orders_df(config)
    resolved_id = dataset_id or f"ds_{uuid4().hex[:12]}"

 
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{resolved_id}.csv"

    df.to_csv(file_path, index=False, encoding="utf-8")
    sha256_hash = compute_file_sha256(file_path)

    return DatasetGenerationResult(
        dataset_id=resolved_id,
        records=len(df),
        random_seed=config.random_seed,
        late_tolerance_minutes=config.late_tolerance_minutes,
        file_path=str(file_path),
        file_hash=sha256_hash,
    )

