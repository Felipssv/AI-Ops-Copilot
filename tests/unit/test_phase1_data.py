from pathlib import Path

import pandas as pd
import pytest

from ai_ops_copilot.data.analytics import compare_groups, compute_overview_metrics
from ai_ops_copilot.data.generator import (
    generate_synthetic_dataset,
    generate_synthetic_orders_df,
)
from ai_ops_copilot.data.ingestion import load_orders_csv
from ai_ops_copilot.data.validation import validate_orders_dataframe
from ai_ops_copilot.domain.datasets import QualityIssueCode, SyntheticDatasetConfig
from ai_ops_copilot.domain.exceptions import InvalidFileFormatError

def test_ca_data_01_synthetic_reproducibility(tmp_path: Path) -> None:
    config = SyntheticDatasetConfig(
        number_of_orders=500,
        random_seed=42,
        late_tolerance_minutes=5,
    )
    res1 = generate_synthetic_dataset(config, tmp_path / "run1", dataset_id="ds_1")
    res2 = generate_synthetic_dataset(config, tmp_path / "run2", dataset_id="ds_2")

    assert res1.records == 500
    assert res1.file_hash == res2.file_hash

def test_ca_data_02_validation_and_late_rule() -> None:
    config = SyntheticDatasetConfig(
        number_of_orders=500,
        random_seed=42,
        late_tolerance_minutes=5,
        inject_quality_issues_rate=0.04,
    )
    df = generate_synthetic_orders_df(config)
    outcome = validate_orders_dataframe(df, dataset_id="ds_val", late_tolerance_minutes=5)

    assert outcome.report.invalid_records > 0
    assert len(outcome.valid_df) == outcome.report.valid_records

    expected_late = (
        outcome.valid_df["actual_prep_minutes"] + outcome.valid_df["actual_delivery_minutes"]
    ) > (
        outcome.valid_df["estimated_prep_minutes"]
        + outcome.valid_df["estimated_delivery_minutes"]
        + 5
    )
    assert (outcome.valid_df["is_late"] == expected_late).all()

def test_ca_data_03_reject_invalid_csv(tmp_path: Path) -> None:
    with pytest.raises(InvalidFileFormatError):
        load_orders_csv(tmp_path / "nao_existe.csv")

    bad_csv = tmp_path / "incompleto.csv"
    pd.DataFrame({"order_id": ["ORD-1"], "region": ["south"]}).to_csv(bad_csv, index=False)

    with pytest.raises(InvalidFileFormatError):
        load_orders_csv(bad_csv)

def test_ca_data_04_analytics_metrics_fidelity() -> None:
    df = generate_synthetic_orders_df(SyntheticDatasetConfig(number_of_orders=400, random_seed=7))
    overview = compute_overview_metrics(df, dataset_id="ds_metrics")
    by_region = compare_groups(df, dataset_id="ds_metrics", group_by="region", metric="late_rate")

    assert overview.total_orders == 400
    assert overview.late_orders + overview.on_time_orders == 400
    assert abs(overview.late_rate - float(df["is_late"].mean())) < 1e-4
    assert sum(item.sample_count for item in by_region.results) == 400