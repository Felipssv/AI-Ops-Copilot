from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_ops_copilot.data.generator import generate_synthetic_orders_df
from ai_ops_copilot.domain.datasets import SyntheticDatasetConfig
from ai_ops_copilot.domain.exceptions import DatasetNotFoundError
from ai_ops_copilot.domain.ml import RiskLevel
from ai_ops_copilot.domain.orders import (
    ALLOWED_ML_FEATURES,
    FORBIDDEN_LEAKAGE_COLUMNS,
    OrderPredictionFeatures,
)
from ai_ops_copilot.ml.artifacts import load_model_pipeline, save_model_pipeline
from ai_ops_copilot.ml.inference import predict_order_delay_risk
from ai_ops_copilot.ml.preprocessing import extract_features_and_target, split_dataset
from ai_ops_copilot.ml.training import run_training_experiment


def test_ca_ml_01_no_data_leakage() -> None:
    df = generate_synthetic_orders_df(
        SyntheticDatasetConfig(number_of_orders=400, random_seed=42)
    )
    X, _ = extract_features_and_target(df)

    assert list(X.columns) == list(ALLOWED_ML_FEATURES)
    assert FORBIDDEN_LEAKAGE_COLUMNS.isdisjoint(X.columns)

    df_altered = df.copy()
    df_altered["actual_prep_minutes"] = 999
    df_altered["actual_delivery_minutes"] = 999
    X_altered, _ = extract_features_and_target(df_altered)
    assert X.equals(X_altered)

    splits = split_dataset(df, random_seed=42)
    assert len(splits.X_train) == 280
    assert len(splits.X_val) == 60
    assert len(splits.X_test) == 60
    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == 400


def test_ca_ml_02_baseline_vs_candidates_experiment() -> None:
    df = generate_synthetic_orders_df(
        SyntheticDatasetConfig(number_of_orders=600, random_seed=42)
    )
    res = run_training_experiment(df, dataset_id="ds_ml", random_seed=42)

    assert res.report.baseline_evaluation.model_name == "dummy_baseline"
    assert len(res.report.candidate_evaluations) == 3
    assert res.report.selected_model_name in {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }

    selected_eval = next(
        item
        for item in res.report.candidate_evaluations
        if item.model_name == res.report.selected_model_name
    )
    assert (
        selected_eval.test_metrics.pr_auc
        > res.report.baseline_evaluation.test_metrics.pr_auc
    )


def test_ca_ml_03_artifact_persistence_and_inference_contract(tmp_path: Path) -> None:
    df = generate_synthetic_orders_df(
        SyntheticDatasetConfig(number_of_orders=300, random_seed=42)
    )
    res = run_training_experiment(df, dataset_id="ds_inf", random_seed=42)

    artifact_path = save_model_pipeline(res.best_pipeline, tmp_path, "model_01")
    loaded_pipeline = load_model_pipeline(artifact_path)

    features = OrderPredictionFeatures(
        hour_of_day=19,
        day_of_week=5,
        region="central",
        order_category="restaurant",
        order_value=140.0,
        estimated_prep_minutes=25,
        estimated_delivery_minutes=25,
    )
    pred = predict_order_delay_risk(loaded_pipeline, features, model_id="model_01")

    assert pred.model_id == "model_01"
    assert 0.0 <= pred.risk_probability <= 1.0
    assert pred.risk_level in {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH}

    with pytest.raises(ValidationError):
        OrderPredictionFeatures(
            hour_of_day=19,
            day_of_week=5,
            region="central",
            order_category="restaurant",
            order_value=140.0,
            estimated_prep_minutes=25,
            estimated_delivery_minutes=25,
            actual_prep_minutes=30,
        )

    with pytest.raises(DatasetNotFoundError):
        load_model_pipeline(tmp_path / "inexistente.joblib")
