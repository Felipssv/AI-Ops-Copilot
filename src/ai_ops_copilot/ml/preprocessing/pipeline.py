import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_ops_copilot.data.analytics import enrich_orders_features
from ai_ops_copilot.domain.orders import ALLOWED_ML_FEATURES

NUMERIC_ML_FEATURES: tuple[str, ...] = (
    "hour_of_day", "day_of_week", "is_weekend", 
    "order_value", "estimated_prep_minutes", 
    "estimated_delivery_minutes")

CATEGORICAL_ML_FEATURES: tuple[str, ...] = ("region", "order_category")


@dataclass(frozen=True)
class DatasetSplits():
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

def extract_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "hour_of_day" in df.columns:
        enriched = df.copy()
    else:
        enriched = enrich_orders_features(df)

    X = enriched[list(ALLOWED_ML_FEATURES)].copy()

    X["is_weekend"] = X["is_weekend"].astype(int)
    y = enriched["is_late"].astype(int)

    return (X,y)

def split_dataset(df: pd.DataFrame, random_seed: int = 42) -> DatasetSplits:
    X, y = extract_features_and_target(df)

    X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.30, random_state=random_seed, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp)

    return DatasetSplits(X_train,X_val,X_test,y_train,y_val,y_test)

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), list(NUMERIC_ML_FEATURES)), 
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL_ML_FEATURES))
        ])