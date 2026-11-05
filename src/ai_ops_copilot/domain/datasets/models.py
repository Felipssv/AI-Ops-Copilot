from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class DatasetSourceType(StrEnum):
    SYNTHETIC = "synthetic"
    IMPORTED = "imported"

class DatasetValidationStatus(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    INVALID = "invalid"

class QualityIssueCode(StrEnum):
    MISSING_VALUES = "missing_values"
    DUPLICATE_ORDER_ID = "duplicate_order_id"
    INVALID_DATETIME = "invalid_datetime"
    UNKNOWN_CATEGORY = "unknown_category"
    OUT_OF_RANGE = "out_of_range"
    INCONSISTENT_LATE_FLAG = "inconsistent_late_flag"

class SyntheticDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number_of_orders: int = Field(default=10000, ge=10, le=500000)
    random_seed: int = Field(default=42, ge=0)
    late_tolerance_minutes: int = Field(default=5, ge=0, le=60)
    noise_level: float = Field(default=1.0, ge=0.0, le=5.0)
    inject_quality_issues_rate: float = Field(default=0.0, ge=0.0, le=0.5)


class DatasetGenerationResult(BaseModel):
    dataset_id: str
    status: str = "created"
    records: int
    random_seed: int
    late_tolerance_minutes: int
    file_path: str | None = None
    file_hash: str | None = None

class QualityWarning(BaseModel):
    field: str
    issue: QualityIssueCode
    count: int = Field(..., ge=1)

class DatasetQualityReport(BaseModel):
    dataset_id: str
    total_records: int = Field(..., ge=0)
    valid_records: int = Field(..., ge=0)
    invalid_records: int = Field(..., ge=0)
    warnings: list[QualityWarning] = Field(default_factory=list)
    status: Literal["completed", "failed"] = "completed"


GroupByDimension = Literal["hour_of_day", "day_of_week", "is_weekend", "region", "order_category"]

AnalyticsMetric = Literal["late_rate", "order_count","avg_delivery_minutes", "avg_prep_minutes", 
                         "avg_order_value", "avg_delay_minutes"]

class OverviewMetrics(BaseModel):
    dataset_id: str
    total_orders: int
    late_orders: int
    on_time_orders: int
    late_rate: float
    avg_order_value: float
    avg_estimated_prep_minutes: float
    avg_actual_prep_minutes: float
    avg_estimated_delivery_minutes: float
    avg_actual_delivery_minutes: float
    avg_promised_total_minutes: float
    avg_actual_total_minutes: float
    avg_delay_minutes: float

class LateRateSummary(BaseModel):
    dataset_id: str
    total_orders: int
    late_orders: int
    on_time_orders: int
    late_rate: float
    avg_delay_when_late_minutes: float
    max_delay_minutes: float

class GroupMetricItem(BaseModel):
    group: str
    value: float
    sample_count: int

class GroupComparisonResponse(BaseModel):
    dataset_id: str
    group_by: GroupByDimension
    metric: AnalyticsMetric
    results: list[GroupMetricItem]