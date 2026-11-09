from enum import StrEnum

from pydantic import BaseModel, Field



class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ConfusionMatrixSummary(BaseModel):
    true_negatives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_positives: int = Field(ge=0)

class ClassificationMetrics(BaseModel):
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    pr_auc: float = Field(ge=0.0, le=1.0)
    confusion_matrix: ConfusionMatrixSummary

class ModelSplitEvaluation(BaseModel):
    model_name: str
    train_metrics: ClassificationMetrics
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics

class ExperimentReport(BaseModel):
    experiment_id: str
    dataset_id: str
    random_seed: int
    selected_model_name: str
    baseline_evaluation: ModelSplitEvaluation
    candidate_evaluations: list[ModelSplitEvaluation]
    feature_columns: list[str]
    model_artifact_path: str | None = None


class DelayPredictionResponse(BaseModel):
    model_id: str
    model_version: str
    is_late: bool
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel

def classify_risk_level(probability: float) -> RiskLevel:
    if probability < 0.35:
        return RiskLevel.LOW
    elif 0.35 <= probability < 0.65:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.HIGH