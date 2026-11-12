from dataclasses import dataclass
from uuid import uuid4

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from ai_ops_copilot.domain.ml.models import ClassificationMetrics
from ai_ops_copilot.domain.orders import ALLOWED_ML_FEATURES
from ai_ops_copilot.domain.ml import ExperimentReport, ModelSplitEvaluation

from ai_ops_copilot.ml.evaluation import compute_classification_metrics

from ai_ops_copilot.ml.preprocessing import DatasetSplits, build_preprocessor, split_dataset

import pandas as pd

@dataclass(frozen=True)
class TrainingResult:
    report: ExperimentReport
    best_pipeline: Pipeline

def _evaluate_single_split(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> ClassificationMetrics:
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:,1]
    return compute_classification_metrics(y, y_pred, y_prob)

def _evaluate_pipeline(model_name: str, pipeline: Pipeline, splits: DatasetSplits) -> ModelSplitEvaluation:
    train_metrics = _evaluate_single_split(pipeline, splits.X_train, splits.y_train)
    validation_metrics = _evaluate_single_split(pipeline, splits.X_val, splits.y_val)
    test_metrics = _evaluate_single_split(pipeline, splits.X_test, splits.y_test)

    return ModelSplitEvaluation(model_name=model_name, train_metrics=train_metrics, validation_metrics=validation_metrics, test_metrics=test_metrics)

def run_training_experiment(df: pd.DataFrame, dataset_id: str, random_seed: int = 42, experiment_id: str | None = None) -> TrainingResult:
    resolved_id = experiment_id or f"exp_{uuid4().hex[:12]}"
    splits = split_dataset(df, random_seed=random_seed)

    baseline_pipe = Pipeline(
        steps=[("preprocessor", build_preprocessor()), 
               ("classifier", DummyClassifier(strategy="prior", random_state=random_seed))]).fit(
                   splits.X_train, splits.y_train
               )

    baseline_eval = _evaluate_pipeline("dummy_baseline", baseline_pipe, splits)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=random_seed),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=10,random_state=random_seed,n_jobs=1),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=80,max_depth=6,random_state=random_seed)
    }

    candidate_evaluations = []
    trained_pipelines = {}

    for name, estimator in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]).fit(
            splits.X_train, splits.y_train
        )

        candidate_evaluations.append(_evaluate_pipeline(name, pipe, splits))

        trained_pipelines[name] = pipe

    best_eval = max(candidate_evaluations, key=lambda item: (item.validation_metrics.f1_score, 
                                                             item.validation_metrics.pr_auc))

    report = ExperimentReport(
        experiment_id=resolved_id, 
        dataset_id=dataset_id, 
        random_seed=random_seed, 
        selected_model_name=best_eval.model_name, 
        baseline_evaluation=baseline_eval, 
        candidate_evaluations=candidate_evaluations, 
        feature_columns=list(ALLOWED_ML_FEATURES)
        )

    return TrainingResult(report=report, best_pipeline=trained_pipelines[best_eval.model_name])




