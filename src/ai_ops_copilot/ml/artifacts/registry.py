from pathlib import Path

import joblib

from sklearn.pipeline import Pipeline

from ai_ops_copilot.domain.exceptions import DatasetNotFoundError

def save_model_pipeline(pipeline: Pipeline, output_dir: Path | str, model_id: str)-> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    artifact_path = path / f"{model_id}.joblib"

    joblib.dump(pipeline, artifact_path)

    return artifact_path

def load_model_pipeline(artifact_path: Path | str)-> Pipeline:
    path = Path(artifact_path)

    if not path.exists():
        raise DatasetNotFoundError(path.name)

    return joblib.load(path)