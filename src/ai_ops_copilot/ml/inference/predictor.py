import pandas as pd

from sklearn.pipeline import Pipeline

from ai_ops_copilot.domain.ml import DelayPredictionResponse, classify_risk_level

from ai_ops_copilot.domain.orders import ALLOWED_ML_FEATURES, OrderPredictionFeatures

def predict_order_delay_risk(pipeline: Pipeline, features: OrderPredictionFeatures, model_id: str, model_version: str="1.0.0")-> DelayPredictionResponse:
    row = {
        "hour_of_day": features.hour_of_day,
        "day_of_week": features.day_of_week,
        "is_weekend": int(features.is_weekend),
        "region": features.region.value,
        "order_category": features.order_category.value,
        "order_value": features.order_value,
        "estimated_prep_minutes": features.estimated_prep_minutes,
        "estimated_delivery_minutes": features.estimated_delivery_minutes,
    }

    input_df = pd.DataFrame([row])[list(ALLOWED_ML_FEATURES)]

    pred_class = bool(pipeline.predict(input_df)[0])

    prob = round(float(pipeline.predict_proba(input_df)[0, 1]), 4)

    return DelayPredictionResponse(model_id=model_id, model_version=model_version, is_late=pred_class, risk_probability=prob, risk_level=classify_risk_level(prob))