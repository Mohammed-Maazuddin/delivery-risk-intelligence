
from pathlib import Path

import joblib
import pandas as pd


# Find the project and model folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"


# Load the saved model and preprocessing objects
model = joblib.load(MODEL_DIR / "logistic_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
features = joblib.load(MODEL_DIR / "feature_names.joblib")


def predict_delivery_risk(order_features):
    """Predict whether an order is at risk of late delivery."""

    # Arrange input features in the correct order
    order_data = pd.DataFrame([order_features], columns=features)

    # Scale features using the saved scaler
    order_scaled = scaler.transform(order_data)

    # Predict the probability score for a late delivery
    risk_score = model.predict_proba(order_scaled)[0][1]

    # Classify using the default threshold
    prediction = int(risk_score >= 0.5)

    return {
        "risk_score": round(float(risk_score), 4),
        "prediction": prediction,
        "risk_label": "Late risk" if prediction == 1 else "Lower predicted risk"
    }
