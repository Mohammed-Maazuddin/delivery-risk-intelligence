
from pathlib import Path

import joblib
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from data_preprocessing import load_data, build_dataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"


def evaluate_model():
    # Load the saved model and preprocessing objects
    model = joblib.load(MODEL_DIR / "logistic_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    features = joblib.load(MODEL_DIR / "feature_names.joblib")

    # Rebuild the dataset
    orders, items = load_data()
    data = build_dataset(orders, items)

    # Select validation-period orders only
    dates = data["order_purchase_timestamp"]

    validation_mask = (
        (dates >= "2018-01-01")
        & (dates < "2018-05-01")
    )

    validation_data = data.loc[validation_mask]

    X_val = validation_data[features]
    y_val = validation_data["late_delivery"]

    # Generate predictions
    X_val_scaled = scaler.transform(X_val)
    probabilities = model.predict_proba(X_val_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Display evaluation results
    print("Validation Average Precision:",
          round(average_precision_score(y_val, probabilities), 4))

    print("Validation ROC-AUC:",
          round(roc_auc_score(y_val, probabilities), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        predictions,
        target_names=["On time", "Late"],
        zero_division=0
    ))


if __name__ == "__main__":
    evaluate_model()
