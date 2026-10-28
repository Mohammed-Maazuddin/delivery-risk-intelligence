
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from data_preprocessing import load_data, build_dataset


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# 1. Load and prepare data
orders, items = load_data()
data = build_dataset(orders, items)

features = [
    "total_items",
    "total_price",
    "total_freight",
    "seller_count",
    "promised_delivery_days",
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "approval_day_of_week",
    "approval_hour",
    "approval_delay_hours"
]

target = "late_delivery"


# 2. Create chronological splits
train_mask = data["order_purchase_timestamp"] < "2018-01-01"

validation_mask = (
    (data["order_purchase_timestamp"] >= "2018-01-01")
    & (data["order_purchase_timestamp"] < "2018-05-01")
)

final_mask = data["order_purchase_timestamp"] >= "2018-05-01"

X_train = data.loc[train_mask, features]
y_train = data.loc[train_mask, target]

X_val = data.loc[validation_mask, features]
y_val = data.loc[validation_mask, target]


# 3. Scale features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# 4. Train Logistic Regression
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# 5. Evaluate on validation data
val_prob = model.predict_proba(X_val_scaled)[:, 1]

average_precision = average_precision_score(y_val, val_prob)
roc_auc = roc_auc_score(y_val, val_prob)

print("Training orders:", len(X_train))
print("Validation orders:", len(X_val))
print("Final holdout orders (reserved):", int(final_mask.sum()))

print("\nValidation Average Precision:", round(average_precision, 4))
print("Validation ROC-AUC:", round(roc_auc, 4))


# 6. Save model artifacts
joblib.dump(model, MODEL_DIR / "logistic_model.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(features, MODEL_DIR / "feature_names.joblib")

print("\nModel and preprocessing artifacts saved successfully.")
