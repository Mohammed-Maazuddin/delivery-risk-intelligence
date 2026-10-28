
from predict import predict_delivery_risk


# Sample order features in the same order used during training
sample_order = {
    "total_items": 2,
    "total_price": 150.0,
    "total_freight": 25.0,
    "seller_count": 1,
    "promised_delivery_days": 15,
    "purchase_month": 6,
    "purchase_day_of_week": 2,
    "purchase_hour": 14,
    "approval_day_of_week": 2,
    "approval_hour": 14,
    "approval_delay_hours": 1.5
}


# Get the prediction
result = predict_delivery_risk(sample_order)

print("Prediction result:")
print(result)
