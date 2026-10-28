
from data_preprocessing import load_data, build_dataset


# Step 1: Load the CSV files
orders, items = load_data()

print("Orders loaded:", orders.shape)
print("Order items loaded:", items.shape)


# Step 2: Build the model-ready dataset
model_data = build_dataset(orders, items)

print("\nModel-ready dataset:", model_data.shape)
print("\nTarget distribution:")
print(model_data["late_delivery"].value_counts())

print("\nLate-delivery percentage:")
print(round(model_data["late_delivery"].mean() * 100, 2), "%")

print("\nMissing values in model features:")
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

print(model_data[features].isna().sum())
