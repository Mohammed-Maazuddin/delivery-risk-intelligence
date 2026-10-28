
from pathlib import Path
import pandas as pd


# Find the project folder and raw data folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def load_data():
    """Load orders and order items from CSV files."""

    orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
    items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")

    return orders, items


def prepare_orders(orders):
    """Clean timestamps and create the target variable."""

    orders = orders.copy()

    # Convert date columns into datetime format
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for column in date_columns:
        orders[column] = pd.to_datetime(
            orders[column], errors="coerce"
        )

    # Keep delivered orders with the dates needed by our model
    orders = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
        & orders["order_approved_at"].notna()
    ].copy()

    # Target: 1 = late, 0 = on time
    orders["late_delivery"] = (
        orders["order_delivered_customer_date"].dt.normalize()
        > orders["order_estimated_delivery_date"].dt.normalize()
    ).astype(int)

    # Time-related features
    orders["promised_delivery_days"] = (
        orders["order_estimated_delivery_date"].dt.normalize()
        - orders["order_purchase_timestamp"].dt.normalize()
    ).dt.days

    orders["purchase_month"] = orders["order_purchase_timestamp"].dt.month
    orders["purchase_day_of_week"] = orders["order_purchase_timestamp"].dt.dayofweek
    orders["purchase_hour"] = orders["order_purchase_timestamp"].dt.hour

    orders["approval_day_of_week"] = orders["order_approved_at"].dt.dayofweek
    orders["approval_hour"] = orders["order_approved_at"].dt.hour

    orders["approval_delay_hours"] = (
        orders["order_approved_at"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    return orders


def prepare_items(items):
    """Combine item information into one row per order."""

    order_items = items.groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        seller_count=("seller_id", "nunique")
    ).reset_index()

    return order_items


def build_dataset(orders, items):
    """Combine order information and item information."""

    orders = prepare_orders(orders)
    order_items = prepare_items(items)

    # Match item information to each eligible order
    data = orders.merge(
        order_items,
        on="order_id",
        how="inner",
        validate="one_to_one"
    )

    return data
