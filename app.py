
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import predict_delivery_risk


# --------------------------------------------------
# 1. Page configuration — must come first
# --------------------------------------------------
st.set_page_config(
    page_title="Delivery Risk Intelligence",
    page_icon="📦",
    layout="wide"
)


# --------------------------------------------------
# 2. Sidebar status
# --------------------------------------------------
st.sidebar.success("Prediction module imported successfully")


# --------------------------------------------------
# 3. Dashboard header
# --------------------------------------------------
st.title("📦 Delivery Risk Intelligence")
st.subheader("Pre-Dispatch Delivery Risk Prediction")

st.markdown("""
This dashboard explores delivery delays in Brazilian e-commerce
and demonstrates how machine learning can support delivery-risk
analysis and operational planning.
""")

col1, col2, col3 = st.columns(3)

col1.metric("Orders Evaluated", "25,352")
col2.metric("Late Delivery Rate", "4.40%")
col3.metric("Model ROC-AUC", "0.5882")

st.info(
    "These figures represent the final holdout evaluation "
    "for May–August 2018, not live delivery predictions."
)


# --------------------------------------------------
# 4. Project overview
# --------------------------------------------------
st.header("Project Overview")

st.markdown("""
- **Dataset:** Olist Brazilian E-Commerce
- **Model:** Logistic Regression
- **Validation:** Chronological split
- **Final evaluation:** May–August 2018 holdout
- **Objective:** Investigate whether order information can
  help identify deliveries at risk of arriving late.
""")


# --------------------------------------------------
# 5. Model performance
# --------------------------------------------------
st.divider()
st.header("📊 Model Performance")

st.write(
    "The Logistic Regression model was evaluated on an "
    "untouched chronological holdout covering May–August 2018."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Final Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Average Precision",
            "ROC-AUC",
            "Late-order Precision",
            "Late-order Recall"
        ],
        "Score": [0.0572, 0.5882, 0.0636, 0.3022]
    })

    st.dataframe(
        metrics_df,
        hide_index=True,
        use_container_width=True
    )

with col2:
    st.subheader("Confusion Matrix")

    confusion_df = pd.DataFrame({
        "Actual": ["On time", "Late"],
        "Predicted On Time": [19275, 778],
        "Predicted Late": [4962, 337]
    })

    st.dataframe(
        confusion_df,
        hide_index=True,
        use_container_width=True
    )

st.caption(
    "Results use a 0.50 classification threshold. "
    "Predictive performance is modest, particularly for late deliveries."
)


# --------------------------------------------------
# 6. Classification performance chart
# --------------------------------------------------
st.divider()
st.header("📈 Classification Performance")

performance_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-score"],
    "On-time orders": [0.96, 0.80, 0.87],
    "Late orders": [0.06, 0.30, 0.11]
})

performance_long = performance_df.melt(
    id_vars="Metric",
    var_name="Delivery Class",
    value_name="Score"
)

fig = px.bar(
    performance_long,
    x="Metric",
    y="Score",
    color="Delivery Class",
    barmode="group",
    title="Classification Metrics by Delivery Class",
    range_y=[0, 1],
    text_auto=".2f"
)

fig.update_layout(
    yaxis_title="Score",
    xaxis_title="",
    legend_title="",
    height=420
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 7. Interactive delivery-risk calculator
# --------------------------------------------------
st.divider()
st.header("🧮 Delivery Risk Calculator")

st.warning(
    "Demonstration only: the model has modest predictive performance "
    "and has not been validated for live operational decisions."
)

with st.form("risk_calculator"):
    st.subheader("Order details")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_items = st.number_input(
            "Total items",
            min_value=1,
            max_value=21,
            value=1
        )

        total_price = st.number_input(
            "Total price (BRL)",
            min_value=0.0,
            value=100.0
        )

        total_freight = st.number_input(
            "Total freight (BRL)",
            min_value=0.0,
            value=20.0
        )

        seller_count = st.number_input(
            "Number of sellers",
            min_value=1,
            max_value=5,
            value=1
        )

    with col2:
        promised_delivery_days = st.number_input(
            "Promised delivery days",
            min_value=1,
            max_value=180,
            value=24
        )

        purchase_month = st.selectbox(
            "Purchase month",
            list(range(1, 13)),
            index=5
        )

        purchase_day_of_week = st.selectbox(
            "Purchase weekday (Monday = 0)",
            list(range(7)),
            index=2
        )

        purchase_hour = st.slider(
            "Purchase hour",
            0, 23, 12
        )

    with col3:
        approval_day_of_week = st.selectbox(
            "Approval weekday (Monday = 0)",
            list(range(7)),
            index=2
        )

        approval_hour = st.slider(
            "Approval hour",
            0, 23, 12
        )

        approval_delay_hours = st.number_input(
            "Purchase-to-approval delay (hours)",
            min_value=0.0,
            value=1.0
        )

    submitted = st.form_submit_button("Estimate risk")


# --------------------------------------------------
# 8. Display prediction using the reusable module
# --------------------------------------------------
if submitted:
    order_features = {
        "total_items": total_items,
        "total_price": total_price,
        "total_freight": total_freight,
        "seller_count": seller_count,
        "promised_delivery_days": promised_delivery_days,
        "purchase_month": purchase_month,
        "purchase_day_of_week": purchase_day_of_week,
        "purchase_hour": purchase_hour,
        "approval_day_of_week": approval_day_of_week,
        "approval_hour": approval_hour,
        "approval_delay_hours": approval_delay_hours
    }

    # Generate prediction through src/predict.py
    result = predict_delivery_risk(order_features)

    st.subheader("Model output")

    st.metric(
        "Model risk score",
        f"{result['risk_score']:.1%}"
    )

    st.progress(float(result["risk_score"]))

    if result["prediction"] == 1:
        st.warning("Prediction: Late-delivery risk")
    else:
        st.success("Prediction: Lower predicted risk")

    st.info(
        "This score is not a calibrated probability of late delivery. "
        "The model has modest predictive performance and is intended "
        "for demonstration only."
    )

    with st.expander("How should I interpret this result?"):
        st.markdown("""
        - The score comes from the trained Logistic Regression model.
        - A higher score indicates a stronger model signal, not a
          proven probability of late delivery.
        - Final holdout ROC-AUC: **0.5882**
        - Final holdout Average Precision: **0.0572**
        - Use this calculator for demonstration and learning only.
        """)


# --------------------------------------------------
# 9. Limitations and future improvements
# --------------------------------------------------
st.divider()
st.header("⚠️ Limitations & Future Improvements")

tab1, tab2 = st.tabs([
    "Current Limitations",
    "Future Improvements"
])

with tab1:
    st.markdown("""
    - **Modest predictive performance:** Final holdout ROC-AUC
      was 0.5882 and Average Precision was 0.0572.
    - **Class imbalance:** Only 4.40% of final holdout orders
      were late.
    - **Historical data:** The model uses Olist Brazilian
      e-commerce data from 2016–2018.
    - **Limited features:** Seller reliability and detailed
      shipping-route information are not included.
    - **Score calibration:** The displayed model score is not
      a calibrated probability of late delivery.
    - **Prediction timing:** Approval-related inputs mean this
      calculator is intended as a post-approval demonstration,
      not a pre-purchase prediction system.
    """)

with tab2:
    st.markdown("""
    - Engineer time-safe historical seller reliability features.
    - Investigate geographic distance and delivery-route features.
    - Test additional models and compare them using chronological
      validation.
    - Explore probability calibration using training/validation
      data only.
    - Evaluate operational thresholds based on review capacity
      and the cost of missed late deliveries.
    - Validate the system on newer data before any real-world use.
    """)

st.caption(
    "This project is an educational machine-learning prototype "
    "and is not intended for live logistics decisions."
)
