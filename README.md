# 📦 Delivery Risk Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### Pre-Dispatch Delivery Risk Prediction Using Machine Learning

An end-to-end Data Science project exploring whether order-level information can help identify late deliveries in Brazilian e-commerce. The project includes data auditing, feature engineering, chronological model validation, and an interactive Streamlit dashboard.

---

## 🎯 Project Objective

Late deliveries significantly harm customer satisfaction and inflate operational costs. This project investigates whether signals available around order approval can help distinguish orders at high risk of arriving past their estimated delivery date.

---

## 📊 Dataset

- **Source:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Period:** 2016–2018
- **Eligible delivered orders:** 96,456 orders with the required model features
- **Final holdout:** 25,352 orders from May–August 2018
- **Target:** `late_delivery` — `1` if actual delivery occurred after the estimated delivery date, otherwise `0`

> **Note:** The target is based on completed deliveries. Therefore, the project represents a proof-of-concept investigation rather than an uncalibrated live production dispatch system.

---

## 🛠️ Technology Stack

- **Core:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Joblib
- **Visualization:** Plotly Express, Matplotlib, Seaborn
- **Interactive UI:** Streamlit
- **Notebooks:** Jupyter Lab / Notebook

---

## ⚙️ Methodology & Pipeline

1. **Data Ingestion & Audit:** Inspect Olist order and order-item datasets (`notebooks/01_data_audit.ipynb`).
2. **Data Preprocessing & Cleaning:** Standardize timestamps, filter valid delivered orders, and compute delivery delays.
3. **Feature Engineering:**
   - Order value: `total_price`, `total_freight`
   - Logistics complexity: `total_items`, `seller_count`
   - Operational timing: `promised_delivery_days`, `approval_delay_hours`, purchase/approval days of week and hours.
4. **Chronological Validation Strategy:** Use strict time-based splits (pre-2018 train, Jan–Apr 2018 validation, May–Aug 2018 final holdout) to prevent future-data leakage.
5. **Model Training & Evaluation:** Compare models with balanced weighting to address class imbalance (~6.7% baseline late deliveries).
6. **Deployment Artifacts:** Export serialized model (`logistic_model.joblib`), scaler (`scaler.joblib`), and feature schemas (`feature_names.joblib`).
7. **Interactive Dashboard:** Build a Streamlit dashboard (`app.py`) for live metric inspection and real-time risk scoring demonstration.

---

## 📈 Final Holdout Results

| Metric | Result |
|---|---:|
| Orders evaluated | 25,352 |
| Actual late-delivery rate | 4.40% |
| Average Precision | 0.0572 |
| ROC-AUC | 0.5882 |
| Late-order precision at threshold 0.50 | 6.36% |
| Late-order recall at threshold 0.50 | 30.22% |

The results indicate modest predictive performance, highlighting key operational challenges in supply-chain delay forecasting without live transit telematics.

---

## 🖥️ Streamlit Dashboard

The web application features:
- **Project Overview & KPI Cards:** Instant metrics on evaluated orders and late-delivery prevalence.
- **Model Evaluation:** Interactive confusion matrices and classification summaries.
- **Interactive Risk Calculator:** Real-time form enabling users to simulate order characteristics (item count, price, freight, seller count, promised days, approval latency) and inspect predicted risk scores.
- **Transparent Operational Caveats:** Practical guidelines on model limitations and recommended next steps.

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Mohammed-Maazuddin/delivery-risk-intelligence.git
cd delivery-risk-intelligence
```

### 2. Set up virtual environment & install dependencies

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the dashboard

Pre-trained model artifacts are included in `models/`, allowing immediate execution:

```bash
streamlit run app.py
```

### 4. (Optional) Retraining & Data Pipeline

To re-run the full training pipeline:
1. Download the datasets from [Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. Place the unzipped CSV files into `data/raw/` (`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, etc.).
3. Run training:
   ```bash
   python src/train.py
   ```

---

## 📁 Project Structure

```text
delivery-risk-intelligence/
├── .streamlit/
│   └── config.toml                  # Streamlit application styling & server config
├── data/
│   ├── processed/                   # Processed datasets (git-ignored)
│   └── raw/                         # Raw Kaggle CSVs (git-ignored)
├── models/
│   ├── feature_names.joblib         # Feature schema used during model fitting
│   ├── logistic_model.joblib        # Pre-trained Logistic Regression classifier
│   └── scaler.joblib                # Standard scaler fitted on training data
├── notebooks/
│   └── 01_data_audit.ipynb          # Exploratory data analysis & auditing
├── reports/                         # Generated figures and reports
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Timestamp parsing & feature aggregation
│   ├── model_evaluation.py         # Holdout evaluation & metric reporting
│   ├── predict.py                   # Inference function for single-order scoring
│   ├── test_prediction.py           # Smoke test for prediction pipeline
│   ├── test_preprocessing.py        # Smoke test for preprocessing
│   └── train.py                     # Chronological split & model training script
├── .gitignore                       # Git ignore rules for caches, envs, and large data
├── LICENSE                          # MIT License
├── README.md                        # Documentation
├── app.py                           # Streamlit dashboard application
└── requirements.txt                 # Project dependencies
```

---

## ⚠️ Limitations

- **Predictive Power:** Order-level static attributes provide limited predictive leverage compared to real-time carrier tracking or route telematics.
- **Historical Context:** Dataset covers historical 2016–2018 Brazilian e-commerce operations.
- **Class Imbalance:** Late delivery is an infrequent event (~4–7%), requiring careful threshold selection.
- **Probability Calibration:** Output probabilities are non-calibrated ranking scores.

---

## 🔮 Future Roadmap

- Integrate postal-code geocoordinate distances between customer and seller.
- Engineer historical seller reliability and fulfillment speed metrics.
- Experiment with gradient-boosted trees (XGBoost, LightGBM) using chronological cross-validation.
- Optimize operational decision thresholds based on business penalty costs.

---

## 👨‍💻 Author

**Mohammed Maazuddin**
- GitHub: [@Mohammed-Maazuddin](https://github.com/Mohammed-Maazuddin)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
