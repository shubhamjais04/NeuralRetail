# 🛍️ NeuralRetail — AI Sales Intelligence Platform

<p align="center">
  <a href="https://neuralretail-amdox.streamlit.app">
    <img src="https://img.shields.io/badge/🚀 Live Demo-Streamlit-FF4B4B?style=for-the-badge" alt="Live Demo"/>
  </a>
  <a href="https://github.com/shubhamjais04/NeuralRetail">
    <img src="https://img.shields.io/badge/GitHub-NeuralRetail-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/XGBoost-Churn_Model-EC6C25?style=for-the-badge" alt="XGBoost"/>
</p>

---

## 📌 Project Overview

**NeuralRetail** is an end-to-end AI-powered sales intelligence platform built during my Data Science internship at **Amdox Technologies**. It ingests retail transactional data to produce demand forecasts, customer segmentation, churn predictions, and model tracking — all served through an interactive 5-page Streamlit dashboard and a REST API.

> **Dataset:** UCI Online Retail Dataset — 541,909 transactions across 38 countries (2010–2011)

---

## 🎯 Business Impact

| Objective | Algorithm | Metric | Target |
|-----------|-----------|--------|--------|
| Churn Reduction | XGBoost + LightGBM | AUC-ROC | ≥ 0.90 |
| Forecast Accuracy | Linear Regression + Lag Features | MAPE | ≤ 10% |
| Customer Segmentation | K-Means RFM | Silhouette Score | ≥ 0.55 |

---

## 📸 Dashboard Screenshots

### 📊 Executive Overview — KPIs & Revenue Trends
<img src="assets/screenshots/executive_overview.png" width="100%"/>

---

### 📈 Demand Forecast — Actual vs Predicted Revenue
<img src="assets/screenshots/demand_forecast.png" width="100%"/>

---

### 👥 Customer Intelligence — RFM Segmentation
<img src="assets/screenshots/customer_intelligence.png" width="100%"/>

---

### ⚠️ Churn Risk Analysis — Risk Levels & SHAP Explainability
<img src="assets/screenshots/churn_risk.png" width="100%"/>

---

### 🧪 Model Tracker — MLflow Experiments & Model Files
<img src="assets/screenshots/model_tracker.png" width="100%"/>

---

## 🤖 ML Models

| Model | Task | Algorithm | Metric |
|-------|------|-----------|--------|
| Churn Prediction | Classification | XGBoost + LightGBM | AUC-ROC |
| Demand Forecasting | Regression | Linear Regression + Lag Features | MAPE |
| Customer Segmentation | Clustering | K-Means RFM | Silhouette Score |

---

## 🏗️ Project Structure

```
NeuralRetail/
├── assets/
│   └── screenshots/          ← Dashboard screenshots
├── data/
│   ├── online_retail_clean.csv
│   ├── rfm_features.csv
│   ├── daily_sales_features.csv
│   ├── forecast_results.csv
│   └── Online_Retail.xlsx
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_drift_detection.ipynb
├── src/
│   ├── api/                  ← FastAPI REST API
│   ├── features/
│   ├── ingestion/
│   └── models/               ← Trained .pkl model files
├── dashboard/
│   ├── app.py                ← Main Streamlit dashboard
│   └── requirements.txt
├── evidently_reports/        ← Drift detection reports
├── mlflow_runs/              ← MLflow experiment tracking
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/shubhamjais04/NeuralRetail.git
cd NeuralRetail
```

### 2. Create virtual environment
```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the notebooks in order
```
notebooks/01_EDA.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_model_training.ipynb
notebooks/04_drift_detection.ipynb
```

### 4. Launch the dashboard
```bash
cd dashboard
streamlit run app.py
```

### 5. Launch the API
```bash
cd src/api
uvicorn main:app --reload --port 8000
```
API docs available at: `http://localhost:8000/docs`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| ML Models | XGBoost, LightGBM, Scikit-learn |
| Explainability | SHAP |
| Forecasting | Linear Regression + Lag Features |
| Experiment Tracking | MLflow |
| Drift Detection | Evidently AI |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn |
| Data Processing | Pandas, NumPy |

---

## 🌐 Live Demo

👉 **[neuralretail-amdox.streamlit.app](https://neuralretail-amdox.streamlit.app)**

---

## 👨‍💻 Author

**Shubham Jaiswal**
Data Science & Analytics Intern — Amdox Technologies

[![GitHub](https://img.shields.io/badge/GitHub-shubhamjais04-181717?style=flat&logo=github)](https://github.com/shubhamjais04)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shubham_Jaiswal-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/shubhamjaiswal04)
