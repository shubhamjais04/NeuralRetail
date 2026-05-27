# 🛍️ NeuralRetail — AI Sales Intelligence Platform
### Amdox Technologies | Data Science & Analytics | April 2026

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)

---

## 📌 Project Overview
NeuralRetail is an end-to-end AI-powered sales intelligence platform built during my Data Science internship at **Amdox Technologies**. It ingests retail transactional data to produce demand forecasts, customer intelligence, churn predictions, and inventory insights — all served through an interactive Streamlit dashboard and REST API.

---

## 🎯 Business Impact
| Objective | Metric | Target |
|---|---|---|
| Churn Reduction | AUC-ROC | ≥ 0.90 |
| Forecast Accuracy | MAPE | ≤ 10% |
| Customer Segments | Silhouette Score | ≥ 0.55 |

---

## 🏗️ Project Structure

```
NeuralRetail/
├── data/                              # Datasets
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb   # RFM + Lag Features
│   ├── 03_model_training.ipynb        # XGBoost + Forecasting
│   └── 04_drift_detection.ipynb       # Evidently AI Drift
├── src/
│   ├── models/                        # Saved ML models
│   └── api/main.py                    # FastAPI scoring API
├── dashboard/app.py                   # Streamlit dashboard
├── evidently_reports/                 # Drift + EDA reports
├── mlflow_runs/                       # MLflow experiment logs
└── requirements.txt
```
---

## 🤖 ML Models
| Model | Task | Algorithm | Metric |
|---|---|---|---|
| Churn Prediction | Classification | XGBoost + LightGBM | AUC-ROC |
| Demand Forecasting | Regression | Linear Regression + Lag Features | MAPE |
| Customer Segmentation | Clustering | K-Means RFM | Silhouette |

---

## 📊 Dashboard Pages
1. **Executive Overview** — KPIs, revenue trends, top products
2. **Demand Forecast** — Actual vs predicted revenue
3. **Customer Intelligence** — RFM segments, distributions
4. **Churn Risk Analysis** — Risk levels, SHAP explainability
5. **Model Tracker** — MLflow experiments, model files

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

### 4. Launch dashboard
```bash
cd dashboard
streamlit run app.py
```

### 5. Launch API
```bash
cd src/api
uvicorn main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
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

## 👨‍💻 Author
**Shubham Jaiswal**  
Data Science Intern — Amdox Technologies  
[GitHub](https://github.com/shubhamjais04) | [LinkedIn](https://linkedin.com/in/shubhjais04)
