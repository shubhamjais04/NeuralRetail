# ============================================================
# NeuralRetail — FastAPI Scoring API
# Amdox Technologies | Data Science & Analytics
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# ============================================================
# Initialize FastAPI app
# ============================================================
app = FastAPI(
    title="NeuralRetail API",
    description="AI-Powered Sales Intelligence API — Amdox Technologies",
    version="1.0.0"
)

# ============================================================
# Load models at startup
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE, '../models')

# Load churn model
churn_model    = joblib.load(os.path.join(MODELS_PATH, 'churn_model.pkl'))
churn_features = joblib.load(os.path.join(MODELS_PATH, 'churn_features.pkl'))

# Load forecast model
forecast_model    = joblib.load(os.path.join(MODELS_PATH, 'forecast_model.pkl'))
forecast_features = joblib.load(os.path.join(MODELS_PATH, 'forecast_features.pkl'))

# Load RFM data for segments
rfm = pd.read_csv(os.path.join(BASE, '../../data/rfm_features.csv'))

print("✅ All models loaded successfully!")

# ============================================================
# Request/Response schemas using Pydantic
# ============================================================

class ChurnRequest(BaseModel):
    """Input data for churn prediction"""
    Recency   : float   # Days since last purchase
    Frequency : float   # Number of purchases
    Monetary  : float   # Total amount spent
    R_Score   : float   # Recency score (1-5)
    F_Score   : float   # Frequency score (1-5)
    M_Score   : float   # Monetary score (1-5)
    RFM_Score : float   # Combined RFM score

    class Config:
        json_schema_extra = {
            "example": {
                "Recency"  : 30,
                "Frequency": 5,
                "Monetary" : 1500.0,
                "R_Score"  : 4,
                "F_Score"  : 3,
                "M_Score"  : 4,
                "RFM_Score": 3.67
            }
        }

class DemandRequest(BaseModel):
    """Input data for demand forecasting"""
    Revenue_Lag_1  : float  # Revenue yesterday
    Revenue_Lag_7  : float  # Revenue 7 days ago
    Revenue_Lag_14 : float  # Revenue 14 days ago
    Rolling_Mean_7 : float  # 7-day average revenue
    Rolling_Mean_30: float  # 30-day average revenue
    DayOfWeek      : int    # 0=Monday, 6=Sunday
    Month          : int    # 1-12
    Quarter        : int    # 1-4
    IsWeekend      : int    # 1 if weekend, 0 otherwise

    class Config:
        json_schema_extra = {
            "example": {
                "Revenue_Lag_1"  : 15000.0,
                "Revenue_Lag_7"  : 14000.0,
                "Revenue_Lag_14" : 13000.0,
                "Rolling_Mean_7" : 14500.0,
                "Rolling_Mean_30": 13800.0,
                "DayOfWeek"      : 1,
                "Month"          : 11,
                "Quarter"        : 4,
                "IsWeekend"      : 0
            }
        }

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Root endpoint — welcome message"""
    return {
        "message" : "Welcome to NeuralRetail API",
        "company" : "Amdox Technologies",
        "version" : "1.0.0",
        "docs"    : "/docs"
    }

@app.get("/health")
def health_check():
    """Health check — confirms API is running"""
    return {
        "status" : "healthy",
        "models" : {
            "churn_model"    : "loaded",
            "forecast_model" : "loaded"
        }
    }

@app.post("/predict/churn")
def predict_churn(request: ChurnRequest):
    """
    Predict whether a customer will churn.
    Returns churn probability and risk level.
    """
    try:
        # Prepare input features
        input_data = pd.DataFrame([{
            'Recency'  : request.Recency,
            'Frequency': request.Frequency,
            'Monetary' : request.Monetary,
            'R_Score'  : request.R_Score,
            'F_Score'  : request.F_Score,
            'M_Score'  : request.M_Score,
            'RFM_Score': request.RFM_Score
        }])

        # Get churn probability
        churn_proba = churn_model.predict_proba(input_data)[0][1]
        churn_pred  = int(churn_proba >= 0.5)

        # Assign risk level
        if churn_proba >= 0.75:
            risk_level = "High Risk"
        elif churn_proba >= 0.50:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        # Retention recommendation
        if churn_proba >= 0.75:
            recommendation = "Send immediate retention offer — 20% discount voucher"
        elif churn_proba >= 0.50:
            recommendation = "Send personalized email campaign with product recommendations"
        else:
            recommendation = "Customer is healthy — continue standard engagement"

        return {
            "churn_probability" : round(float(churn_proba), 4),
            "churn_prediction"  : churn_pred,
            "risk_level"        : risk_level,
            "recommendation"    : recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/demand")
def predict_demand(request: DemandRequest):
    """
    Predict revenue demand for next period.
    Returns predicted revenue amount.
    """
    try:
        # Prepare input features
        input_data = pd.DataFrame([{
            'Revenue_Lag_1'  : request.Revenue_Lag_1,
            'Revenue_Lag_7'  : request.Revenue_Lag_7,
            'Revenue_Lag_14' : request.Revenue_Lag_14,
            'Rolling_Mean_7' : request.Rolling_Mean_7,
            'Rolling_Mean_30': request.Rolling_Mean_30,
            'DayOfWeek'      : request.DayOfWeek,
            'Month'          : request.Month,
            'Quarter'        : request.Quarter,
            'IsWeekend'      : request.IsWeekend
        }])

        # Predict revenue
        predicted_revenue = forecast_model.predict(input_data)[0]

        return {
            "predicted_revenue" : round(float(predicted_revenue), 2),
            "currency"          : "GBP",
            "model"             : "Linear Regression + Lag Features"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/segments")
def get_segments():
    """
    Get customer segment summary.
    Returns count and revenue per segment.
    """
    try:
        segment_summary = rfm.groupby('Segment').agg(
            Customer_Count = ('CustomerID', 'count'),
            Total_Revenue  = ('Monetary',   'sum'),
            Avg_RFM_Score  = ('RFM_Score',  'mean'),
            Churn_Rate     = ('Churned',    'mean')
        ).reset_index()

        segment_summary['Total_Revenue'] = segment_summary['Total_Revenue'].round(2)
        segment_summary['Avg_RFM_Score'] = segment_summary['Avg_RFM_Score'].round(2)
        segment_summary['Churn_Rate']    = segment_summary['Churn_Rate'].round(4)

        return {
            "total_customers": int(len(rfm)),
            "segments"       : segment_summary.to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))