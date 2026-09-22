"""
Production FastAPI Backend for Employee Attrition & Risk Scoring Service
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="Employee Attrition Risk Scoring API",
    description="Enterprise RESTful microservice providing real-time ML-powered employee attrition predictions, risk scores, and tier classifications.",
    version="1.0.0"
)

# Load pipeline artifacts
PREPROCESSOR_PATH = "models/preprocessor.joblib"
MODEL_PATH = "models/xgb_model.joblib"
METADATA_PATH = "models/metadata.json"

if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model artifacts missing in models/ directory. Run train_pipeline.py first.")

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

class EmployeeData(BaseModel):
    Age: int = Field(..., example=35)
    BusinessTravel: str = Field(..., example="Travel_Rarely")
    DailyRate: int = Field(..., example=1102)
    Department: str = Field(..., example="Sales")
    DistanceFromHome: int = Field(..., example=1)
    Education: int = Field(..., example=2)
    EducationField: str = Field(..., example="Life Sciences")
    EnvironmentSatisfaction: int = Field(..., example=2)
    Gender: str = Field(..., example="Female")
    HourlyRate: int = Field(..., example=94)
    JobInvolvement: int = Field(..., example=3)
    JobLevel: int = Field(..., example=2)
    JobRole: str = Field(..., example="Sales Executive")
    JobSatisfaction: int = Field(..., example=4)
    MaritalStatus: str = Field(..., example="Single")
    MonthlyIncome: int = Field(..., example=5993)
    MonthlyRate: int = Field(..., example=19479)
    NumCompaniesWorked: int = Field(..., example=8)
    OverTime: str = Field(..., example="Yes")
    PercentSalaryHike: int = Field(..., example=11)
    PerformanceRating: int = Field(..., example=3)
    RelationshipSatisfaction: int = Field(..., example=1)
    StockOptionLevel: int = Field(..., example=0)
    TotalWorkingYears: int = Field(..., example=8)
    TrainingTimesLastYear: int = Field(..., example=0)
    WorkLifeBalance: int = Field(..., example=1)
    YearsAtCompany: int = Field(..., example=6)
    YearsInCurrentRole: int = Field(..., example=4)
    YearsSinceLastPromotion: int = Field(..., example=0)
    YearsWithCurrManager: int = Field(..., example=5)

class PredictionResult(BaseModel):
    attrition_probability: float
    risk_score: float
    risk_tier: str
    is_high_risk: bool
    optimal_threshold_used: float

@app.get("/")
def root():
    return {
        "service": "Employee Attrition Risk Scoring API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessor_loaded": True
    }

@app.get("/model-info")
def get_model_info():
    return {
        "model_type": "XGBoost Classifier",
        "optimal_threshold": metadata.get("optimal_threshold"),
        "metrics": {
            "oof_pr_auc": metadata.get("pr_auc"),
            "oof_roc_auc": metadata.get("roc_auc"),
            "f1_score": metadata.get("f1_score"),
            "precision": metadata.get("precision"),
            "recall": metadata.get("recall")
        },
        "risk_tier_distribution": metadata.get("risk_tier_distribution")
    }

def _process_input_dataframe(df_input: pd.DataFrame) -> np.ndarray:
    # Ensure all expected columns are present
    cat_cols = metadata["cat_cols"]
    num_cols = metadata["num_cols"]
    expected_cols = cat_cols + num_cols
    
    missing = [c for c in expected_cols if c not in df_input.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")
        
    df_ordered = df_input[expected_cols]
    return preprocessor.transform(df_ordered)

@app.post("/predict", response_model=PredictionResult)
def predict_single(data: EmployeeData):
    try:
        input_dict = data.model_dump()
        df_input = pd.DataFrame([input_dict])
        
        X_trans = _process_input_dataframe(df_input)
        prob = float(model.predict_proba(X_trans)[0, 1])
        risk_score = round(prob * 100, 2)
        
        thresh = metadata.get("optimal_threshold", 0.45)
        
        if risk_score >= 70.0:
            tier = "High Risk"
        elif risk_score >= 35.0:
            tier = "Medium Risk"
        else:
            tier = "Low Risk"
            
        return PredictionResult(
            attrition_probability=round(prob, 4),
            risk_score=risk_score,
            risk_tier=tier,
            is_high_risk=prob >= thresh,
            optimal_threshold_used=thresh
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResult])
def predict_batch(employees: List[EmployeeData]):
    try:
        input_list = [e.model_dump() for e in employees]
        df_input = pd.DataFrame(input_list)
        
        X_trans = _process_input_dataframe(df_input)
        probs = model.predict_proba(X_trans)[:, 1]
        
        thresh = metadata.get("optimal_threshold", 0.45)
        results = []
        
        for prob in probs:
            p = float(prob)
            risk_score = round(p * 100, 2)
            if risk_score >= 70.0:
                tier = "High Risk"
            elif risk_score >= 35.0:
                tier = "Medium Risk"
            else:
                tier = "Low Risk"
                
            results.append(PredictionResult(
                attrition_probability=round(p, 4),
                risk_score=risk_score,
                risk_tier=tier,
                is_high_risk=p >= thresh,
                optimal_threshold_used=thresh
            ))
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
