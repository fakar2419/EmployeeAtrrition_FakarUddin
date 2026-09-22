# 👥 Enterprise Employee Attrition & Risk Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green.svg)](https://xgboost.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange.svg)](https://shap.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An end-to-end Machine Learning System designed to predict employee attrition risk scores, identify top retention drivers using **SHAP**, simulate HR policy interventions, and expose production endpoints via **FastAPI** and an interactive **Streamlit** executive portal.

---

## 🚀 13-Step End-to-End Pipeline Architecture

```
                                [ HR Attrition Dataset (1,470 Employees) ]
                                                    │
                                                    ▼
                                 [ 1. Data Cleaning & Preprocessing ]
                                     ├── Drop Constant / Zero-Variance Features
                                     ├── Target Mapping (Yes -> 1, No -> 0)
                                     └── OneHotEncoder (Categorical) + Passthrough (Numerical)
                                                    │
                                                    ▼
                                 [ 2 & 3. 5-Fold Stratified Cross-Validation ]
                                     ├── Leakage-Free Preprocessing within Folds
                                     └── XGBoost Classifier with scale_pos_weight = 5.20
                                                    │
                                                    ▼
                                 [ 4 & 5. PR-AUC & Hyperparameter Tuning ]
                                     ├── Primary Metric: PR-AUC (Imbalanced Classification)
                                     └── RandomizedSearchCV Optimization (n_estimators, depth, lr)
                                                    │
                                                    ▼
                                 [ 6 & 7. Threshold Optimization & Error Analysis ]
                                     ├── Optimal Decision Threshold T* = 0.45 (Max F1 Score)
                                     └── Confusion Matrix & Outlier Analysis
                                                    │
                                                    ▼
                                 [ 8 & 9. SHAP Explainability & Risk Scoring ]
                                     ├── TreeExplainer Feature Importance Ranking
                                     └── Risk Score (0-100) -> Low (<35%), Med (35-70%), High (>70%)
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
      [ 10. Streamlit Web Portal (8501) ]                           [ 11. FastAPI REST Backend (8000) ]
       ├── Workforce Overview                                        ├── GET /health & GET /model-info
       ├── Individual SHAP Risk Explainer                            ├── POST /predict (Single Employee)
       ├── 'What-If' HR Policy Simulator                             └── POST /predict/batch (Workforce Batch)
       ├── Batch CSV Assessor                                                        │
       └── Model Diagnostic Center                                                   │
                   │                                                                 │
                   └────────────────────────────────┬────────────────────────────────┘
                                                    ▼
                                 [ 12 & 13. Dockerization & Production GitHub Showcase ]
```

---

## 📊 Model Performance & Benchmarks

Because Employee Attrition is severely imbalanced (16.1% Attrition vs 83.9% Retention), **PR-AUC (Precision-Recall Area Under Curve)** was designated as the primary optimization metric to eliminate false optimism inherent in naive Accuracy:

| Metric | 5-Fold Stratified OOF Score | Benchmark Significance |
| :--- | :---: | :--- |
| **OOF ROC-AUC** | **0.8113** | Strong global class separation across false positive rates |
| **OOF PR-AUC** | **0.5799** | Outstanding precision-recall balance (Baseline random = 0.161) |
| **Optimal Threshold ($T^*$)** | **0.45** | Maximizes F1 Score (`0.5462`), penalizing costly false negatives |
| **High Risk Workforce Tier** | **7.8% (114 emp)** | Pinpoints vulnerable employees requiring immediate HR intervention |

---

## 🔑 Top Global Risk Drivers (SHAP Analysis)

The top feature drivers identified by **SHAP (SHapley Additive exPlanations)** TreeExplainer:

1. **`OverTime`**: Employees working overtime experience significantly higher attrition risk scores.
2. **`MonthlyIncome`**: Lower salary brackets correlate strongly with early departure.
3. **`Age`**: Younger employees demonstrate higher mobility compared to senior staff.
4. **`StockOptionLevel`**: Zero stock options significantly elevates turnover likelihood.
5. **`TotalWorkingYears` & `YearsAtCompany`**: Early tenure years (< 3 years) represent peak churn vulnerability.

---

## 🖥️ System Modules & User Experience

### 1. 📊 Workforce Risk Overview (`Streamlit`)
Executive KPIs tracking total workforce, high-risk counts, department vulnerability heatmaps, and score distribution histograms.

### 2. 👤 Individual Employee Risk Predictor
Inspect any employee record to calculate their exact **Risk Score (0 - 100)**, view gauge meters, and receive automated HR retention action plans.

### 3. 💡 'What-If' HR Policy Simulator
Interactively test policy interventions (e.g. eliminating OverTime for Sales, granting +1 Stock Option level, 10% global raise) and quantify instant high-risk count reduction.

### 4. 📁 Batch Risk Assessor
Upload arbitrary workforce CSV datasets to generate instant risk predictions and export downloadable executive CSV reports.

### 5. ⚡ FastAPI RESTful Microservice
High-performance REST API with automatic OpenAPI documentation (`/docs`) for enterprise HR system integration.

---

## 🛠️ Quick Start & Local Execution

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/fakar2419/EmployeeAtrrition_FakarUddin.git
cd EmployeeAtrrition_FakarUddin

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execute Training Pipeline
```bash
python3 train_pipeline.py
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
*Access dashboard at `http://localhost:8501`*

### 4. Launch FastAPI REST Service
```bash
uvicorn main:app --reload --port 8000
```
*Access interactive API documentation at `http://localhost:8000/docs`*

---

## 🐳 Docker Deployment

To launch both FastAPI and Streamlit services via Docker Compose:

```bash
docker-compose up --build
```

- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI Documentation**: `http://localhost:8000/docs`

---

## 🎓 Placement & Interview Preparation Cheat Sheet

### 1. Why optimize PR-AUC instead of ROC-AUC / Accuracy for Employee Attrition?
> **Answer**: Attrition is a minority class (16.1%). Accuracy is misleading because a naive model predicting zero attrition gets 83.9% accuracy while failing completely. ROC-AUC includes True Negatives in its denominator, which can mask poor minority-class precision on imbalanced datasets. PR-AUC focuses strictly on True Positives, False Positives, and False Negatives, making it the gold standard for imbalanced HR analytics.

### 2. How did you prevent Data Leakage during 5-Fold Stratified Cross-Validation?
> **Answer**: Preprocessing transformations (`OneHotEncoder`, scaling) were fitted strictly on the training partition ($X_{train}$) within each fold before transforming validation slices ($X_{val}$).

### 3. How does `scale_pos_weight` in XGBoost handle class imbalance?
> **Answer**: `scale_pos_weight = count(negative) / count(positive)` (5.20 in our dataset) scales the gradient updates for positive class instances, instructing XGBoost to penalize misclassifying positive attrition cases 5.2x more heavily during loss gradient calculations.

---

## 👤 Author

**Fakar Uddin**
- GitHub: [@fakar2419](https://github.com/fakar2419)
- Enterprise Portfolio & Placement Project
