"""
Enterprise Employee Attrition & Risk Scoring Training Pipeline
Includes:
- Data Preprocessing & Leakage Prevention
- 5-Fold Stratified Cross-Validation
- XGBoost Classifier Training with Imbalance Weighting
- PR-AUC & ROC-AUC Metric Evaluation
- Threshold Optimization (F1/F2 score)
- Error Analysis (FP/FN Inspection)
- SHAP Explainability Calculations
- Model & Metadata Serialization
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from xgboost import XGBClassifier

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

def load_and_clean_data(file_path="HR_Attrition.csv"):
    print("Loading HR Attrition dataset...")
    df = pd.read_csv(file_path)
    
    # 1. Drop constant & uninformative ID columns
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Target encoding: Yes -> 1, No -> 0
    df['Attrition_Target'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    X = df.drop(columns=['Attrition', 'Attrition_Target'])
    y = df['Attrition_Target']
    
    # Identify feature types
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Dataset shape: {X.shape}, Target positive class ratio: {y.mean():.4f}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    
    return df, X, y, cat_cols, num_cols

def build_preprocessor(cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )
    return preprocessor

def train_and_evaluate_5fold(X, y, cat_cols, num_cols):
    print("\n--- Running 5-Fold Stratified Cross Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    oof_probs = np.zeros(len(X))
    
    # Preprocessor fitting per fold to prevent data leakage
    preprocessor = build_preprocessor(cat_cols, num_cols)
    
    # Compute class ratio for scale_pos_weight
    ratio = (len(y) - sum(y)) / sum(y)
    print(f"Imbalance ratio (scale_pos_weight): {ratio:.2f}")
    
    cv_pr_aucs = []
    cv_roc_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Transform data within fold
        X_train_trans = preprocessor.fit_transform(X_train)
        X_val_trans = preprocessor.transform(X_val)
        
        # XGBoost Model
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train_trans, y_train)
        
        val_probs = model.predict_proba(X_val_trans)[:, 1]
        oof_probs[val_idx] = val_probs
        
        fold_pr_auc = average_precision_score(y_val, val_probs)
        fold_roc_auc = roc_auc_score(y_val, val_probs)
        cv_pr_aucs.append(fold_pr_auc)
        cv_roc_aucs.append(fold_roc_auc)
        
        print(f"Fold {fold}: PR-AUC = {fold_pr_auc:.4f}, ROC-AUC = {fold_roc_auc:.4f}")
        
    overall_pr_auc = average_precision_score(y, oof_probs)
    overall_roc_auc = roc_auc_score(y, oof_probs)
    
    print(f"\nMean CV PR-AUC: {np.mean(cv_pr_aucs):.4f} +/- {np.std(cv_pr_aucs):.4f}")
    print(f"Overall OOF PR-AUC: {overall_pr_auc:.4f}")
    print(f"Overall OOF ROC-AUC: {overall_roc_auc:.4f}")
    
    return oof_probs, overall_pr_auc, overall_roc_auc

def optimize_threshold(y_true, probs):
    print("\n--- Optimizing Decision Threshold ---")
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_f1 = 0.0
    best_f2 = 0.0
    
    thresh_results = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0
        
        thresh_results.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f1, 'f2': f2})
        
        if f1 > best_f1:
            best_f1 = f1
            best_f2 = f2
            best_thresh = t
            
    res_df = pd.DataFrame(thresh_results)
    print(f"Optimal Threshold (F1 Max): {best_thresh:.2f} -> F1 Score: {best_f1:.4f}")
    
    return best_thresh, res_df

def train_final_model(X, y, cat_cols, num_cols):
    print("\n--- Training Final Model & Saving Pipeline ---")
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_trans = preprocessor.fit_transform(X)
    
    # Get feature names after encoding
    ohe_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_feature_names = list(ohe_cat_features) + num_cols
    
    ratio = (len(y) - sum(y)) / sum(y)
    
    # Hyperparameter search using RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5]
    }
    
    base_xgb = XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_grid,
        n_iter=10,
        scoring='average_precision',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_trans, y)
    
    best_model = search.best_estimator_
    print(f"Best Hyperparameters: {search.best_params_}")
    
    # Save artifacts
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(best_model, "models/xgb_model.joblib")
    
    return preprocessor, best_model, all_feature_names, X_trans

def generate_shap_explainability(best_model, X_trans, feature_names):
    print("\n--- Calculating SHAP Values ---")
    try:
        import shap
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_trans)
        
        # Save feature importance summary
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'mean_shap': mean_abs_shap
        }).sort_values('mean_shap', ascending=False)
        
        shap_df.to_csv("models/shap_feature_importance.csv", index=False)
        print("SHAP feature importances saved to models/shap_feature_importance.csv")
        return shap_df
    except Exception as e:
        print(f"Warning: SHAP calculation failed or skipped: {e}")
        return None

def main():
    df, X, y, cat_cols, num_cols = load_and_clean_data()
    oof_probs, pr_auc, roc_auc = train_and_evaluate_5fold(X, y, cat_cols, num_cols)
    best_thresh, thresh_df = optimize_threshold(y, oof_probs)
    
    # Final Model & SHAP
    preprocessor, best_model, feature_names, X_trans = train_final_model(X, y, cat_cols, num_cols)
    shap_df = generate_shap_explainability(best_model, X_trans, feature_names)
    
    # Calculate final predictions & metrics at optimal threshold
    final_preds = (oof_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y, final_preds)
    report = classification_report(y, final_preds, output_dict=True)
    
    # Calculate Employee Risk Scores
    risk_scores = (oof_probs * 100).round(2)
    risk_tiers = pd.cut(
        risk_scores,
        bins=[-1, 35, 70, 100],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    tier_counts = risk_tiers.value_counts().to_dict()
    print("\n--- Employee Risk Score Tiers ---")
    for tier, count in tier_counts.items():
        print(f"{tier}: {count} employees ({count/len(df)*100:.1f}%)")
        
    # Save Metadata JSON
    metadata = {
        'optimal_threshold': float(best_thresh),
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'f1_score': float(f1_score(y, final_preds)),
        'precision': float(precision_score(y, final_preds)),
        'recall': float(recall_score(y, final_preds)),
        'confusion_matrix': cm.tolist(),
        'feature_names': feature_names,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'risk_tier_distribution': {str(k): int(v) for k, v in tier_counts.items()}
    }
    
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\n✅ Training Pipeline & Artifact Generation Completed Successfully!")

if __name__ == "__main__":
    main()
