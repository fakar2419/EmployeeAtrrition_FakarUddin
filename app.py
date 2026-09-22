import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Enterprise Employee Attrition & Risk Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #1A365D 0%, #2B6CB0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-size: 1.05rem;
    color: #4A5568;
    margin-bottom: 1.8rem;
}

.metric-card {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 1.2rem;
    border-left: 5px solid #3182CE;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    margin-bottom: 1rem;
}

.metric-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #718096;
    font-weight: 600;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1A202C;
}

.risk-pill-high {
    background-color: #FED7D7;
    color: #9B2C2C;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
}

.risk-pill-med {
    background-color: #FEFCBF;
    color: #975A16;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
}

.risk-pill-low {
    background-color: #C6F6D5;
    color: #22543D;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.joblib")
    model = joblib.load("models/xgb_model.joblib")
    with open("models/metadata.json", "r") as f:
        metadata = json.load(f)
    return preprocessor, model, metadata

@st.cache_data
def load_data():
    df = pd.read_csv("HR_Attrition.csv")
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    return df, df_clean

preprocessor, model, metadata = load_artifacts()
raw_df, df_clean = load_data()

# Sidebar Navigation
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=75)
else:
    st.sidebar.markdown("👥", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='font-family: Outfit; font-weight:700;'>HR Intelligence</h2>", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation Portal",
    [
        "📊 Workforce Risk Overview",
        "👤 Individual Risk Predictor",
        "💡 'What-If' HR Policy Simulator",
        "📁 Batch Risk Assessor",
        "🔬 Model Diagnostics & SHAP"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Model Metrics**\n- OOF ROC-AUC: `{metadata['roc_auc']:.4f}`\n- OOF PR-AUC: `{metadata['pr_auc']:.4f}`\n- Optimal Threshold: `{metadata['optimal_threshold']:.2f}`")

# Predict Probabilities for cached dataset
@st.cache_data
def get_cached_predictions():
    X = df_clean.drop(columns=['Attrition'])
    X_trans = preprocessor.transform(X)
    probs = model.predict_proba(X_trans)[:, 1]
    scores = (probs * 100).round(2)
    tiers = pd.cut(scores, bins=[-1, 35, 70, 100], labels=['Low Risk', 'Medium Risk', 'High Risk'])
    res_df = raw_df.copy()
    res_df['Risk_Score'] = scores
    res_df['Risk_Tier'] = tiers
    return res_df

res_df = get_cached_predictions()

# PAGE 1: WORKFORCE RISK OVERVIEW
if nav_page == "📊 Workforce Risk Overview":
    st.markdown("<h1 class='main-title'>Workforce Risk Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Executive dashboard monitoring employee attrition risks, risk tier distribution, and department vulnerabilities.</p>", unsafe_allow_html=True)
    
    # Filter Sidebar
    depts = ["All"] + list(res_df['Department'].unique())
    selected_dept = st.sidebar.selectbox("Filter Department", depts)
    
    if selected_dept != "All":
        filtered_df = res_df[res_df['Department'] == selected_dept]
    else:
        filtered_df = res_df
        
    total_emp = len(filtered_df)
    high_risk_count = (filtered_df['Risk_Tier'] == 'High Risk').sum()
    med_risk_count = (filtered_df['Risk_Tier'] == 'Medium Risk').sum()
    avg_score = filtered_df['Risk_Score'].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Workforce</div>
            <div class='metric-value'>{total_emp:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color: #E53E3E;'>
            <div class='metric-label'>High Risk Employees</div>
            <div class='metric-value' style='color:#E53E3E;'>{high_risk_count:,} ({high_risk_count/total_emp*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color: #DD6B20;'>
            <div class='metric-label'>Medium Risk Employees</div>
            <div class='metric-value' style='color:#DD6B20;'>{med_risk_count:,} ({med_risk_count/total_emp*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color: #319795;'>
            <div class='metric-label'>Average Risk Score</div>
            <div class='metric-value'>{avg_score:.1f} / 100</div>
        </div>
        """, unsafe_allow_html=True)
        
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        tier_counts = filtered_df['Risk_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Risk Tier', 'Count']
        fig_donut = px.pie(
            tier_counts,
            values='Count',
            names='Risk Tier',
            title="Workforce Risk Tier Breakdown",
            color='Risk Tier',
            color_discrete_map={
                'Low Risk': '#38A169',
                'Medium Risk': '#DD6B20',
                'High Risk': '#E53E3E'
            },
            hole=0.45
        )
        fig_donut.update_layout(template="plotly_white")
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_chart2:
        fig_hist = px.histogram(
            filtered_df,
            x="Risk_Score",
            nbins=30,
            title="Employee Risk Score Distribution (0 - 100)",
            color_discrete_sequence=["#3182CE"]
        )
        fig_hist.add_vline(x=35, line_dash="dash", line_color="#DD6B20", annotation_text="Medium Risk (35)")
        fig_hist.add_vline(x=70, line_dash="dash", line_color="#E53E3E", annotation_text="High Risk (70)")
        fig_hist.update_layout(template="plotly_white", xaxis_title="Risk Score", yaxis_title="Employee Count")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    st.markdown("### Department & Job Role Vulnerability Heatmap")
    heat_df = filtered_df.groupby(['Department', 'JobRole'])['Risk_Score'].mean().reset_index()
    fig_heat = px.density_heatmap(
        heat_df,
        x="Department",
        y="JobRole",
        z="Risk_Score",
        title="Mean Risk Score by Department & Role",
        color_continuous_scale="Reds"
    )
    fig_heat.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

# PAGE 2: INDIVIDUAL RISK PREDICTOR
elif nav_page == "👤 Individual Risk Predictor":
    st.markdown("<h1 class='main-title'>Individual Employee Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Select an employee or input custom details to get real-time risk score, SHAP risk drivers, and action plans.</p>", unsafe_allow_html=True)
    
    emp_ids = res_df['EmployeeNumber'].tolist() if 'EmployeeNumber' in res_df.columns else list(range(len(res_df)))
    selected_emp = st.selectbox("Select Employee Record ID", emp_ids)
    
    idx = emp_ids.index(selected_emp)
    row_raw = raw_df.iloc[idx]
    score = res_df.iloc[idx]['Risk_Score']
    tier = res_df.iloc[idx]['Risk_Tier']
    
    col_info, col_gauge = st.columns([2, 1])
    
    with col_info:
        st.markdown(f"### Employee Info (Record #{selected_emp})")
        st.write(f"**Department:** {row_raw['Department']} | **Job Role:** {row_raw['JobRole']}")
        st.write(f"**Age:** {row_raw['Age']} | **Monthly Income:** ${row_raw['MonthlyIncome']:,} | **OverTime:** `{row_raw['OverTime']}`")
        st.write(f"**Total Working Years:** {row_raw['TotalWorkingYears']} | **Years at Company:** {row_raw['YearsAtCompany']}")
        
        if tier == 'High Risk':
            st.markdown(f"**Risk Level:** <span class='risk-pill-high'>HIGH RISK ({score:.1f}%)</span>", unsafe_allow_html=True)
        elif tier == 'Medium Risk':
            st.markdown(f"**Risk Level:** <span class='risk-pill-med'>MEDIUM RISK ({score:.1f}%)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Risk Level:** <span class='risk-pill-low'>LOW RISK ({score:.1f}%)</span>", unsafe_allow_html=True)
            
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Attrition Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1A365D"},
                'steps': [
                    {'range': [0, 35], 'color': "#C6F6D5"},
                    {'range': [35, 70], 'color': "#FEFCBF"},
                    {'range': [70, 100], 'color': "#FED7D7"}
                ],
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Retention Action Plan
    st.markdown("### Recommended HR Retention Action Plan")
    if tier == 'High Risk':
        st.error("""
        **🚨 High Risk Action Needed:**
        1. Schedule immediate 1-on-1 stay interview within 48 hours.
        2. Evaluate compensation alignment against market benchmarks (Current: ${:,}/mo).
        3. Address OverTime burn-out (Current OverTime Status: {}).
        4. Review Stock Options level (Current Level: {}).
        """.format(row_raw['MonthlyIncome'], row_raw['OverTime'], row_raw['StockOptionLevel']))
    elif tier == 'Medium Risk':
        st.warning("""
        **⚠️ Medium Risk Recommendations:**
        1. Schedule quarterly career development review.
        2. Conduct work-life balance feedback session.
        3. Evaluate promotion timeline (Years since last promotion: {}).
        """.format(row_raw['YearsSinceLastPromotion']))
    else:
        st.success("""
        **✅ Low Risk Maintenance:**
        1. Maintain regular check-ins.
        2. Provide growth opportunities and recognition.
        """)

# PAGE 3: WHAT-IF POLICY SIMULATOR
elif nav_page == "💡 'What-If' HR Policy Simulator":
    st.markdown("<h1 class='main-title'>'What-If' HR Policy Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Simulate strategic workforce interventions (e.g. eliminating overtime, salary raises, stock grants) and quantify risk reduction & ROI.</p>", unsafe_allow_html=True)
    
    st.sidebar.subheader("Policy Intervention Controls")
    ot_policy = st.sidebar.radio("OverTime Policy", ["Keep Current", "Eliminate All OverTime"])
    salary_hike = st.sidebar.slider("Global Salary Raise (%)", 0, 25, 10, step=5)
    stock_grant = st.sidebar.slider("Stock Option Boost (+Levels)", 0, 2, 1)
    
    # Copy clean dataset for simulation
    sim_df = df_clean.copy()
    if 'Attrition' in sim_df.columns:
        sim_df = sim_df.drop(columns=['Attrition'])
        
    if ot_policy == "Eliminate All OverTime":
        sim_df['OverTime'] = 'No'
        
    if salary_hike > 0:
        sim_df['MonthlyIncome'] = (sim_df['MonthlyIncome'] * (1 + salary_hike / 100.0)).astype(int)
        
    if stock_grant > 0:
        sim_df['StockOptionLevel'] = (sim_df['StockOptionLevel'] + stock_grant).clip(upper=3)
        
    sim_trans = preprocessor.transform(sim_df)
    sim_probs = model.predict_proba(sim_trans)[:, 1]
    sim_scores = (sim_probs * 100).round(2)
    sim_tiers = pd.cut(sim_scores, bins=[-1, 35, 70, 100], labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    orig_high = (res_df['Risk_Tier'] == 'High Risk').sum()
    new_high = (sim_tiers == 'High Risk').sum()
    high_reduced = orig_high - new_high
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Baseline High Risk Count</div>
            <div class='metric-value'>{orig_high:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color: #38A169;'>
            <div class='metric-label'>Simulated High Risk Count</div>
            <div class='metric-value' style='color:#38A169;'>{new_high:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card' style='border-left-color: #3182CE;'>
            <div class='metric-label'>High Risk Reduction</div>
            <div class='metric-value'>-{high_reduced:,} ({high_reduced/orig_high*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Plot comparison
    comp_df = pd.DataFrame({
        'Baseline': res_df['Risk_Score'],
        'Simulated': sim_scores
    })
    fig_comp = px.histogram(
        comp_df,
        barmode="overlay",
        title="Risk Score Distribution Comparison (Baseline vs Policy Simulation)",
        color_discrete_map={'Baseline': '#E53E3E', 'Simulated': '#38A169'}
    )
    fig_comp.update_layout(template="plotly_white", xaxis_title="Risk Score", yaxis_title="Employee Count")
    st.plotly_chart(fig_comp, use_container_width=True)

# PAGE 4: BATCH RISK ASSESSOR
elif nav_page == "📁 Batch Risk Assessor":
    st.markdown("<h1 class='main-title'>Batch Employee Risk Assessor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Upload a CSV file containing employee attributes to score batch workforce attrition risk instantly.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Employee CSV File", type=["csv"])
    
    if uploaded_file is not None:
        user_batch = pd.read_csv(uploaded_file)
        cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Attrition']
        clean_batch = user_batch.drop(columns=[c for c in cols_to_drop if c in user_batch.columns])
        
        batch_trans = preprocessor.transform(clean_batch)
        batch_probs = model.predict_proba(batch_trans)[:, 1]
        batch_scores = (batch_probs * 100).round(2)
        batch_tiers = pd.cut(batch_scores, bins=[-1, 35, 70, 100], labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        out_df = user_batch.copy()
        out_df['Predicted_Risk_Score'] = batch_scores
        out_df['Predicted_Risk_Tier'] = batch_tiers
        
        st.markdown("### Batch Scoring Results")
        st.dataframe(out_df[['Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'Predicted_Risk_Score', 'Predicted_Risk_Tier']], use_container_width=True)
        
        csv_data = out_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scored Workforce CSV Report",
            data=csv_data,
            file_name="Scored_Employee_Workforce_Report.csv",
            mime="text/csv"
        )
    else:
        st.info("Tip: You can re-upload `HR_Attrition.csv` to test batch scoring capabilities.")

# PAGE 5: MODEL DIAGNOSTICS & SHAP
elif nav_page == "🔬 Model Diagnostics & SHAP":
    st.markdown("<h1 class='main-title'>Model Diagnostics & SHAP Explainability</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Inspect 5-Fold Stratified Cross Validation metrics, optimal decision threshold, and global feature importance.</p>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OOF ROC-AUC", f"{metadata['roc_auc']:.4f}")
    c2.metric("OOF PR-AUC", f"{metadata['pr_auc']:.4f}")
    c3.metric("F1-Score", f"{metadata['f1_score']:.4f}")
    c4.metric("Optimal Threshold", f"{metadata['optimal_threshold']:.2f}")
    
    if os.path.exists("models/shap_feature_importance.csv"):
        shap_df = pd.read_csv("models/shap_feature_importance.csv").head(15)
        fig_shap = px.bar(
            shap_df,
            x="mean_shap",
            y="feature",
            orientation="h",
            title="Top 15 Global Risk Drivers (Mean Absolute SHAP Value)",
            color="mean_shap",
            color_continuous_scale="Blues"
        )
        fig_shap.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP feature importance chart artifact not found.")
