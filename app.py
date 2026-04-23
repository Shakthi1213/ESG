"""
ESG Score & Risk Prediction System
====================================
A complete Streamlit application that:
  1. Predicts ESG Score (regression) from financial & sustainability inputs
  2. Uses the predicted ESG Score + risk features to classify ESG Risk Level
  3. Displays results with actionable recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Score & Risk Prediction System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main header */
.main-header {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
}
.main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

/* Result cards */
.result-card {
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.score-card  { background: #e8f5e9; border-left: 6px solid #2d6a4f; }
.risk-high   { background: #ffebee; border-left: 6px solid #c62828; }
.risk-medium { background: #fff8e1; border-left: 6px solid #f9a825; }
.risk-low    { background: #e8f5e9; border-left: 6px solid #2e7d32; }

/* Recommendation box */
st.markdown("""
<style>
.rec-box {
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 0.5rem;
    font-size: 16px;
    font-weight: 500;
}

/* High Risk */
.rec-high { 
    background:#8B0000; 
    border:1px solid #5A0000; 
    color:white; 
}

/* Medium Risk */
.rec-medium { 
    background:#B8860B; 
    border:1px solid #8B6508; 
    color:white; 
}

/* Low Risk */
.rec-low { 
    background:#006400; 
    border:1px solid #004d00; 
    color:white; 
}
</style>
""", unsafe_allow_html=True)

/* Step badges */
.step-badge {
    display: inline-block;
    background: #2d6a4f;
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    font-weight: bold;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(__file__)

@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    """Load all saved model artifacts. Returns dict or raises on failure."""
    files = {
        "score_model":    "esg_score_model.pkl",
        "risk_model":     "esg_risk_model_modelC.pkl",
        "scaler":         "scaler__1_.pkl",
        "score_features": "esg_score_features.pkl",
        "risk_features":  "esg_risk_features_modelC.pkl",
    }
    artifacts = {}
    for key, fname in files.items():
        path = os.path.join(MODEL_DIR, fname)
        try:
            artifacts[key] = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                artifacts[key] = pickle.load(f)
    return artifacts


def safe_load():
    try:
        return load_artifacts(), None
    except Exception as e:
        return None, str(e)


artifacts, load_error = safe_load()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🌱 ESG Score &amp; Risk Prediction System</h1>
  <p>AI-powered environmental, social &amp; governance analytics — predict ESG scores and classify risk levels in real time.</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Could not load model files: {load_error}\n\nMake sure all `.pkl` files are in the same folder as `app.py`.")
    st.stop()

score_features = artifacts["score_features"]   # list of 8 feature names
risk_features  = artifacts["risk_features"]    # list of 1000+ feature names (OHE)
score_model    = artifacts["score_model"]
risk_model     = artifacts["risk_model"]
scaler         = artifacts["scaler"]

# ── Sidebar: user inputs ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Parameters")
    st.markdown("Fill in your company's financial and sustainability data below.")

    st.markdown("### 📊 Financial Metrics")
    year        = st.number_input("Year",            min_value=2000, max_value=2030, value=2023, step=1)
    ebit        = st.number_input("EBIT (USD M)",    min_value=-5000.0, max_value=50000.0, value=500.0, step=10.0,
                                  help="Earnings Before Interest and Taxes")
    roe         = st.number_input("ROE (%)",         min_value=-100.0, max_value=200.0,  value=12.0,  step=0.5,
                                  help="Return on Equity")
    revenue     = st.number_input("Revenue (USD M)", min_value=0.0,    max_value=500000.0, value=1000.0, step=10.0)
    profit_margin = st.number_input("Profit Margin (%)", min_value=-100.0, max_value=100.0, value=10.0, step=0.5)
    market_cap  = st.number_input("Market Cap (USD M)", min_value=0.0, max_value=2000000.0, value=5000.0, step=50.0)
    growth_rate = st.number_input("Growth Rate (%)",  min_value=-100.0, max_value=500.0, value=5.0,  step=0.5)

    st.markdown("### 🌿 ESG / Sustainability Metrics")
    e_score     = st.slider("Environmental Score (E)",  0.0, 100.0, 55.0, 0.5)
    g_score     = st.slider("Governance Score (G)",     0.0, 100.0, 60.0, 0.5)
    csr         = st.number_input("CSR Spending (USD M)", min_value=0.0, max_value=5000.0, value=50.0, step=1.0,
                                  help="Corporate Social Responsibility expenditure")
    percent_et  = st.number_input("% Energy from Renewables (Percent_ET)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
    percent_w   = st.number_input("% Water Recycled (Percent_W)",           min_value=0.0, max_value=100.0, value=25.0, step=0.5)

    st.markdown("### 🏭 Environmental Impact")
    carbon_emissions   = st.number_input("Carbon Emissions (tonnes)", min_value=0.0, max_value=10_000_000.0, value=35000.0, step=100.0)
    water_usage        = st.number_input("Water Usage (m³)",           min_value=0.0, max_value=10_000_000.0, value=18000.0, step=100.0)
    energy_consumption = st.number_input("Energy Consumption (MWh)",   min_value=0.0, max_value=10_000_000.0, value=70000.0, step=100.0)

    st.markdown("### 🏢 Company Profile")
    industry = st.selectbox("Industry", ["Energy", "Finance", "Healthcare", "Manufacturing",
                                          "Retail", "Technology", "Transportation", "Utilities"])
    region   = st.selectbox("Region",   ["Asia", "Europe", "Latin America", "Middle East",
                                          "North America", "Oceania"])

    company_id = st.number_input("Company ID (if known)", min_value=1, max_value=10000, value=1, step=1)

    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

# ── Helper: build score input ──────────────────────────────────────────────────
def build_score_input():
    """Build a DataFrame exactly matching score_features."""
    raw = {
        "Year":       year,
        "E_score":    e_score,
        "G_score":    g_score,
        "Percent_ET": percent_et,
        "Percent_W":  percent_w,
        "CSR":        csr,
        "EBIT":       ebit,
        "ROE":        roe,
    }
    df = pd.DataFrame([{col: raw.get(col, 0.0) for col in score_features}])
    return df


# ── Helper: build risk input ───────────────────────────────────────────────────
def build_risk_input(predicted_esg_score: float):
    """
    Build a DataFrame exactly matching risk_features (one-hot encoded).
    All OHE columns default to 0; only the selected company/industry/region gets 1.
    """
    row = {col: 0 for col in risk_features}

    # Numeric columns we can fill directly
    numeric_map = {
        "CompanyID":         company_id,
        "Year":              year,
        "Revenue":           revenue,
        "ProfitMargin":      profit_margin,
        "MarketCap":         market_cap,
        "GrowthRate":        growth_rate,
        "CarbonEmissions":   carbon_emissions,
        "WaterUsage":        water_usage,
        "EnergyConsumption": energy_consumption,
        # inject predicted ESG score if feature present
        "ESG_Score":         predicted_esg_score,
        "ESG_Overall":       predicted_esg_score,
        "ESG_score":         predicted_esg_score,
    }
    for k, v in numeric_map.items():
        if k in row:
            row[k] = v

    # One-hot: CompanyName
    cn_col = f"CompanyName_Company_{company_id}"
    if cn_col in row:
        row[cn_col] = 1

    # One-hot: Industry
    ind_col = f"Industry_{industry}"
    if ind_col in row:
        row[ind_col] = 1

    # One-hot: Region
    reg_col = f"Region_{region}"
    if reg_col in row:
        row[reg_col] = 1

    df = pd.DataFrame([row])
    # Ensure correct column order
    df = df[risk_features]
    return df


# ── Risk label mapping ─────────────────────────────────────────────────────────
RISK_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
RISK_CSS   = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
RISK_REC_CSS = {"High": "rec-high", "Medium": "rec-medium", "Low": "rec-low"}

RECOMMENDATIONS = {
    "High": [
        ("🌫️ Reduce Carbon Emissions",
         "Implement carbon capture technologies, transition to renewable energy sources, and set science-based emission reduction targets aligned with net-zero goals."),
        ("🏛️ Improve Governance Structure",
         "Establish an independent ESG board committee, strengthen audit functions, improve executive accountability, and adopt best-practice corporate governance codes."),
        ("💚 Increase CSR Spending",
         "Allocate a higher proportion of profits toward community development, employee well-being, and environmental remediation projects."),
        ("📋 Improve Transparency & Reporting",
         "Adopt GRI or SASB reporting standards, publish an annual ESG report, and disclose material ESG risks in financial filings."),
    ],
    "Medium": [
        ("⚡ Improve Environmental Efficiency",
         "Conduct energy audits, invest in efficient machinery, and target a 15–20% reduction in your environmental footprint over the next three years."),
        ("💧 Optimize Energy & Water Usage",
         "Install smart metering, implement water recycling systems, and source at least 40% of energy from renewables within two years."),
        ("🤝 Strengthen Social Initiatives",
         "Expand employee diversity & inclusion programs, improve supply chain labour standards, and increase community engagement activities."),
    ],
    "Low": [
        ("🌿 Maintain Sustainability Practices",
         "Continue current ESG programs and pursue ISO 14001 environmental certification to formalise your management systems."),
        ("🔬 Invest in Green Innovation",
         "Channel R&D investment into clean technology, circular-economy products, and sustainable supply chain solutions to stay ahead of regulation."),
        ("📈 Strengthen Long-Term ESG Strategy",
         "Set ambitious 2030 ESG targets, integrate ESG KPIs into executive compensation, and engage institutional investors on your ESG roadmap."),
    ],
}


# ── Main content ───────────────────────────────────────────────────────────────
col_info, col_results = st.columns([1, 2], gap="large")

with col_info:
    st.markdown("### 📋 How it Works")
    st.markdown("""
<div style="background:#0B3D91;border-radius:10px;padding:1.2rem;">
<p><span class="step-badge">1</span><strong>Input Data</strong><br>
Enter your company's financial and ESG metrics in the sidebar.</p>
<p><span class="step-badge">2</span><strong>Score Prediction</strong><br>
The regression model predicts your overall ESG score.</p>
<p><span class="step-badge">3</span><strong>Risk Classification</strong><br>
The predicted score is fed into the risk classifier alongside your financial profile.</p>
<p style="margin-bottom:0"><span class="step-badge">4</span><strong>Recommendations</strong><br>
Tailored, actionable guidance is generated based on your risk level.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🎯 Score Features Used")
    st.code("\n".join(score_features), language="text")

with col_results:
    if not predict_btn:
        st.info("👈 Fill in your company details in the sidebar and click **Run Prediction** to get started.")
    else:
        with st.spinner("Running predictions…"):
            # ── Step 1: ESG Score prediction ───────────────────────────────────
            try:
                X_score = build_score_input()
                # Apply scaler (fitted on score features)
                try:
                    X_score_scaled = scaler.transform(X_score)
                except Exception:
                    X_score_scaled = X_score.values  # fallback: no scaling

                predicted_score = float(score_model.predict(X_score_scaled)[0])
                predicted_score = max(0.0, min(100.0, predicted_score))  # clamp to [0,100]
                score_ok = True
            except Exception as e:
                score_ok = False
                score_error = str(e)

            # ── Step 2: ESG Risk classification ───────────────────────────────
            if score_ok:
                try:
                    X_risk = build_risk_input(predicted_score)
                    raw_pred = risk_model.predict(X_risk)[0]
                    # Normalise label to High / Medium / Low
                    label_map = {
                        0: "Low", 1: "Medium", 2: "High",
                        "0": "Low", "1": "Medium", "2": "High",
                        "low": "Low", "medium": "Medium", "high": "High",
                        "Low": "Low", "Medium": "Medium", "High": "High",
                    }
                    risk_label = label_map.get(raw_pred, str(raw_pred).capitalize())
                    if risk_label not in ("Low", "Medium", "High"):
                        risk_label = "Medium"  # safe fallback
                    risk_ok = True
                except Exception as e:
                    risk_ok = False
                    risk_error = str(e)

        # ── Display results ────────────────────────────────────────────────────
        if not score_ok:
            st.error(f"ESG Score prediction failed: {score_error}")
        else:
            st.markdown("## 📊 Prediction Results")
            r1, r2 = st.columns(2)

            with r1:
                score_color = (
                    "#c62828" if predicted_score < 40 else
                    "#f9a825" if predicted_score < 65 else
                    "#2e7d32"
                )
                st.markdown(f"""
<div class="result-card score-card">
  <div style="font-size:0.9rem;color:#555;font-weight:600;margin-bottom:4px;">PREDICTED ESG SCORE</div>
  <div style="font-size:3rem;font-weight:800;color:{score_color};">{predicted_score:.1f}</div>
  <div style="font-size:0.85rem;color:#777;">out of 100</div>
</div>
""", unsafe_allow_html=True)

            with r2:
                if not risk_ok:
                    st.error(f"Risk classification failed: {risk_error}")
                else:
                    emoji = RISK_EMOJI.get(risk_label, "⚪")
                    css   = RISK_CSS.get(risk_label, "")
                    st.markdown(f"""
<div class="result-card {css}">
  <div style="font-size:0.9rem;color:#555;font-weight:600;margin-bottom:4px;">ESG RISK LEVEL</div>
  <div style="font-size:2.5rem;font-weight:800;">{emoji} {risk_label}</div>
  <div style="font-size:0.85rem;color:#777;">Classification result</div>
</div>
""", unsafe_allow_html=True)

            # ── Score gauge bar ────────────────────────────────────────────────
            pct = int(predicted_score)
            bar_color = "#c62828" if pct < 40 else "#f9a825" if pct < 65 else "#40916c"
            st.markdown(f"""
<div style="background:#e0e0e0;border-radius:20px;height:18px;margin:0.5rem 0 1.5rem;">
  <div style="width:{pct}%;background:{bar_color};height:18px;border-radius:20px;
              display:flex;align-items:center;justify-content:flex-end;padding-right:8px;">
    <span style="color:white;font-size:0.75rem;font-weight:700;">{pct}</span>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Recommendations ────────────────────────────────────────────────
            if risk_ok:
                rec_css = RISK_REC_CSS.get(risk_label, "")
                recs    = RECOMMENDATIONS.get(risk_label, [])

                st.markdown("## 💡 Recommendations")
                st.markdown(f"""
<div class="rec-box {rec_css}">
  <h4 style="margin-top:0;">{RISK_EMOJI.get(risk_label,'')} {risk_label} Risk — Action Plan</h4>
""", unsafe_allow_html=True)

                for title, detail in recs:
                    st.markdown(f"""
<div style="margin-bottom:1rem;">
  <strong>{title}</strong><br>
  <span style="font-size:0.9rem;color:#444;">{detail}</span>
</div>
""", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Input summary table ────────────────────────────────────────
                with st.expander("🔍 View Input Summary"):
                    summary = {
                        "Year": year, "Industry": industry, "Region": region,
                        "EBIT": ebit, "ROE (%)": roe, "Revenue": revenue,
                        "Profit Margin (%)": profit_margin, "Market Cap": market_cap,
                        "Growth Rate (%)": growth_rate,
                        "E Score": e_score, "G Score": g_score,
                        "CSR Spending": csr,
                        "% Energy Renewable": percent_et, "% Water Recycled": percent_w,
                        "Carbon Emissions": carbon_emissions,
                        "Water Usage": water_usage, "Energy Consumption": energy_consumption,
                    }
                    st.dataframe(pd.DataFrame(summary, index=["Value"]).T.rename(columns={"Value": "Input"}),
                                 use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem;'>"
    "ESG Score &amp; Risk Prediction System &nbsp;|&nbsp; Powered by Scikit-learn &amp; Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
