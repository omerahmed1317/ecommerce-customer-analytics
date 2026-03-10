"""
================================================================
STEP 6: STREAMLIT CHURN PREDICTION APP
================================================================
E-Commerce Customer Analytics Project
File: 06_streamlit_app.py

PURPOSE:
  Interactive web app where you enter a customer's stats and
  get an instant churn probability prediction from the trained
  Random Forest model.

HOW TO RUN LOCALLY:
  pip install streamlit
  streamlit run 06_streamlit_app.py

HOW TO DEPLOY FREE:
  1. Push to GitHub
  2. Go to share.streamlit.io
  3. Connect repo → set main file = 06_streamlit_app.py
  4. Click Deploy
  → Live URL in ~2 minutes

WHAT THIS DEMONSTRATES:
  - You can turn a trained ML model into a usable product
  - Real-world ML deployment skill (not just Jupyter notebooks)
  - Stakeholders can interact with predictions without coding
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0D1117; }
  .stApp { background-color: #0D1117; }
  h1, h2, h3 { color: #FFFFFF !important; }
  .metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
  }
  .risk-high   { border-left: 4px solid #F85149 !important; }
  .risk-medium { border-left: 4px solid #FFB300 !important; }
  .risk-low    { border-left: 4px solid #3FB950 !important; }
  .stSlider > div > div > div { background-color: #FF4D00; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/churn_model.pkl"
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found. Please run 04_churn_prediction.py first.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    data = {}
    paths = {
        "customers": "data/powerbi_customers.csv",
        "rfm":       "data/rfm_segments.csv",
        "churn":     "data/churn_labels.csv",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
    return data

payload = load_model()
model        = payload["model"]
FEATURE_COLS = payload["feature_cols"]
model_auc    = payload["roc_auc"]
churn_rate   = payload["churn_rate"]
data         = load_data()

# ── HEADER ───────────────────────────────────────────────────────
st.markdown("## 🛒 Customer Churn Intelligence Platform")
st.markdown("*E-Commerce Customer Behavior & Churn Prediction — ML-Powered*")
st.divider()

# ── TOP METRICS ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

if "customers" in data:
    df_cust = data["customers"]
    total_customers   = len(df_cust)
    churned_customers = df_cust["is_churned"].sum() if "is_churned" in df_cust.columns else 0
    active_customers  = total_customers - churned_customers
    avg_revenue       = df_cust["total_revenue"].mean() if "total_revenue" in df_cust.columns else 0
else:
    total_customers, churned_customers, active_customers, avg_revenue = 0, 0, 0, 0

with col1:
    st.metric("Total Customers",    f"{total_customers:,}")
with col2:
    st.metric("Active Customers",   f"{active_customers:,}",
              delta=f"{active_customers/total_customers*100:.1f}%" if total_customers else None)
with col3:
    st.metric("Churned",            f"{churned_customers:,}",
              delta=f"-{churned_customers/total_customers*100:.1f}%" if total_customers else None,
              delta_color="inverse")
with col4:
    st.metric("Model ROC-AUC",      f"{model_auc:.3f}",
              delta="vs 0.500 random", delta_color="normal")

st.divider()

# ── MAIN LAYOUT ──────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6])

# ── LEFT: INPUT PANEL ────────────────────────────────────────────
with left_col:
    st.markdown("### 📊 Customer Profile Input")
    st.markdown("*Adjust the sliders to match a customer's behaviour:*")

    total_orders = st.slider(
        "Total Orders Placed", min_value=1, max_value=100,
        value=8, help="How many unique orders has this customer placed?"
    )
    total_revenue = st.number_input(
        "Total Revenue (GBP)", min_value=0.0, max_value=50000.0,
        value=450.0, step=50.0,
        help="Total amount spent by the customer across all orders"
    )
    avg_order_value = st.number_input(
        "Avg Order Value (GBP)", min_value=0.0, max_value=5000.0,
        value=56.0, step=5.0,
        help="Average spend per order"
    )
    last_order_days = st.slider(
        "Days Since Last Order", min_value=0, max_value=365,
        value=45,
        help="How many days ago did they last purchase? (lower = more recent)"
    )
    days_active = st.slider(
        "Days Active (customer lifetime)", min_value=0, max_value=730,
        value=180,
        help="Number of days between first and last purchase"
    )
    unique_products = st.slider(
        "Unique Products Bought", min_value=1, max_value=200,
        value=15,
        help="How many different product types has this customer purchased?"
    )

    st.markdown("##### Advanced")
    total_items = st.number_input(
        "Total Items Ordered", min_value=1, max_value=10000,
        value=total_orders * 6, step=10
    )
    num_countries = st.slider("Countries Ordered From", 1, 5, 1)

    # RFM scores — auto-estimate from inputs
    R_score = 5 if last_order_days < 30 else 4 if last_order_days < 60 else 3 if last_order_days < 120 else 2 if last_order_days < 200 else 1
    F_score = 5 if total_orders >= 30 else 4 if total_orders >= 15 else 3 if total_orders >= 7 else 2 if total_orders >= 3 else 1
    M_score = 5 if total_revenue >= 3000 else 4 if total_revenue >= 1000 else 3 if total_revenue >= 400 else 2 if total_revenue >= 100 else 1
    RFM_score = R_score + F_score + M_score

    order_rate        = total_orders / max(days_active, 1)
    avg_items_per_ord = total_items  / max(total_orders, 1)

    st.caption(f"Auto-calculated RFM: R={R_score} F={F_score} M={M_score} → Score={RFM_score}")

# ── RIGHT: PREDICTION PANEL ──────────────────────────────────────
with right_col:
    st.markdown("### 🎯 Churn Prediction")

    features_input = np.array([[
        total_orders, total_revenue, avg_order_value, total_items,
        unique_products, days_active, last_order_days,
        order_rate, avg_items_per_ord,
        R_score, F_score, M_score, RFM_score
    ]])

    churn_prob  = model.predict_proba(features_input)[0][1]
    churn_label = model.predict(features_input)[0]

    # Risk level
    if churn_prob >= 0.70:
        risk_level = "HIGH RISK"
        risk_color = "#F85149"
        risk_icon  = "🔴"
        risk_action = "Immediate re-engagement needed. Send personalised win-back email + discount."
    elif churn_prob >= 0.40:
        risk_level = "MEDIUM RISK"
        risk_color = "#FFB300"
        risk_icon  = "🟡"
        risk_action = "Monitor closely. Consider loyalty reward or product recommendation campaign."
    else:
        risk_level = "LOW RISK"
        risk_color = "#3FB950"
        risk_icon  = "🟢"
        risk_action = "Customer is healthy. Focus on upselling premium products."

    # Big prediction display
    st.markdown(f"""
    <div style="background:#161B22; border:1px solid #30363D; border-radius:12px;
                padding:30px; text-align:center; border-left: 6px solid {risk_color}; margin-bottom:16px;">
        <div style="font-size:48px; font-weight:900; color:{risk_color};">
            {churn_prob*100:.1f}%
        </div>
        <div style="font-size:18px; color:#C9D1D9; margin-top:4px;">
            Churn Probability
        </div>
        <div style="font-size:22px; font-weight:bold; color:{risk_color}; margin-top:12px;">
            {risk_icon} {risk_level}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action recommendation
    st.info(f"**Recommended Action:** {risk_action}")

    # Gauge chart
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    # Background arc (full semicircle)
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#21262D", linewidth=18, solid_capstyle="round")

    # Color zones
    zones = [
        (np.linspace(np.pi, np.pi*0.6, 100), "#3FB950"),    # green 0-40%
        (np.linspace(np.pi*0.6, np.pi*0.3, 100), "#FFB300"), # yellow 40-70%
        (np.linspace(np.pi*0.3, 0, 100), "#F85149"),         # red 70-100%
    ]
    for arc, color in zones:
        ax.plot(np.cos(arc), np.sin(arc), color=color, linewidth=18,
                solid_capstyle="butt", alpha=0.35)

    # Needle
    needle_angle = np.pi * (1 - churn_prob)
    ax.annotate("", xy=(0.7*np.cos(needle_angle), 0.7*np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=risk_color, lw=2.5))
    ax.plot(0, 0, "o", color=risk_color, markersize=10)

    ax.text(0, -0.25, f"{churn_prob*100:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=risk_color,
            fontfamily="monospace")
    ax.text(-0.9, -0.15, "0%",    ha="center", color="#8B949E", fontsize=9)
    ax.text(0,    0.85,  "50%",   ha="center", color="#8B949E", fontsize=9)
    ax.text(0.9,  -0.15, "100%",  ha="center", color="#8B949E", fontsize=9)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.4, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Feature breakdown
    st.markdown("##### Customer Score Breakdown")
    score_data = {
        "Recency Score":   (R_score, 5),
        "Frequency Score": (F_score, 5),
        "Monetary Score":  (M_score, 5),
    }
    for label, (score, max_score) in score_data.items():
        pct = score / max_score
        color = "#3FB950" if pct >= 0.6 else "#FFB300" if pct >= 0.4 else "#F85149"
        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#C9D1D9; font-size:13px;">{label}</span>
                <span style="color:{color}; font-weight:bold; font-size:13px;">{score}/{max_score}</span>
            </div>
            <div style="background:#21262D; border-radius:4px; height:8px;">
                <div style="width:{pct*100}%; background:{color}; height:8px; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── SEGMENT ANALYSIS TAB ─────────────────────────────────────────
st.divider()
st.markdown("### 📈 Portfolio-Wide Segment Analysis")

if "rfm" in data:
    rfm_df = data["rfm"]
    seg_counts = rfm_df["Segment"].value_counts()

    SEGMENT_COLORS = {
        "Champions":           "#FFB300",
        "Loyal":               "#3FB950",
        "Potential Loyalists": "#00B4D8",
        "Promising":           "#A371F7",
        "At Risk":             "#FF4D00",
        "Cannot Lose Them":    "#F85149",
        "Lost":                "#484F58",
    }

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        colors = [SEGMENT_COLORS.get(s, "#00B4D8") for s in seg_counts.index]
        bars = ax.barh(seg_counts.index, seg_counts.values,
                       color=colors, height=0.6, edgecolor="none")
        for bar, val in zip(bars, seg_counts.values):
            ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9, color="#C9D1D9")
        ax.set_xlabel("Customers", color="#8B949E")
        ax.set_title("Customers per Segment", color="#FFFFFF", fontweight="bold")
        ax.tick_params(colors="#8B949E")
        for spine in ax.spines.values(): spine.set_color("#30363D")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        seg_revenue = rfm_df.groupby("Segment")["monetary"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        colors = [SEGMENT_COLORS.get(s, "#00B4D8") for s in seg_revenue.index]
        ax.bar(seg_revenue.index, seg_revenue.values,
               color=colors, edgecolor="none", width=0.65)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"GBP{int(x/1000)}k")
        )
        ax.set_title("Revenue by Segment", color="#FFFFFF", fontweight="bold")
        ax.tick_params(colors="#8B949E", axis="both")
        plt.xticks(rotation=30, ha="right", fontsize=8)
        for spine in ax.spines.values(): spine.set_color("#30363D")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ── FOOTER ───────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#484F58; font-size:12px; padding:16px;">
    E-Commerce Customer Analytics · Python · Scikit-learn · Streamlit · Random Forest<br>
    <a href="https://github.com/omerahmed1317" style="color:#00B4D8;">github.com/omerahmed1317</a>
</div>
""", unsafe_allow_html=True)
