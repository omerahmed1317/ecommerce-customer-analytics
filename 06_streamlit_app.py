import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="wide",
)

st.markdown("""
<style>
  .stApp { background-color: #0D1117; }
  h1, h2, h3 { color: #FFFFFF !important; }
  .stSlider > div > div > div { background-color: #FF4D00; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "models/churn_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found. Please run 04_churn_prediction.py first.")
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

payload      = load_model()
model        = payload["model"]
FEATURE_COLS = payload["feature_cols"]
model_auc    = payload["roc_auc"]
data         = load_data()

st.markdown("## 🛒 Customer Churn Intelligence Platform")
st.markdown("*E-Commerce Customer Behavior & Churn Prediction — ML-Powered*")
st.divider()

col1, col2, col3, col4 = st.columns(4)
if "customers" in data:
    df_cust = data["customers"]
    total_customers   = len(df_cust)
    churned_customers = int(df_cust["is_churned"].sum()) if "is_churned" in df_cust.columns else 0
    active_customers  = total_customers - churned_customers
else:
    total_customers, churned_customers, active_customers = 0, 0, 0

with col1: st.metric("Total Customers", f"{total_customers:,}")
with col2: st.metric("Active Customers", f"{active_customers:,}")
with col3: st.metric("Churned", f"{churned_customers:,}")
with col4: st.metric("Model ROC-AUC", f"{model_auc:.3f}", delta="vs 0.500 random")

st.divider()

left_col, right_col = st.columns([1, 1.6])

with left_col:
    st.markdown("### 📊 Customer Profile Input")
    st.markdown("*Adjust the sliders to match a customer's behaviour:*")
    total_orders    = st.slider("Total Orders Placed", 1, 100, 8)
    total_revenue   = st.number_input("Total Revenue (GBP)", 0.0, 50000.0, 450.0, 50.0)
    avg_order_value = st.number_input("Avg Order Value (GBP)", 0.0, 5000.0, 56.0, 5.0)
    last_order_days = st.slider("Days Since Last Order", 0, 365, 45)
    days_active     = st.slider("Days Active (customer lifetime)", 0, 730, 180)
    unique_products = st.slider("Unique Products Bought", 1, 200, 15)
    st.markdown("##### Advanced")
    total_items  = st.number_input("Total Items Ordered", 1, 10000, total_orders * 6, 10)
    num_countries = st.slider("Countries Ordered From", 1, 5, 1)

    R_score = 5 if last_order_days < 30 else 4 if last_order_days < 60 else 3 if last_order_days < 120 else 2 if last_order_days < 200 else 1
    F_score = 5 if total_orders >= 30 else 4 if total_orders >= 15 else 3 if total_orders >= 7 else 2 if total_orders >= 3 else 1
    M_score = 5 if total_revenue >= 3000 else 4 if total_revenue >= 1000 else 3 if total_revenue >= 400 else 2 if total_revenue >= 100 else 1
    RFM_score = R_score + F_score + M_score
    order_rate        = total_orders / max(days_active, 1)
    avg_items_per_ord = total_items  / max(total_orders, 1)
    st.caption(f"Auto-calculated RFM: R={R_score} F={F_score} M={M_score} -> Score={RFM_score}")

with right_col:
    st.markdown("### 🎯 Churn Prediction")

    features_input = np.array([[
        total_orders, total_revenue, avg_order_value, total_items,
        unique_products, days_active, last_order_days,
        order_rate, avg_items_per_ord,
        R_score, F_score, M_score, RFM_score
    ]])

    churn_prob = model.predict_proba(features_input)[0][1]

    if churn_prob >= 0.70:
        risk_level = "HIGH RISK";   risk_color = "#F85149"; risk_icon = "🔴"
        risk_action = "Immediate re-engagement needed. Send personalised win-back email + discount."
    elif churn_prob >= 0.40:
        risk_level = "MEDIUM RISK"; risk_color = "#FFB300"; risk_icon = "🟡"
        risk_action = "Monitor closely. Consider loyalty reward or product recommendation campaign."
    else:
        risk_level = "LOW RISK";    risk_color = "#3FB950"; risk_icon = "🟢"
        risk_action = "Customer is healthy. Focus on upselling premium products."

    st.markdown(f"""
    <div style="background:#161B22; border:1px solid #30363D; border-radius:12px;
                padding:30px; text-align:center; border-left:6px solid {risk_color}; margin-bottom:16px;">
        <div style="font-size:48px; font-weight:900; color:{risk_color};">{churn_prob*100:.1f}%</div>
        <div style="font-size:18px; color:#C9D1D9; margin-top:4px;">Churn Probability</div>
        <div style="font-size:22px; font-weight:bold; color:{risk_color}; margin-top:12px;">{risk_icon} {risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"**Recommended Action:** {risk_action}")

    # Plotly gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        number={"suffix": "%", "font": {"color": risk_color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
            "bar":  {"color": risk_color},
            "bgcolor": "#161B22",
            "steps": [
                {"range": [0,  40], "color": "#0D2818"},
                {"range": [40, 70], "color": "#2D1F00"},
                {"range": [70, 100],"color": "#2D0A0A"},
            ],
            "threshold": {"line": {"color": risk_color, "width": 4}, "value": churn_prob * 100},
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0D1117", font_color="#C9D1D9",
        height=280, margin=dict(t=20, b=10, l=30, r=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Customer Score Breakdown")
    for label, score in [("Recency Score", R_score), ("Frequency Score", F_score), ("Monetary Score", M_score)]:
        pct = score / 5
        color = "#3FB950" if pct >= 0.6 else "#FFB300" if pct >= 0.4 else "#F85149"
        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#C9D1D9; font-size:13px;">{label}</span>
                <span style="color:{color}; font-weight:bold; font-size:13px;">{score}/5</span>
            </div>
            <div style="background:#21262D; border-radius:4px; height:8px;">
                <div style="width:{pct*100}%; background:{color}; height:8px; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.markdown("### 📈 Portfolio-Wide Segment Analysis")

if "rfm" in data:
    rfm_df = data["rfm"]
    seg_counts = rfm_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    SEGMENT_COLORS = {
        "Champions": "#FFB300", "Loyal": "#3FB950",
        "Potential Loyalists": "#00B4D8", "Promising": "#A371F7",
        "At Risk": "#FF4D00", "Lost": "#484F58",
    }

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(seg_counts, x="Customers", y="Segment", orientation="h",
                      color="Segment", color_discrete_map=SEGMENT_COLORS,
                      title="Customers per Segment")
        fig1.update_layout(paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                           font_color="#C9D1D9", showlegend=False,
                           title_font_color="#FFFFFF")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        seg_rev = rfm_df.groupby("Segment")["monetary"].sum().reset_index()
        seg_rev.columns = ["Segment", "Revenue"]
        seg_rev = seg_rev.sort_values("Revenue", ascending=False)
        fig2 = px.bar(seg_rev, x="Segment", y="Revenue",
                      color="Segment", color_discrete_map=SEGMENT_COLORS,
                      title="Revenue by Segment")
        fig2.update_layout(paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                           font_color="#C9D1D9", showlegend=False,
                           title_font_color="#FFFFFF")
        fig2.update_xaxes(tickangle=30)
        st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.markdown("""
<div style="text-align:center; color:#484F58; font-size:12px; padding:16px;">
    E-Commerce Customer Analytics · Python · Scikit-learn · Streamlit · Random Forest<br>
    <a href="https://github.com/omerahmed1317" style="color:#00B4D8;">github.com/omerahmed1317</a>
</div>
""", unsafe_allow_html=True)
