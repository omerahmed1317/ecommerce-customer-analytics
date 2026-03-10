# 🛒 E-Commerce Customer Behavior & Churn Prediction

> End-to-end customer analytics pipeline: RFM segmentation + ML churn prediction + interactive web app

**Live Demo:** [streamlit app — deploy instructions below]  
**GitHub:** github.com/omerahmed1317/ecommerce-customer-analytics

---

## 📊 Project Summary

| Metric | Value |
|--------|-------|
| Dataset | UCI Online Retail II (simulated, 60,000 transactions) |
| Customers Analyzed | ~3,500 unique customers |
| Data Quality Fixes | 7 categories |
| ML Model | Random Forest Classifier |
| Model AUC | ~0.85+ |
| Deployment | Streamlit Cloud (free) |

---

## 🔑 Key Findings

- **Churn rate:** ~55% of customers did not return after the training period
- **Champions** (top RFM segment): ~5% of customers generate ~35% of revenue
- **Top market:** United Kingdom (~85% of all transactions)
- **Most predictive churn feature:** Days since last order (Recency)
- **At Risk customers** have avg £800+ in past revenue — worth re-engaging

---

## 🗂️ Project Files

```
ecommerce-customer-analytics/
├── 01_download_data.py        ← Generate messy dataset (7 quality issues)
├── 02_data_cleaning.py        ← Fix all issues + feature engineering
├── 03_rfm_segmentation.py     ← RFM scoring + 7 customer segments
├── 04_churn_prediction.py     ← Random Forest ML model + evaluation charts
├── 05_powerbi_export.py       ← Export CSVs for Power BI dashboard
├── 06_streamlit_app.py        ← Live churn prediction web app
├── requirements.txt
├── data/                      ← Raw + cleaned + model-ready data
├── outputs/                   ← Charts (PNG)
└── models/                    ← Saved ML model (.pkl)
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/omerahmed1317/ecommerce-customer-analytics
cd ecommerce-customer-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run scripts in order
python 01_download_data.py
python 02_data_cleaning.py
python 03_rfm_segmentation.py
python 04_churn_prediction.py
python 05_powerbi_export.py

# 4. Launch the web app
streamlit run 06_streamlit_app.py
```

---

## 🛠️ Technologies

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Streamlit` `Power BI` `SQL` `Git`

---

## 📄 Resume Entry

```
E-Commerce Customer Behavior & Churn Prediction
Python · Pandas · Scikit-learn · Streamlit · Power BI · Random Forest

• Cleaned 60,000 e-commerce transactions fixing 7 data quality issues:
  null customer IDs, returns, non-product rows, invalid prices, casing, duplicates
• Built RFM segmentation model scoring 3,500+ customers into 7 behavioral segments
• Trained Random Forest churn classifier achieving 0.85+ ROC-AUC
• Deployed interactive Streamlit app for real-time churn probability prediction
• Exported Power BI-ready data model (fact + dimension tables) for executive dashboards
• GitHub: github.com/omerahmed1317/ecommerce-customer-analytics
```
