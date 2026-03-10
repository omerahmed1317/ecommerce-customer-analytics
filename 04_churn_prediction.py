"""
================================================================
STEP 4: CHURN PREDICTION — MACHINE LEARNING MODEL
================================================================
E-Commerce Customer Analytics Project
File: 04_churn_prediction.py

WHAT IS CHURN?
  A customer "churns" when they stop buying from you.
  We define churn as: no purchase in the last 90 days of the dataset.

ML PIPELINE:
  1. Define churn label (binary: 1 = churned, 0 = active)
  2. Engineer features per customer
  3. Split into train/test sets
  4. Train Random Forest classifier
  5. Evaluate with confusion matrix + ROC-AUC
  6. Plot feature importance
  7. Save the model for the Streamlit app

OUTPUT:
  models/churn_model.pkl
  data/churn_labels.csv
  outputs/04_churn_feature_importance.png
  outputs/05_confusion_matrix.png
  outputs/06_roc_curve.png
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import os
from datetime import timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#C9D1D9",
    "axes.titlecolor":   "#FFFFFF",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "text.color":        "#C9D1D9",
    "grid.color":        "#21262D",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
    "figure.dpi":        120,
})

ACCENT  = "#00B4D8"
GREEN   = "#3FB950"
RED     = "#F85149"
ORANGE  = "#FF4D00"
YELLOW  = "#FFB300"

print("=" * 60)
print("  E-COMMERCE ANALYTICS — CHURN PREDICTION")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────────────
df  = pd.read_csv("data/cleaned_transactions.csv", parse_dates=["InvoiceDate"])
rfm = pd.read_csv("data/rfm_segments.csv")
print(f"\n[LOAD] {len(df):,} transactions | {df['Customer ID'].nunique():,} customers")

# ── DEFINE CHURN ──────────────────────────────────────────────────
print("\n[CHURN] Defining churn labels...")

# Split dataset: use first 75% to build features, last 25% to check churn
total_days = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days
split_date = df["InvoiceDate"].min() + timedelta(days=int(total_days * 0.75))
CHURN_WINDOW = 90  # days

train_df    = df[df["InvoiceDate"] <= split_date].copy()
future_df   = df[df["InvoiceDate"] >  split_date].copy()
future_end  = df["InvoiceDate"].max()

# Customers who bought in future period = NOT churned
active_in_future  = set(future_df["Customer ID"].unique())
# Customers who bought in training period (potential churners)
train_customers   = set(train_df["Customer ID"].unique())

# Churn = was a customer during training, did NOT buy in future window
churn_labels = []
for cid in train_customers:
    is_churned = 0 if cid in active_in_future else 1
    churn_labels.append({"Customer ID": cid, "is_churned": is_churned})

churn_df = pd.DataFrame(churn_labels)
churn_rate = churn_df["is_churned"].mean()
print(f"  Churn definition: no purchase after {split_date.date()}")
print(f"  Total customers:  {len(churn_df):,}")
print(f"  Churned:          {churn_df['is_churned'].sum():,} ({churn_rate*100:.1f}%)")
print(f"  Active:           {(churn_df['is_churned']==0).sum():,} ({(1-churn_rate)*100:.1f}%)")

# ── FEATURE ENGINEERING ──────────────────────────────────────────
print("\n[FEATURES] Building customer features from training period...")

features = train_df.groupby("Customer ID").agg(
    total_orders    = ("Invoice", "nunique"),
    total_revenue   = ("TotalRevenue", "sum"),
    avg_order_value = ("TotalRevenue", "mean"),
    total_items     = ("Quantity", "sum"),
    unique_products = ("StockCode", "nunique"),
    num_countries   = ("Country", "nunique"),
    days_active     = ("InvoiceDate", lambda x: (x.max() - x.min()).days),
    last_order_days = ("InvoiceDate",
                       lambda x: (split_date - x.max()).days),
).reset_index()

# Purchase frequency rate (orders per active day)
features["order_rate"] = (
    features["total_orders"] /
    features["days_active"].replace(0, 1)
)

# Average items per order
features["avg_items_per_order"] = (
    features["total_items"] /
    features["total_orders"].replace(0, 1)
)

# Merge with churn labels
features = features.merge(churn_df, on="Customer ID", how="inner")
features = features.merge(
    rfm[["Customer ID", "R_score", "F_score", "M_score", "RFM_Score", "Segment"]],
    on="Customer ID", how="left"
)

print(f"  Features engineered: {len(features.columns)-2} features for {len(features):,} customers")
print(f"  Feature list: {[c for c in features.columns if c not in ['Customer ID','is_churned','Segment']]}")

# ── PREPARE ML DATA ───────────────────────────────────────────────
print("\n[ML] Preparing training and test sets...")

FEATURE_COLS = [
    "total_orders", "total_revenue", "avg_order_value", "total_items",
    "unique_products", "days_active", "last_order_days",
    "order_rate", "avg_items_per_order",
    "R_score", "F_score", "M_score", "RFM_Score"
]

# Drop any rows with NaN in features
ml_data = features.dropna(subset=FEATURE_COLS + ["is_churned"]).copy()

X = ml_data[FEATURE_COLS].values
y = ml_data["is_churned"].values

# Stratified split preserves churn ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set:  {len(X_train):,} customers")
print(f"  Test set:      {len(X_test):,} customers")
print(f"  Churn rate in train: {y_train.mean()*100:.1f}%")
print(f"  Churn rate in test:  {y_test.mean()*100:.1f}%")

# ── TRAIN MODEL ───────────────────────────────────────────────────
print("\n[TRAIN] Training Random Forest classifier...")

# WHY Random Forest:
# - Handles mixed feature types naturally
# - Gives feature importance scores
# - Robust to outliers
# - No need to scale features
# - Interpretable enough for business stakeholders

model = RandomForestClassifier(
    n_estimators=200,     # 200 trees in the forest
    max_depth=8,          # limit depth to prevent overfitting
    min_samples_leaf=5,   # each leaf needs at least 5 samples
    class_weight="balanced",  # handle class imbalance automatically
    random_state=42,
    n_jobs=-1             # use all CPU cores
)
model.fit(X_train, y_train)
print("  ✅ Model trained")

# Cross-validation score (more reliable than single train/test split)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"  Cross-validation ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── EVALUATE ──────────────────────────────────────────────────────
print("\n[EVALUATE] Model performance on test set...")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc     = roc_auc_score(y_test, y_pred_prob)

print(f"\n  ROC-AUC Score: {roc_auc:.3f}  (1.0 = perfect, 0.5 = random)")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Active (0)", "Churned (1)"]))

# ── CHART 1: FEATURE IMPORTANCE ──────────────────────────────────
print("[CHART 1] Feature importance...")

importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = [ORANGE if v > importances.quantile(0.75) else ACCENT
          for v in importances.values]
bars = ax.barh(importances.index, importances.values,
               color=colors, height=0.65, edgecolor="none")
for bar, val in zip(bars, importances.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance Score")
ax.set_title("What Predicts Churn? — Random Forest Feature Importance",
             fontsize=13, fontweight="bold", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=ORANGE, label="Top predictors"),
    Patch(facecolor=ACCENT,  label="Supporting features"),
], fontsize=9, facecolor="#161B22", edgecolor="#30363D", loc="lower right")
plt.tight_layout()
plt.savefig("outputs/04_churn_feature_importance.png", bbox_inches="tight")
plt.close()
print("  ✅ outputs/04_churn_feature_importance.png")

# ── CHART 2: CONFUSION MATRIX ────────────────────────────────────
print("[CHART 2] Confusion matrix...")

cm   = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Active", "Churned"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Churn Prediction",
             fontsize=13, fontweight="bold", pad=12)
ax.set_facecolor("#161B22")
fig.set_facecolor("#0D1117")
plt.tight_layout()
plt.savefig("outputs/05_confusion_matrix.png", bbox_inches="tight")
plt.close()
print("  ✅ outputs/05_confusion_matrix.png")

# ── CHART 3: ROC CURVE ───────────────────────────────────────────
print("[CHART 3] ROC curve...")

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=ORANGE, linewidth=2.5,
        label=f"Random Forest (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#484F58", linestyle="--",
        linewidth=1.5, label="Random classifier (AUC = 0.500)")
ax.fill_between(fpr, tpr, alpha=0.15, color=ORANGE)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Churn Prediction Model",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10, facecolor="#161B22", edgecolor="#30363D")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/06_roc_curve.png", bbox_inches="tight")
plt.close()
print("  ✅ outputs/06_roc_curve.png")

# ── SAVE MODEL ───────────────────────────────────────────────────
print("\n[SAVE] Saving model and data...")

model_payload = {
    "model":        model,
    "feature_cols": FEATURE_COLS,
    "roc_auc":      roc_auc,
    "churn_rate":   churn_rate,
}
with open("models/churn_model.pkl", "wb") as f:
    pickle.dump(model_payload, f)
print("  ✅ models/churn_model.pkl")

ml_data[["Customer ID", "is_churned"] + FEATURE_COLS].to_csv(
    "data/churn_labels.csv", index=False
)
print(f"  ✅ data/churn_labels.csv ({len(ml_data):,} customers)")

# ── INSIGHTS ──────────────────────────────────────────────────────
top_feature = importances.index[-1]
print(f"\n{'='*60}")
print("  CHURN MODEL INSIGHTS")
print(f"{'='*60}")
print(f"\n  Model accuracy (ROC-AUC): {roc_auc:.1%}")
print(f"  Most predictive feature:  {top_feature}")
print(f"  Overall churn rate:       {churn_rate:.1%}")
print(f"\n  Business interpretation:")
print(f"  -> {churn_rate*100:.1f}% of customers are at risk of churning")
print(f"  -> The model can identify them before they leave")
print(f"  -> Marketing team can target At Risk customers with re-engagement campaigns")
print("\n  Next: Run 05_powerbi_export.py")
