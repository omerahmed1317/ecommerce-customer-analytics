"""
================================================================
STEP 3: RFM CUSTOMER SEGMENTATION
================================================================
E-Commerce Customer Analytics Project
File: 03_rfm_segmentation.py

WHAT IS RFM?
  R = Recency    — How recently did the customer buy?
  F = Frequency  — How many times did they buy?
  M = Monetary   — How much did they spend total?

WHY RFM?
  It's the most widely used customer segmentation technique in
  retail analytics. Used by Amazon, Shopify, Salesforce, and
  virtually every e-commerce company on earth.

SEGMENTS CREATED:
  Champions      -> Bought recently, buy often, spend the most
  Loyal          -> Buy often and respond well to promotions
  Potential       -> Recent customers who haven't bought often
  At Risk        -> Above average customers who haven't bought recently
  Cannot Lose    -> Made big purchases but haven't returned
  Lost           -> Lowest recency, frequency and monetary scores

OUTPUT:
  data/rfm_segments.csv
  outputs/02_rfm_segments_treemap.png
  outputs/03_customer_segments.png
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import os

os.makedirs("outputs", exist_ok=True)

# ── PLOT STYLE ────────────────────────────────────────────────────
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
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
    "figure.dpi":        120,
})

ACCENT   = "#00B4D8"
ACCENT2  = "#48CAE4"
ORANGE   = "#FF4D00"
GREEN    = "#3FB950"
YELLOW   = "#FFB300"
RED      = "#F85149"
PURPLE   = "#A371F7"

print("=" * 60)
print("  E-COMMERCE ANALYTICS — RFM SEGMENTATION")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────────────
df = pd.read_csv("data/cleaned_transactions.csv", parse_dates=["InvoiceDate"])
print(f"\n[LOAD] {len(df):,} transactions | {df['Customer ID'].nunique():,} customers")

# ── CALCULATE REFERENCE DATE ──────────────────────────────────────
# Use 1 day after the last transaction as the "today" reference
# WHY: Recency is measured as days since last purchase relative to now
REFERENCE_DATE = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"  Reference date (analysis 'today'): {REFERENCE_DATE.date()}")

# ── COMPUTE RFM METRICS PER CUSTOMER ─────────────────────────────
print("\n[RFM] Computing Recency, Frequency, Monetary per customer...")

rfm = df.groupby("Customer ID").agg(
    last_purchase  = ("InvoiceDate", "max"),
    frequency      = ("Invoice", "nunique"),      # unique orders, not rows
    monetary       = ("TotalRevenue", "sum"),
).reset_index()

# Recency = days since last purchase (lower = better = more recent)
rfm["recency"] = (REFERENCE_DATE - rfm["last_purchase"]).dt.days

print(f"  Customers analyzed: {len(rfm):,}")
print(f"\n  RFM Summary Statistics:")
print(f"  Recency  — min: {rfm['recency'].min()} days | "
      f"median: {rfm['recency'].median():.0f} | max: {rfm['recency'].max()}")
print(f"  Frequency — min: {rfm['frequency'].min()} orders | "
      f"median: {rfm['frequency'].median():.0f} | max: {rfm['frequency'].max()}")
print(f"  Monetary — min: GBP{rfm['monetary'].min():.2f} | "
      f"median: GBP{rfm['monetary'].median():.2f} | max: GBP{rfm['monetary'].max():.2f}")

# ── SCORE EACH METRIC 1-5 ─────────────────────────────────────────
print("\n[SCORE] Assigning 1-5 scores to each RFM metric...")

# WHY pd.qcut: divides into equal-sized quantile buckets
# Recency: REVERSE scoring — recent buyers (low days) get HIGH score (5)
# Frequency and Monetary: higher = better = higher score

try:
    rfm["R_score"] = pd.qcut(rfm["recency"],  q=5,
                              labels=[5,4,3,2,1], duplicates="drop").astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5,
                              labels=[1,2,3,4,5], duplicates="drop").astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  q=5,
                              labels=[1,2,3,4,5], duplicates="drop").astype(int)
except Exception:
    rfm["R_score"] = pd.cut(rfm["recency"],  bins=5,
                             labels=[5,4,3,2,1]).astype(int)
    rfm["F_score"] = pd.cut(rfm["frequency"], bins=5,
                             labels=[1,2,3,4,5]).astype(int)
    rfm["M_score"] = pd.cut(rfm["monetary"],  bins=5,
                             labels=[1,2,3,4,5]).astype(int)

rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
rfm["RFM_Segment_Code"] = (rfm["R_score"].astype(str)
                           + rfm["F_score"].astype(str)
                           + rfm["M_score"].astype(str))

print(f"  ✅ Scores assigned. RFM Score range: "
      f"{rfm['RFM_Score'].min()} – {rfm['RFM_Score'].max()}")

# ── ASSIGN SEGMENT LABELS ─────────────────────────────────────────
print("\n[SEGMENT] Assigning customer segment labels...")

def assign_segment(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    score = row["RFM_Score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal"
    elif r >= 4 and f <= 2:
        return "Potential Loyalists"
    elif r >= 3 and f <= 2 and m >= 3:
        return "Promising"
    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    elif r <= 2 and f >= 4 and m >= 4:
        return "Cannot Lose Them"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    elif score >= 9:
        return "Loyal"
    elif score >= 6:
        return "Potential Loyalists"
    else:
        return "Lost"

rfm["Segment"] = rfm.apply(assign_segment, axis=1)

seg_counts = rfm["Segment"].value_counts()
print("\n  Customer Segments:")
for seg, count in seg_counts.items():
    pct = count / len(rfm) * 100
    print(f"    {seg:<25} {count:>5,} customers ({pct:.1f}%)")

# ── SEGMENT REVENUE ANALYSIS ──────────────────────────────────────
total_revenue = rfm["monetary"].sum()
seg_revenue = rfm.groupby("Segment")["monetary"].sum().sort_values(ascending=False)
print(f"\n  Revenue by Segment (total: GBP{total_revenue:,.2f}):")
for seg, rev in seg_revenue.items():
    pct = rev / total_revenue * 100
    print(f"    {seg:<25} GBP{rev:>10,.2f}  ({pct:.1f}%)")

# ── CHART 1: CUSTOMER SEGMENT DISTRIBUTION ────────────────────────
print("\n[CHART 1] Customer segment distribution...")

SEGMENT_COLORS = {
    "Champions":          "#FFB300",
    "Loyal":              "#3FB950",
    "Potential Loyalists":"#00B4D8",
    "Promising":          "#A371F7",
    "At Risk":            "#FF4D00",
    "Cannot Lose Them":   "#F85149",
    "Lost":               "#484F58",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: bar chart of customer counts
segs  = seg_counts.index.tolist()
cnts  = seg_counts.values
cols  = [SEGMENT_COLORS.get(s, ACCENT) for s in segs]
bars  = ax1.barh(segs, cnts, color=cols, height=0.65, edgecolor="none")
for bar, val in zip(bars, cnts):
    ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", fontsize=9)
ax1.set_xlabel("Number of Customers")
ax1.set_title("Customers per Segment", fontweight="bold", fontsize=13)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right: revenue share donut
rev_data = seg_revenue.reindex(segs).fillna(0)
rev_cols  = [SEGMENT_COLORS.get(s, ACCENT) for s in rev_data.index]
wedges, texts, autotexts = ax2.pie(
    rev_data.values, labels=rev_data.index,
    colors=rev_cols, autopct="%1.1f%%", startangle=90,
    pctdistance=0.8,
    wedgeprops=dict(width=0.6, edgecolor="#0D1117", linewidth=2)
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight("bold")
    at.set_color("#0D1117")
ax2.set_title("Revenue Share by Segment", fontweight="bold", fontsize=13)

fig.suptitle("RFM Customer Segmentation — E-Commerce 2020-2021",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/03_customer_segments.png", bbox_inches="tight")
plt.close()
print("  ✅ outputs/03_customer_segments.png")

# ── CHART 2: RFM SCATTER ─────────────────────────────────────────
print("[CHART 2] RFM scatter plot...")

fig, ax = plt.subplots(figsize=(12, 7))
for seg, color in SEGMENT_COLORS.items():
    mask = rfm["Segment"] == seg
    ax.scatter(rfm.loc[mask, "recency"],
               rfm.loc[mask, "monetary"],
               c=color, alpha=0.6, s=rfm.loc[mask, "frequency"]*8,
               label=seg, edgecolors="none")

ax.set_xlabel("Recency (days since last purchase)", fontsize=11)
ax.set_ylabel("Monetary Value (GBP total spend)", fontsize=11)
ax.set_title("Customer RFM Map — Recency vs Monetary Value\n(bubble size = frequency)",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"GBP{int(x):,}"))
ax.legend(loc="upper right", fontsize=8, facecolor="#161B22",
          edgecolor="#30363D", markerscale=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/02_rfm_scatter.png", bbox_inches="tight")
plt.close()
print("  ✅ outputs/02_rfm_scatter.png")

# ── SAVE RFM DATA ─────────────────────────────────────────────────
rfm_out = rfm.drop(columns=["last_purchase"])
rfm_out.to_csv("data/rfm_segments.csv", index=False)
print(f"\n  ✅ data/rfm_segments.csv ({len(rfm_out):,} customers)")

# ── KEY INSIGHTS ──────────────────────────────────────────────────
champions     = rfm[rfm["Segment"] == "Champions"]
at_risk       = rfm[rfm["Segment"] == "At Risk"]
champ_revenue = champions["monetary"].sum()
champ_pct_rev = champ_revenue / total_revenue * 100
champ_pct_cus = len(champions) / len(rfm) * 100

print("\n" + "=" * 60)
print("  KEY RFM INSIGHTS")
print("=" * 60)
print(f"\n  Total customers analyzed:      {len(rfm):,}")
print(f"  Champions ({champ_pct_cus:.1f}% of customers)")
print(f"    -> Generate {champ_pct_rev:.1f}% of total revenue")
print(f"    -> Avg spend: GBP{champions['monetary'].mean():.2f}")
print(f"\n  At Risk customers: {len(at_risk):,}")
print(f"    -> Avg days since last purchase: {at_risk['recency'].mean():.0f}")
print(f"    -> Revenue at risk: GBP{at_risk['monetary'].sum():,.2f}")
print(f"\n  Lost customers: {(rfm['Segment']=='Lost').sum():,}")
print("\n  Next: Run 04_churn_prediction.py")
