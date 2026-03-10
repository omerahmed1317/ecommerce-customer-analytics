"""
================================================================
STEP 5: POWER BI EXPORT
================================================================
E-Commerce Customer Analytics Project
File: 05_powerbi_export.py

PURPOSE:
  Exports 3 clean, pre-formatted CSV files that are ready to
  open directly in Power BI Desktop. Also generates all static
  charts used in the README and portfolio.

WHY EXPORT FOR POWER BI?
  Power BI is the most widely used BI tool in enterprise.
  Showing that your Python pipeline feeds into Power BI
  demonstrates you understand the full analytics stack.

EXPORTS:
  data/powerbi_transactions.csv   ← fact table
  data/powerbi_customers.csv      ← customer dimension
  data/powerbi_products.csv       ← product dimension

CHARTS:
  outputs/01_revenue_by_country.png
  outputs/07_monthly_revenue.png
  outputs/08_top_products.png
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs("outputs", exist_ok=True)

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
ORANGE  = "#FF4D00"
GREEN   = "#3FB950"
YELLOW  = "#FFB300"

print("=" * 60)
print("  E-COMMERCE ANALYTICS — POWER BI EXPORT")
print("=" * 60)

# ── LOAD DATA ────────────────────────────────────────────────────
df  = pd.read_csv("data/cleaned_transactions.csv", parse_dates=["InvoiceDate"])
rfm = pd.read_csv("data/rfm_segments.csv")
churn = pd.read_csv("data/churn_labels.csv")

print(f"\n[LOAD] {len(df):,} transactions loaded")
df["YearMonth"] = df["InvoiceDate"].dt.strftime("%Y-%m")

# ── EXPORT 1: TRANSACTIONS (FACT TABLE) ─────────────────────────
print("\n[EXPORT 1] Transactions fact table...")

pbi_transactions = df[[
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "TotalRevenue",
    "Customer ID", "Country", "Year", "Month", "DayOfWeek", "Hour"
]].copy()

# Add formatted date columns Power BI needs
pbi_transactions["DateKey"]       = pbi_transactions["InvoiceDate"].dt.strftime("%Y%m%d").astype(int)
pbi_transactions["MonthName"]     = pbi_transactions["InvoiceDate"].dt.strftime("%B")
pbi_transactions["YearMonth"]     = pbi_transactions["InvoiceDate"].dt.strftime("%Y-%m")
pbi_transactions["Quarter"]       = "Q" + pbi_transactions["InvoiceDate"].dt.quarter.astype(str)
pbi_transactions["IsWeekend"]     = pbi_transactions["DayOfWeek"].isin(["Saturday","Sunday"]).astype(int)

pbi_transactions.to_csv("data/powerbi_transactions.csv", index=False)
print(f"  ✅ data/powerbi_transactions.csv ({len(pbi_transactions):,} rows, {len(pbi_transactions.columns)} columns)")

# ── EXPORT 2: CUSTOMER DIMENSION ────────────────────────────────
print("\n[EXPORT 2] Customer dimension table...")

# Merge RFM + churn + transaction aggregates
customer_agg = df.groupby("Customer ID").agg(
    first_purchase = ("InvoiceDate", "min"),
    last_purchase  = ("InvoiceDate", "max"),
    total_orders   = ("Invoice", "nunique"),
    total_revenue  = ("TotalRevenue", "sum"),
    avg_basket     = ("TotalRevenue", "mean"),
    favourite_country = ("Country", lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
).reset_index()

pbi_customers = customer_agg.merge(
    rfm[["Customer ID", "R_score","F_score","M_score","RFM_Score","Segment","recency","frequency","monetary"]],
    on="Customer ID", how="left"
).merge(
    churn[["Customer ID","is_churned"]],
    on="Customer ID", how="left"
)

pbi_customers["customer_lifetime_days"] = (
    (pbi_customers["last_purchase"] - pbi_customers["first_purchase"])
    .dt.days
)
pbi_customers["is_churned"] = pbi_customers["is_churned"].fillna(0).astype(int)
pbi_customers["churn_label"] = pbi_customers["is_churned"].map({0: "Active", 1: "Churned"})

pbi_customers.to_csv("data/powerbi_customers.csv", index=False)
print(f"  ✅ data/powerbi_customers.csv ({len(pbi_customers):,} customers)")

# ── EXPORT 3: PRODUCT DIMENSION ─────────────────────────────────
print("\n[EXPORT 3] Product dimension table...")

pbi_products = df.groupby(["StockCode","Description"]).agg(
    times_sold      = ("Invoice", "nunique"),
    total_qty_sold  = ("Quantity", "sum"),
    total_revenue   = ("TotalRevenue", "sum"),
    avg_unit_price  = ("Price", "mean"),
    num_customers   = ("Customer ID", "nunique"),
).reset_index().sort_values("total_revenue", ascending=False)

pbi_products["revenue_rank"] = range(1, len(pbi_products) + 1)

pbi_products.to_csv("data/powerbi_products.csv", index=False)
print(f"  ✅ data/powerbi_products.csv ({len(pbi_products):,} products)")

# ── CHART 1: REVENUE BY COUNTRY ─────────────────────────────────
print("\n[CHART 1] Revenue by country...")

country_rev = (df.groupby("Country")["TotalRevenue"]
               .sum().sort_values(ascending=False).head(10))

fig, ax = plt.subplots(figsize=(12, 6))
colors = [ORANGE if i == 0 else ACCENT for i in range(len(country_rev))]
bars = ax.bar(country_rev.index, country_rev.values,
              color=colors, edgecolor="none", width=0.65)
for bar, val in zip(bars, country_rev.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"GBP{val/1000:.0f}k", ha="center", fontsize=9, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"GBP{int(x/1000)}k"))
ax.set_xlabel("Country")
ax.set_ylabel("Total Revenue")
ax.set_title("Revenue by Country — Top 10 Markets",
             fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=30, ha="right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/01_revenue_by_country.png", bbox_inches="tight")
plt.close()
print("  ✅ outputs/01_revenue_by_country.png")

# ── CHART 2: MONTHLY REVENUE TREND ──────────────────────────────
print("[CHART 2] Monthly revenue trend...")

monthly = (df.groupby("YearMonth")["TotalRevenue"]
             .sum().reset_index().sort_values("YearMonth"))

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(range(len(monthly)), monthly["TotalRevenue"],
                alpha=0.25, color=ACCENT)
ax.plot(range(len(monthly)), monthly["TotalRevenue"],
        color=ACCENT, linewidth=2.5, marker="o", markersize=5)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["YearMonth"], rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"GBP{int(x/1000)}k"))
ax.set_ylabel("Monthly Revenue")
ax.set_title("Monthly Revenue Trend 2020-2021",
             fontsize=13, fontweight="bold", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/07_monthly_revenue.png", bbox_inches="tight")
plt.close()
print("  ✅ outputs/07_monthly_revenue.png")

# ── CHART 3: TOP PRODUCTS ────────────────────────────────────────
print("[CHART 3] Top products by revenue...")

top_prod = pbi_products.head(10)
fig, ax = plt.subplots(figsize=(12, 6))
colors = [ORANGE if i == 0 else ACCENT for i in range(len(top_prod))]
bars = ax.barh(
    [d[:40] for d in top_prod["Description"]],
    top_prod["total_revenue"], color=colors, height=0.65, edgecolor="none"
)
for bar, val in zip(bars, top_prod["total_revenue"]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f"GBP{val:,.0f}", va="center", fontsize=8)
ax.set_xlabel("Total Revenue (GBP)")
ax.set_title("Top 10 Products by Revenue",
             fontsize=13, fontweight="bold", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/08_top_products.png", bbox_inches="tight")
plt.close()
print("  ✅ outputs/08_top_products.png")

# ── SUMMARY ──────────────────────────────────────────────────────
total_rev = df["TotalRevenue"].sum()
top_country = country_rev.index[0]
top_country_pct = country_rev.iloc[0] / total_rev * 100

print(f"\n{'='*60}")
print("  POWER BI EXPORT SUMMARY")
print(f"{'='*60}")
print(f"\n  3 CSV files exported and ready for Power BI Desktop:")
print(f"  1. powerbi_transactions.csv — {len(pbi_transactions):,} rows")
print(f"  2. powerbi_customers.csv    — {len(pbi_customers):,} customers")
print(f"  3. powerbi_products.csv     — {len(pbi_products):,} products")
print(f"\n  Total revenue analyzed:     GBP{total_rev:,.2f}")
print(f"  Top market:                 {top_country} ({top_country_pct:.1f}% of revenue)")
print(f"  Top product:                {pbi_products.iloc[0]['Description'][:50]}")
print(f"\n  HOW TO USE IN POWER BI:")
print(f"  1. Open Power BI Desktop")
print(f"  2. Get Data -> Text/CSV -> select all 3 CSV files")
print(f"  3. Create relationships: Customer ID (transactions ↔ customers)")
print(f"  4. Build visuals: revenue cards, segment donut, churn bar")
print(f"\n  Next: Run 06_streamlit_app.py")
