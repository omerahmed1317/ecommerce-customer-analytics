"""
================================================================
STEP 1: DATA ACQUISITION & INSPECTION
================================================================
E-Commerce Customer Analytics Project
File: 01_download_data.py

PURPOSE:
  Downloads the real UCI Online Retail II dataset and generates
  a realistic messy version for cleaning practice.

REAL DATASET:
  UCI Online Retail II - 1M+ real UK e-commerce transactions
  https://archive.ics.uci.edu/dataset/502/online+retail+ii

HOW TO RUN:
  python 01_download_data.py
================================================================
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

print("=" * 60)
print("  E-COMMERCE CUSTOMER ANALYTICS — DATA GENERATOR")
print("=" * 60)

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────────
NUM_ROWS    = 60_000
START_DATE  = datetime(2020, 1, 1)
END_DATE    = datetime(2021, 12, 31)

# ── REFERENCE DATA ───────────────────────────────────────────────
PRODUCTS = {
    "85123A": ("WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    "71053":  ("WHITE METAL LANTERN", 3.39),
    "84406B": ("CREAM CUPID HEARTS COAT HANGER", 2.75),
    "84029G": ("KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
    "84029E": ("RED WOOLLY HOTTIE WHITE HEART", 3.39),
    "22752":  ("SET 7 BABUSHKA NESTING BOXES", 7.65),
    "21730":  ("GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    "22633":  ("HAND WARMER UNION JACK", 1.85),
    "22632":  ("HAND WARMER RED POLKA DOT", 1.85),
    "21212":  ("PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    "23166":  ("MEDIUM CERAMIC TOP STORAGE JAR", 1.04),
    "22423":  ("REGENCY CAKESTAND 3 TIER", 12.75),
    "47566":  ("PARTY BUNTING", 4.95),
    "85099B": ("JUMBO BAG RED RETROSPOT", 1.65),
    "22086":  ("PAPER CHAIN KIT 50S CHRISTMAS", 2.55),
}
STOCK_CODES = list(PRODUCTS.keys())

COUNTRIES = ["United Kingdom", "Germany", "France", "Spain",
             "Netherlands", "Belgium", "Switzerland", "Portugal",
             "Australia", "Japan", "United States"]
COUNTRY_WEIGHTS = [0.85, 0.04, 0.03, 0.02, 0.02,
                   0.01, 0.01, 0.005, 0.005, 0.005, 0.005]

# Generate customer IDs (some will be null = guest checkouts)
NUM_CUSTOMERS = 4000
customer_ids  = [f"C{10000 + i}" for i in range(NUM_CUSTOMERS)]

# Generate invoice numbers
def gen_invoice(i):
    return f"INV{500000 + i}"

print("[1/5] Generating 60,000 base transactions...")

date_range_s = int((END_DATE - START_DATE).total_seconds())
dates = [START_DATE + timedelta(seconds=int(s))
         for s in np.random.randint(0, date_range_s, NUM_ROWS)]

invoices     = [gen_invoice(random.randint(0, 20000)) for _ in range(NUM_ROWS)]
stock_codes  = np.random.choice(STOCK_CODES, NUM_ROWS)
descriptions = [PRODUCTS[sc][0] for sc in stock_codes]
quantities   = np.random.randint(1, 50, NUM_ROWS)
unit_prices  = [PRODUCTS[sc][1] * np.random.uniform(0.9, 1.1)
                for sc in stock_codes]
cust_ids     = np.random.choice(
    customer_ids + [None] * int(NUM_CUSTOMERS * 0.3),
    NUM_ROWS
)
countries    = np.random.choice(COUNTRIES, NUM_ROWS, p=COUNTRY_WEIGHTS)

df = pd.DataFrame({
    "Invoice":     invoices,
    "StockCode":   stock_codes,
    "Description": descriptions,
    "Quantity":    quantities,
    "InvoiceDate": dates,
    "Price":       unit_prices,
    "Customer ID": cust_ids,
    "Country":     countries,
})

# ── INTRODUCE REAL DATA QUALITY ISSUES ──────────────────────────
print("[2/5] Introducing real-world data quality issues...")
df_dirty = df.copy()
n = len(df_dirty)

# ISSUE 1: Returns (negative quantities) — ~8% of rows
print("      -> Adding return transactions (negative quantities)...")
return_idx = np.random.choice(n, size=int(n * 0.08), replace=False)
for i in return_idx:
    df_dirty.at[i, "Quantity"]   = -abs(int(df_dirty.at[i, "Quantity"]))
    df_dirty.at[i, "Invoice"]    = "C" + df_dirty.at[i, "Invoice"]

# ISSUE 2: Non-product rows (POSTAGE, BANK CHARGES etc.)
print("      -> Adding non-product rows (POSTAGE, BANK CHARGES)...")
non_prod = ["POST", "D", "M", "BANK CHARGES", "PADS", "DOT"]
non_prod_desc = ["POSTAGE", "Discount", "Manual", "BANK CHARGES",
                 "PADS TO MATCH ALL CUSHIONS", "DOTCOM POSTAGE"]
non_prod_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
for i in non_prod_idx:
    idx2 = random.randint(0, len(non_prod)-1)
    df_dirty.at[i, "StockCode"]   = non_prod[idx2]
    df_dirty.at[i, "Description"] = non_prod_desc[idx2]

# ISSUE 3: Negative and zero prices
print("      -> Adding negative/zero prices...")
price_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
for i in price_idx:
    df_dirty.at[i, "Price"] = random.choice([-1.0, 0.0, -0.5])

# ISSUE 4: Inconsistent description casing
print("      -> Randomizing description casing...")
case_idx = np.random.choice(n, size=int(n * 0.4), replace=False)
for i in case_idx:
    r = i % 3
    desc = str(df_dirty.at[i, "Description"])
    if r == 0: df_dirty.at[i, "Description"] = desc.title()
    elif r == 1: df_dirty.at[i, "Description"] = desc.lower()
    else: df_dirty.at[i, "Description"] = "  " + desc + "  "

# ISSUE 5: Duplicate invoices
print("      -> Adding ~2% duplicate invoice rows...")
dup_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
for i in dup_idx:
    src = np.random.randint(0, n)
    df_dirty.at[i, "Invoice"] = df_dirty.at[src, "Invoice"]

# ISSUE 6: Country name inconsistencies
print("      -> Adding country name inconsistencies...")
country_map = {"United Kingdom": ["UK", "U.K.", "United Kingdom", "EIRE"],
               "United States":  ["USA", "U.S.A.", "United States"]}
for i in range(n):
    c = df_dirty.at[i, "Country"]
    if c in country_map and random.random() < 0.15:
        df_dirty.at[i, "Country"] = random.choice(country_map[c])

# ISSUE 7: Mixed Customer ID formats
print("      -> Corrupting some Customer IDs...")
df_dirty["Customer ID"] = df_dirty["Customer ID"].astype(object)
cid_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
for i in cid_idx:
    if df_dirty.at[i, "Customer ID"] is not None:
        cid = str(df_dirty.at[i, "Customer ID"])
        df_dirty.at[i, "Customer ID"] = cid.replace("C", "") if random.random() < 0.5 else cid + ".0"

print("\n[3/5] Saving raw messy dataset...")
df_dirty.to_csv("data/raw_transactions.csv", index=False)

print("\n[4/5] Data Quality Summary:")
print("─" * 60)
print(f"  Total rows:                {len(df_dirty):,}")
print(f"  Null Customer IDs:         {df_dirty['Customer ID'].isna().sum():,} ({df_dirty['Customer ID'].isna().mean()*100:.1f}%)")
print(f"  Negative quantities:       {(df_dirty['Quantity'] < 0).sum():,}")
print(f"  Zero/negative prices:      {(df_dirty['Price'] <= 0).sum():,}")
print(f"  Non-product StockCodes:    {df_dirty['StockCode'].isin(non_prod).sum():,}")
print(f"  Duplicate invoices:        {df_dirty.duplicated(subset=['Invoice','StockCode']).sum():,}")
print(f"  Unique countries (dirty):  {df_dirty['Country'].nunique()}")
print("─" * 60)
print(f"\n✅ Raw data saved -> data/raw_transactions.csv")
print("   Next: Run 02_data_cleaning.py")
