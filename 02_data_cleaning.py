"""
================================================================
STEP 2: DATA CLEANING
================================================================
E-Commerce Customer Analytics Project
File: 02_data_cleaning.py

WHAT YOU FIX:
  1. Null Customer IDs (guest checkouts) -> removed
  2. Returns (negative quantities)       -> separated into returns_df
  3. Non-product rows (POST, D, M...)    -> removed
  4. Negative / zero prices              -> removed
  5. Inconsistent description casing     -> standardized Title Case
  6. Duplicate invoice+stockcode rows    -> deduplicated
  7. Inconsistent country names          -> standardized

OUTPUT:
  data/cleaned_transactions.csv
  data/returns.csv
  data/removed_records.csv
  data/cleaning_report.txt
================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=" * 60)
print("  E-COMMERCE ANALYTICS — DATA CLEANING")
print("=" * 60)

df = pd.read_csv("data/raw_transactions.csv", low_memory=False)
original_shape = df.shape
removed_records = []

print(f"\n[LOAD] {len(df):,} rows × {len(df.columns)} columns loaded")
print(f"  Columns: {list(df.columns)}")

# ── FIX 1: Parse dates ───────────────────────────────────────────
print("\n[FIX 1] Parsing InvoiceDate...")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
bad_dates = df["InvoiceDate"].isna().sum()
if bad_dates > 0:
    bad = df[df["InvoiceDate"].isna()].copy()
    bad["removal_reason"] = "unparseable_date"
    removed_records.append(bad)
    df = df[df["InvoiceDate"].notna()].copy()
print(f"  ✅ Dates parsed. Removed {bad_dates} unparseable rows.")
print(f"  Date range: {df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}")

# ── FIX 2: Separate returns (negative quantity) ──────────────────
print("\n[FIX 2] Separating return transactions...")
# WHY: Returns have negative quantities and Invoice starting with 'C'
# They are valid data but analyzed separately — don't mix with sales
returns_df = df[df["Quantity"] < 0].copy()
returns_df["removal_reason"] = "return_transaction"
removed_records.append(returns_df)
df = df[df["Quantity"] > 0].copy()
print(f"  Return transactions found:   {len(returns_df):,}")
print(f"  Return rate: {len(returns_df)/original_shape[0]*100:.1f}% of all transactions")
print(f"  ✅ Returns saved to data/returns.csv for separate analysis")

# ── FIX 3: Remove non-product rows ──────────────────────────────
print("\n[FIX 3] Removing non-product StockCodes...")
# These are internal codes: POST=postage, D=discount, M=manual, etc.
# They inflate revenue figures and distort customer analysis
NON_PRODUCT_CODES = ["POST", "D", "M", "BANK CHARGES", "PADS",
                     "DOT", "CRUK", "S", "AMAZONFEE", "B"]
non_prod_mask = df["StockCode"].astype(str).str.strip().isin(NON_PRODUCT_CODES)
non_prod_rows = df[non_prod_mask].copy()
non_prod_rows["removal_reason"] = "non_product_stockcode"
removed_records.append(non_prod_rows)
df = df[~non_prod_mask].copy()
print(f"  Non-product rows removed: {len(non_prod_rows):,}")
print(f"  ✅ Only real product transactions remain")

# ── FIX 4: Remove null Customer IDs ─────────────────────────────
print("\n[FIX 4] Removing null Customer IDs...")
# WHY: Customer analysis requires knowing WHO bought.
# Null = guest checkout. We cannot build RFM or churn without customer ID.
# Document the % so we can mention it in analysis notes.
null_cid = df["Customer ID"].isna()
null_cid_rows = df[null_cid].copy()
null_cid_rows["removal_reason"] = "null_customer_id"
removed_records.append(null_cid_rows)
df = df[~null_cid].copy()
pct_null = len(null_cid_rows) / original_shape[0] * 100
print(f"  Null Customer ID rows removed: {len(null_cid_rows):,} ({pct_null:.1f}%)")
print(f"  ✅ All remaining rows have Customer IDs")

# ── FIX 5: Standardize Customer ID format ───────────────────────
print("\n[FIX 5] Standardizing Customer ID format...")
# Some IDs lost their 'C' prefix or gained '.0' suffix
df["Customer ID"] = df["Customer ID"].astype(str).str.strip()
df["Customer ID"] = df["Customer ID"].str.replace(r"\.0$", "", regex=True)
# Ensure all start with C
def fix_cid(cid):
    cid = str(cid).strip()
    if cid.startswith("C"):
        return cid
    else:
        return "C" + cid
df["Customer ID"] = df["Customer ID"].apply(fix_cid)
print(f"  ✅ Customer IDs standardized. Sample: {df['Customer ID'].head(3).tolist()}")

# ── FIX 6: Remove negative/zero prices ──────────────────────────
print("\n[FIX 6] Removing invalid prices...")
bad_price = df["Price"] <= 0
bad_price_rows = df[bad_price].copy()
bad_price_rows["removal_reason"] = "invalid_price"
removed_records.append(bad_price_rows)
df = df[~bad_price].copy()
print(f"  Rows with invalid price removed: {len(bad_price_rows):,}")
print(f"  ✅ All prices are positive")

# ── FIX 7: Standardize Description ──────────────────────────────
print("\n[FIX 7] Standardizing Description column...")
before_unique = df["Description"].nunique()
df["Description"] = df["Description"].astype(str).str.strip().str.upper()
after_unique = df["Description"].nunique()
print(f"  Unique descriptions: {before_unique:,} -> {after_unique:,}")
print(f"  ✅ All descriptions are uppercase and stripped")

# ── FIX 8: Standardize Country names ────────────────────────────
print("\n[FIX 8] Standardizing Country names...")
country_fixes = {
    "UK":             "United Kingdom",
    "U.K.":           "United Kingdom",
    "EIRE":           "Ireland",
    "Eire":           "Ireland",
    "USA":            "United States",
    "U.S.A.":         "United States",
    "Unspecified":    "Unknown",
    "unspecified":    "Unknown",
}
before_countries = df["Country"].nunique()
df["Country"] = df["Country"].replace(country_fixes)
after_countries = df["Country"].nunique()
print(f"  Unique countries: {before_countries} -> {after_countries}")
print(f"  ✅ Country names standardized")

# ── FIX 9: Remove duplicates ─────────────────────────────────────
print("\n[FIX 9] Removing duplicate rows...")
dups = df.duplicated(subset=["Invoice", "StockCode", "Customer ID"], keep="first")
dup_rows = df[dups].copy()
dup_rows["removal_reason"] = "duplicate_invoice_stockcode"
removed_records.append(dup_rows)
df = df[~dups].copy()
print(f"  Duplicates removed: {len(dup_rows):,}")
print(f"  ✅ No more duplicate invoice+product combinations")

# ── FEATURE ENGINEERING ──────────────────────────────────────────
print("\n[ENGINEER] Creating new columns...")
df["TotalRevenue"] = df["Quantity"] * df["Price"]
df["Year"]         = df["InvoiceDate"].dt.year
df["Month"]        = df["InvoiceDate"].dt.month
df["DayOfWeek"]    = df["InvoiceDate"].dt.day_name()
df["Hour"]         = df["InvoiceDate"].dt.hour
print(f"  ✅ Added: TotalRevenue, Year, Month, DayOfWeek, Hour")

# ── VALIDATION ───────────────────────────────────────────────────
print("\n[VALIDATE] Final check...")
print(f"  Shape: {df.shape}")
print(f"  Null values in key columns:")
for col in ["Customer ID", "InvoiceDate", "Quantity", "Price", "TotalRevenue"]:
    n_null = df[col].isna().sum()
    print(f"    {'✅' if n_null == 0 else '⚠️'} {col}: {n_null} nulls")
print(f"  Revenue range: GBP{df['TotalRevenue'].min():.2f} to GBP{df['TotalRevenue'].max():.2f}")
print(f"  Unique customers: {df['Customer ID'].nunique():,}")
print(f"  Unique products:  {df['StockCode'].nunique():,}")

# ── SAVE OUTPUTS ─────────────────────────────────────────────────
print("\n[SAVE] Writing files...")
df.to_csv("data/cleaned_transactions.csv", index=False)
print(f"  ✅ data/cleaned_transactions.csv  ({len(df):,} rows)")

returns_df.to_csv("data/returns.csv", index=False)
print(f"  ✅ data/returns.csv ({len(returns_df):,} return rows)")

if removed_records:
    removed_df = pd.concat(removed_records, ignore_index=True)
    removed_df.to_csv("data/removed_records.csv", index=False)
    print(f"  ✅ data/removed_records.csv ({len(removed_df):,} removed rows)")

rows_removed = original_shape[0] - len(df)
report = [
    "=" * 60,
    "  E-COMMERCE DATA — CLEANING REPORT",
    f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "=" * 60,
    f"\nBEFORE: {original_shape[0]:,} rows × {original_shape[1]} columns",
    f"AFTER:  {len(df):,} rows × {len(df.columns)} columns",
    f"REMOVED: {rows_removed:,} rows ({rows_removed/original_shape[0]*100:.1f}%)",
    "\nISSUES FIXED:",
    f"1. Unparseable dates:         removed",
    f"2. Return transactions:       {len(returns_df):,} rows -> saved to returns.csv",
    f"3. Non-product rows:          {len(non_prod_rows):,} rows removed",
    f"4. Null Customer IDs:         {len(null_cid_rows):,} rows removed",
    f"5. Invalid prices:            {len(bad_price_rows):,} rows removed",
    f"6. Description casing:        standardized to UPPERCASE",
    f"7. Country names:             standardized ({before_countries} -> {after_countries})",
    f"8. Duplicate rows:            {len(dup_rows):,} rows removed",
    f"\nFEATURES ADDED: TotalRevenue, Year, Month, DayOfWeek, Hour",
    f"\nFINAL: {df['Customer ID'].nunique():,} unique customers, {df['StockCode'].nunique():,} unique products",
]
with open("data/cleaning_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"  ✅ data/cleaning_report.txt")

print(f"\n{'='*60}")
print(f"  CLEANING COMPLETE: {original_shape[0]:,} -> {len(df):,} rows")
print(f"{'='*60}")
print("  Next: Run 03_rfm_segmentation.py")
