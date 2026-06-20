"""
================================================
 DecodeLabs Industrial Training | Batch 2026
 Data Science Project 1
 Advanced EDA & Feature Engineering
 Dataset : Titanic (loaded from public URL)
================================================

Requirements covered:
  ✅ Missing-value imputation  (Row-delete / Global Median / KNN)
  ✅ Outlier detection & neutralisation (IQR Winsorization)
  ✅ 3+ engineered features
  ✅ One-Hot Encoding (OHE) for categorical columns
  ✅ Multicollinearity / Collinearity eradication
  ✅ Pandera runtime schema contract
  ✅ Vectorised operations throughout (no Python for-loops on rows)

Run:
    pip install pandas numpy scikit-learn pandera seaborn matplotlib
    python ds_project1_eda_feature_engineering.py
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

print("=" * 60)
print("  DecodeLabs | DS Project 1 — Advanced EDA & Feature Eng.")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. LOAD DATASET  (Titanic from public URL)
# ─────────────────────────────────────────────
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("\n[1] Loading dataset …")
df = pd.read_csv(URL)
print(f"    Shape  : {df.shape}")
print(f"    Columns: {list(df.columns)}")


# ─────────────────────────────────────────────
# 2. INITIAL EDA
# ─────────────────────────────────────────────
print("\n[2] Initial EDA")
print("\n── Data Types ──")
print(df.dtypes)

print("\n── Missing Values (count & %) ──")
missing = pd.DataFrame({
    "count"  : df.isnull().sum(),
    "percent": (df.isnull().mean() * 100).round(2)
})
missing = missing[missing["count"] > 0].sort_values("percent", ascending=False)
print(missing)

print("\n── Descriptive Statistics ──")
print(df.describe())

# Correlation heatmap (numeric only, saved as PNG)
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix – Raw Data")
plt.tight_layout()
plt.savefig("correlation_raw.png", dpi=100)
plt.close()
print("    Saved: correlation_raw.png")


# ─────────────────────────────────────────────
# 3. PHASE 1 – SECURING INPUT FIDELITY
#    Missing-Value Imputation (Decision Matrix)
# ─────────────────────────────────────────────
print("\n[3] Phase 1 – Missing-Value Imputation")

df_clean = df.copy()

# ── Calculate missingness proportion per feature ──
miss_pct = df_clean.isnull().mean() * 100

for col in df_clean.columns:
    pct = miss_pct[col]
    if pct == 0:
        continue

    dtype = df_clean[col].dtype

    if pct < 5:
        # RULE: < 5% → drop rows (preserves data, prevents synthetic bias)
        df_clean.dropna(subset=[col], inplace=True)
        print(f"    {col:12s} | {pct:.1f}% missing → DROP ROWS")

    elif 5 <= pct <= 20:
        if dtype == "object":
            # Categorical: mode imputation
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"    {col:12s} | {pct:.1f}% missing → MODE  imputation ('{mode_val}')")
        else:
            # Skewed numeric: Global Median
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"    {col:12s} | {pct:.1f}% missing → MEDIAN imputation ({median_val})")

    else:
        # RULE: > 20% → KNN Imputation
        if dtype != "object":
            print(f"    {col:12s} | {pct:.1f}% missing → KNN imputation")
            knn_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
            imputer = KNNImputer(n_neighbors=5)
            df_clean[knn_cols] = imputer.fit_transform(df_clean[knn_cols])
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"    {col:12s} | {pct:.1f}% missing → MODE  imputation ('{mode_val}') [categorical]")

print(f"\n    Remaining NaN count: {df_clean.isnull().sum().sum()}")
print(f"    Shape after imputation: {df_clean.shape}")


# ─────────────────────────────────────────────
# 4. OUTLIER DETECTION & NEUTRALISATION (IQR)
#    Strategy: Winsorization via numpy.clip()
# ─────────────────────────────────────────────
print("\n[4] Outlier Neutralisation — IQR Winsorization")

numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
exclude = {"Survived", "Pclass", "SibSp", "Parch", "PassengerId"}
target_cols = [c for c in numeric_cols if c not in exclude]

for col in target_cols:
    Q1  = df_clean[col].quantile(0.25)
    Q3  = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    df_clean[col] = np.clip(df_clean[col], lower, upper)   # vectorised — no loops

    print(f"    {col:8s} | IQR={IQR:.2f} | bounds=[{lower:.2f}, {upper:.2f}] | capped={n_outliers}")

print(f"\n    Shape after Winsorization: {df_clean.shape}")


# ─────────────────────────────────────────────
# 5. PHASE 2 – VECTORISED COMPUTATION ENGINE
#    Feature Engineering (5 new features)
# ─────────────────────────────────────────────
print("\n[5] Phase 2 – Feature Engineering")

# Feature 1: Family Size
df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
print("    ✅ FamilySize   = SibSp + Parch + 1")

# Feature 2: IsAlone (binary flag)
df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)
print("    ✅ IsAlone      = 1 if FamilySize == 1 else 0")

# Feature 3: Age Bin
df_clean["AgeBin"] = pd.cut(
    df_clean["Age"],
    bins=[0, 12, 25, 60, 100],
    labels=["Child", "YoungAdult", "Adult", "Senior"]
)
print("    ✅ AgeBin       = cut(Age) → Child / YoungAdult / Adult / Senior")

# Feature 4: Fare per Person
df_clean["FarePerPerson"] = df_clean["Fare"] / df_clean["FamilySize"]
print("    ✅ FarePerPerson = Fare / FamilySize")

# Feature 5: Title (social status proxy from Name)
df_clean["Title"] = df_clean["Name"].str.extract(r",\s*([^.]+)\.")
rare_titles = df_clean["Title"].value_counts()
rare_titles = rare_titles[rare_titles < 10].index
df_clean["Title"] = df_clean["Title"].replace(rare_titles, "Rare")
print("    ✅ Title        = extracted from Name (Mr/Mrs/Miss/Master/Rare)")


# ─────────────────────────────────────────────
# 6. CATEGORICAL ENCODING — ONE-HOT ENCODING
# ─────────────────────────────────────────────
print("\n[6] One-Hot Encoding")

df_clean.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"],
              inplace=True, errors="ignore")

cat_cols = ["Sex", "Embarked", "AgeBin", "Title"]
df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

print(f"    Columns after OHE : {df_encoded.shape[1]}")
print(f"    Shape             : {df_encoded.shape}")


# ─────────────────────────────────────────────
# 7. COLLINEARITY ERADICATION ALGORITHM
#    Pearson threshold = 0.80
# ─────────────────────────────────────────────
print("\n[7] Collinearity Eradication (threshold = 0.80)")

TARGET = "Survived"

# Cast bool columns to int (from get_dummies)
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

corr_matrix = df_encoded.select_dtypes(include=np.number).corr().abs()
upper_tri   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

cols_to_drop = set()
for col in upper_tri.columns:
    correlated = upper_tri.index[upper_tri[col] > 0.80].tolist()
    for cf in correlated:
        corr_col = abs(corr_matrix.loc[col, TARGET]) if TARGET in corr_matrix.columns else 0
        corr_cf  = abs(corr_matrix.loc[cf,  TARGET]) if TARGET in corr_matrix.columns else 0
        drop_candidate = col if corr_col < corr_cf else cf
        cols_to_drop.add(drop_candidate)
        print(f"    Pair ({col} ↔ {cf}) corr={upper_tri.loc[cf, col]:.2f} → DROP '{drop_candidate}'")

if not cols_to_drop:
    print("    No collinear pairs found above 0.80 ✅")

df_final = df_encoded.drop(columns=list(cols_to_drop), errors="ignore")
print(f"\n    Final feature count: {df_final.shape[1]}")


# ─────────────────────────────────────────────
# 8. PHASE 3 – PANDERA RUNTIME SCHEMA CONTRACT
# ─────────────────────────────────────────────
print("\n[8] Phase 3 – Pandera Runtime Schema Validation")

schema_checks = {
    "Survived"     : Column(int,   [Check.isin([0, 1])]),
    "Age"          : Column(float, [Check.greater_than(0),
                                    Check.less_than_or_equal_to(120)]),
    "Fare"         : Column(float, [Check.greater_than_or_equal_to(0)]),
    "FamilySize"   : Column(int,   [Check.greater_than_or_equal_to(1)]),
    "IsAlone"      : Column(int,   [Check.isin([0, 1])]),
    "FarePerPerson": Column(float, [Check.greater_than_or_equal_to(0)]),
    "Pclass"       : Column(int,   [Check.isin([1, 2, 3])]),
}

active_checks = {k: v for k, v in schema_checks.items() if k in df_final.columns}
schema = DataFrameSchema(columns=active_checks, strict=False)

try:
    validated_df = schema.validate(df_final, lazy=True)
    print("    ✅ All schema checks passed — dataset is ML-ready!")
except pa.errors.SchemaErrors as err:
    print("    ⚠️  Schema violations detected:")
    print(err.failure_cases)
    validated_df = df_final


# ─────────────────────────────────────────────
# 9. SAVE OUTPUTS & FINAL SUMMARY
# ─────────────────────────────────────────────
df_final.to_csv("titanic_cleaned.csv", index=False)

print("\n" + "=" * 60)
print("  PIPELINE SUMMARY")
print("=" * 60)
print(f"  Raw dataset shape        : {df.shape}")
print(f"  After imputation         : {df_clean.shape}")
print(f"  After OHE + engineering  : {df_encoded.shape}")
print(f"  After collinearity drop  : {df_final.shape}")
print(f"\n  Features engineered (5):")
print(f"    1. FamilySize      — SibSp + Parch + 1")
print(f"    2. IsAlone         — binary isolation flag")
print(f"    3. AgeBin          — Child/YoungAdult/Adult/Senior")
print(f"    4. FarePerPerson   — normalised ticket cost")
print(f"    5. Title           — social status from Name")
print(f"\n  Imputation strategies:")
print(f"    • Row deletion   (< 5% missing)")
print(f"    • Global Median  (5–20%, skewed numeric)")
print(f"    • Mode           (5–20%, categorical)")
print(f"    • KNN            (> 20%, numeric)")
print(f"\n  Outlier strategy : IQR Winsorization (numpy.clip)")
print(f"  Encoding         : One-Hot (drop_first=True)")
print(f"  Collinearity     : Pearson threshold = 0.80")
print(f"\n  Saved: titanic_cleaned.csv  ← ML-ready dataset")
print(f"  Saved: correlation_raw.png  ← Heatmap")
print("=" * 60)
print("\n  Pipeline complete — ready for model training! 🚀")
