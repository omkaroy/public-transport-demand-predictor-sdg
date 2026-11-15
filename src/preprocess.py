"""
======================================================================
 Public Transport Demand Predictor — Data Preprocessing Pipeline
======================================================================
 Steps:
   1. Load raw CSV
   2. Remove duplicates
   3. Handle missing values
   4. Outlier analysis and capping (IQR method)
   5. Date / time feature extraction
   6. Categorical encoding
   7. Feature selection
   8. Chronological train/test split (no data leakage)
   9. Save processed artefacts
======================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RAW_CSV   = os.path.join(DATA_DIR, "transport_demand_synthetic.csv")
PROC_CSV  = os.path.join(DATA_DIR, "processed_data.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — Load data
# ══════════════════════════════════════════════════════════════════════
def load_data(path: str = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV and parse the Date column."""
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"[Load]        Rows: {len(df):,}  |  Cols: {df.shape[1]}")
    return df


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — Remove duplicates
# ══════════════════════════════════════════════════════════════════════
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows.
    Key columns used for duplicate detection:
        Date, Route_ID, Hour  →  should be unique per record.
    """
    before = len(df)
    # Keep first occurrence of any (Date, Route_ID, Hour) duplicate
    df = df.drop_duplicates(subset=["Date", "Route_ID", "Hour"], keep="first")
    after  = len(df)
    print(f"[Duplicates]  Removed: {before - after:,}  |  Remaining: {after:,}")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — Handle missing values
# ══════════════════════════════════════════════════════════════════════
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Temperature: fill with route-hour median (most representative).
    - Weather    : fill with 'Clear' (mode / most common condition).
    - Any other  : drop rows with remaining nulls.
    """
    missing_before = df.isnull().sum().sum()

    # Temperature → group median
    df["Temperature"] = df.groupby(["Route_ID", "Hour"])["Temperature"] \
                          .transform(lambda s: s.fillna(s.median()))
    # Fallback if entire group is NaN
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    # Weather → mode fill
    df["Weather"] = df["Weather"].fillna("Clear")

    # Drop any remaining nulls (should be none)
    df.dropna(inplace=True)

    missing_after = df.isnull().sum().sum()
    print(f"[Missing]     Before: {missing_before}  |  After: {missing_after}")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — Outlier analysis and capping (IQR)
# ══════════════════════════════════════════════════════════════════════
def handle_outliers(df: pd.DataFrame,
                    col: str = "Passenger_Count",
                    factor: float = 3.0) -> pd.DataFrame:
    """
    Use the IQR (Inter-Quartile Range) method to identify and cap
    extreme outliers in the target column.

    Values beyond  Q1 − factor×IQR  or  Q3 + factor×IQR  are capped
    rather than dropped to preserve row count while limiting distortion.
    factor=3.0 targets only extreme anomalies (data-entry errors).
    """
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - factor * IQR
    hi  = Q3 + factor * IQR

    n_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col]    = df[col].clip(lower=lo, upper=hi)
    print(f"[Outliers]    Capped {n_outliers} rows in '{col}'  "
          f"[{lo:.0f}, {hi:.0f}]")
    return df


# ══════════════════════════════════════════════════════════════════════
#  STEP 5 — Date / time feature extraction
# ══════════════════════════════════════════════════════════════════════
def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional temporal features from the Date column to help
    the model capture seasonal and weekly patterns.
    """
    df["Month"]      = df["Date"].dt.month          # 1–12
    df["Quarter"]    = df["Date"].dt.quarter        # 1–4
    df["DayOfMonth"] = df["Date"].dt.day            # 1–31
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype("int64")

    # Cyclic encoding for Hour and Month (preserves circular nature)
    df["Hour_sin"]   = np.sin(2 * np.pi * df["Hour"]  / 24)
    df["Hour_cos"]   = np.cos(2 * np.pi * df["Hour"]  / 24)
    df["Month_sin"]  = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]  = np.cos(2 * np.pi * df["Month"] / 12)

    print("[Features]    Temporal features extracted.")
    return df


# ══════════════════════════════════════════════════════════════════════
#  STEP 6 — Categorical encoding
# ══════════════════════════════════════════════════════════════════════
def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode Route_ID, Day_of_Week, and Weather.
    Returns the modified DataFrame and a dict of fitted LabelEncoders
    (saved to disk so the dashboard can use them for user inputs).
    """
    encoders = {}
    for col in ["Route_ID", "Day_of_Week", "Weather"]:
        le          = LabelEncoder()
        df[col]     = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[Encode]      '{col}'  classes: {list(le.classes_)}")

    # Persist encoders
    enc_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    joblib.dump(encoders, enc_path)
    print(f"[Encode]      Saved encoders → {enc_path}")
    return df, encoders


# ══════════════════════════════════════════════════════════════════════
#  STEP 7 — Feature selection
# ══════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "Route_ID",      # encoded route identifier
    "Hour",          # 0–23
    "Hour_sin",      # cyclic hour feature
    "Hour_cos",      # cyclic hour feature
    "Day_of_Week",   # encoded weekday
    "Is_Weekend",    # binary flag
    "Temperature",   # °C
    "Weather",       # encoded weather condition
    "Is_Holiday",    # binary flag
    "Month",         # 1–12
    "Month_sin",     # cyclic month feature
    "Month_cos",     # cyclic month feature
    "Quarter",       # 1–4
    "WeekOfYear",    # 1–53
]
TARGET_COL = "Passenger_Count"


# ══════════════════════════════════════════════════════════════════════
#  STEP 8 — Chronological train / test split
# ══════════════════════════════════════════════════════════════════════
def chronological_split(df: pd.DataFrame,
                         test_ratio: float = 0.20
                         ) -> tuple[pd.DataFrame, pd.DataFrame,
                                    pd.DataFrame, pd.DataFrame]:
    """
    Sort by Date then split: first 80 % → train, last 20 % → test.
    This prevents leakage of future data into training — critical for
    time-series-style problems.
    """
    df_sorted  = df.sort_values("Date").reset_index(drop=True)
    split_idx  = int(len(df_sorted) * (1 - test_ratio))

    train = df_sorted.iloc[:split_idx]
    test  = df_sorted.iloc[split_idx:]

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    print(f"[Split]       Train: {len(train):,}  |  Test: {len(test):,}")
    print(f"[Split]       Train dates: {train['Date'].min().date()} → "
          f"{train['Date'].max().date()}")
    print(f"[Split]       Test  dates: {test['Date'].min().date()} → "
          f"{test['Date'].max().date()}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════
def run_pipeline():
    """Execute the full preprocessing pipeline and return split data."""
    print("\n" + "=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data()
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = extract_time_features(df)
    df, encoders = encode_categoricals(df)

    # Save processed data
    df.to_csv(PROC_CSV, index=False)
    print(f"[Save]        Processed CSV → {PROC_CSV}")

    X_train, X_test, y_train, y_test = chronological_split(df)

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60 + "\n")
    return X_train, X_test, y_train, y_test, encoders


if __name__ == "__main__":
    run_pipeline()
