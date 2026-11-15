"""
======================================================================
 Public Transport Demand Predictor — Dataset Generator
======================================================================
 Purpose : Generate a realistic SYNTHETIC dataset for model training.
 Note    : This is NOT real passenger data. It is artificially created
           to simulate plausible public bus ridership patterns.
 SDG     : 11 – Sustainable Cities and Communities
======================================================================
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Configuration ────────────────────────────────────────────────────
ROUTES   = ["R01", "R02", "R03", "R04", "R05"]
WEATHERS = ["Clear", "Cloudy", "Rainy", "Foggy"]

# Base ridership per route (riders at a "normal" hour)
ROUTE_BASE = {"R01": 120, "R02": 95, "R03": 150, "R04": 80, "R05": 110}

# ── Indian public holidays (2023–2024) ───────────────────────────────
HOLIDAYS = {
    "2023-01-26", "2023-03-08", "2023-04-07", "2023-04-14",
    "2023-08-15", "2023-10-02", "2023-10-24", "2023-11-01",
    "2023-12-25", "2024-01-01", "2024-01-26", "2024-03-25",
    "2024-04-14", "2024-08-15", "2024-10-02", "2024-10-12",
    "2024-12-25",
}


def hour_multiplier(hour: int) -> float:
    """
    Returns a demand multiplier based on the hour of the day.
    Peaks at morning (8–9) and evening (17–19) rush hours.
    """
    peaks = {
        6: 0.60, 7: 0.85, 8: 1.40, 9: 1.30, 10: 0.90,
        11: 0.80, 12: 0.95, 13: 0.90, 14: 0.75, 15: 0.80,
        16: 1.00, 17: 1.35, 18: 1.50, 19: 1.25, 20: 1.00,
        21: 0.75, 22: 0.50, 23: 0.30,
    }
    return peaks.get(hour, 0.20)   # very low after midnight


def weather_multiplier(weather: str) -> float:
    """Ridership drops in bad weather — people avoid buses."""
    return {"Clear": 1.00, "Cloudy": 0.95, "Rainy": 0.80, "Foggy": 0.88}[weather]


def generate_dataset(start: str = "2023-01-01",
                     end: str   = "2024-12-31",
                     hours: list = None) -> pd.DataFrame:
    """
    Generate one record per (date × hour × route) combination.

    Parameters
    ----------
    start  : First date (inclusive), YYYY-MM-DD
    end    : Last  date (inclusive), YYYY-MM-DD
    hours  : List of operating hours; defaults to 06:00–23:00

    Returns
    -------
    pd.DataFrame with synthetic passenger counts
    """
    if hours is None:
        hours = list(range(6, 24))          # 06:00 – 23:00

    dates       = pd.date_range(start, end, freq="D")
    records     = []

    for date in dates:
        date_str    = date.strftime("%Y-%m-%d")
        day_of_week = date.day_name()          # "Monday", …
        is_weekend  = int(date.weekday() >= 5) # Sat=5, Sun=6
        is_holiday  = int(date_str in HOLIDAYS)

        # Seasonal temperature (India): hot summers, mild winters
        # Rough sinusoidal approximation
        day_of_year = date.day_of_year
        base_temp   = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temperature = round(float(base_temp + rng.normal(0, 2)), 1)
        temperature = float(np.clip(temperature, 10, 45))

        weather = rng.choice(WEATHERS, p=[0.50, 0.25, 0.15, 0.10])

        for hour in hours:
            for route in ROUTES:
                base          = ROUTE_BASE[route]
                h_mult        = hour_multiplier(hour)
                w_mult        = weather_multiplier(weather)
                weekend_adj   = 0.75 if is_weekend else 1.0
                holiday_adj   = 0.65 if is_holiday else 1.0

                # Noise: ±15 % natural variance
                noise         = rng.normal(1.0, 0.12)

                raw_count     = (base * h_mult * w_mult
                                 * weekend_adj * holiday_adj * noise)
                raw_count     = max(0, raw_count)          # no negatives

                passenger_count = int(round(raw_count))

                records.append({
                    "Date"           : date_str,
                    "Route_ID"       : route,
                    "Hour"           : hour,
                    "Day_of_Week"    : day_of_week,
                    "Is_Weekend"     : is_weekend,
                    "Temperature"    : temperature,
                    "Weather"        : weather,
                    "Is_Holiday"     : is_holiday,
                    "Passenger_Count": passenger_count,
                })

    df = pd.DataFrame(records)

    # ── Inject a small proportion of missing values (realistic) ──────
    for col in ["Temperature", "Weather"]:
        mask = rng.random(len(df)) < 0.005   # 0.5 % missing
        df.loc[mask, col] = np.nan

    # ── Inject a few duplicate rows ───────────────────────────────────
    dup_indices = rng.choice(len(df), size=30, replace=False)
    duplicates  = df.iloc[dup_indices].copy()
    df          = pd.concat([df, duplicates], ignore_index=True)

    # ── Inject rare outliers (data-entry style errors) ────────────────
    outlier_idx = rng.choice(len(df), size=15, replace=False)
    df.loc[outlier_idx, "Passenger_Count"] = rng.integers(500, 800, size=15)

    return df


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set stdout encoding to utf-8 to handle emoji on Windows terminals
    import sys, io
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")

    print(">> Generating synthetic dataset ...")
    df = generate_dataset()

    out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "transport_demand_synthetic.csv")
    out_path = os.path.normpath(out_path)
    df.to_csv(out_path, index=False)

    print(f"[OK] Dataset saved  -> {out_path}")
    print(f"     Shape          : {df.shape}")
    print(f"     Date range     : {df['Date'].min()}  ->  {df['Date'].max()}")
    print(f"     Routes         : {sorted(df['Route_ID'].unique())}")
    print(f"     Passenger range: {df['Passenger_Count'].min()} - "
          f"{df['Passenger_Count'].max()}")
    print("\n[!]  DISCLAIMER: This dataset is entirely SYNTHETIC.")
    print("     It does not represent real passenger observations.")
