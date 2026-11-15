"""
======================================================================
 Public Transport Demand Predictor — Standalone Prediction Engine
======================================================================
 Provides both an importable Python API and an interactive CLI to 
 generate ridership demand predictions using the trained ML model.

 Usage via CLI:
   python src/predict.py --route R01 --date 2025-05-12 --hour 8 --weather Clear --temp 28.5
   
 Or interactive mode:
   python src/predict.py --interactive
======================================================================
"""

import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Paths
BASE_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEAT_PATH  = os.path.join(MODEL_DIR, "feature_cols.json")


def load_artifacts():
    """Load trained model, label encoders, and feature column list."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run train_model.py first.")
    
    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENC_PATH)
    with open(FEAT_PATH, "r") as f:
        feature_cols = json.load(f)
        
    return model, encoders, feature_cols


def predict_demand(route_id: str,
                   date_str: str,
                   hour: int,
                   weather: str,
                   temperature: float,
                   is_holiday: int = 0,
                   bus_capacity: int = 60,
                   threshold_low: int = 25,
                   threshold_mod: int = 50,
                   threshold_high: int = 75) -> dict:
    """
    Generate demand prediction and capacity recommendations for a single transit trip.
    """
    model, encoders, feature_cols = load_artifacts()
    
    # 1. Parse date and compute calendar attributes
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_name    = dt.strftime("%A")
    is_weekend  = 1 if dt.weekday() >= 5 else 0
    month       = dt.month
    quarter     = (month - 1) // 3 + 1
    week_of_yr  = dt.isocalendar()[1]
    
    # 2. Cyclical temporal encodings
    hour_sin    = np.sin(2 * np.pi * hour / 24)
    hour_cos    = np.cos(2 * np.pi * hour / 24)
    month_sin   = np.sin(2 * np.pi * month / 12)
    month_cos   = np.cos(2 * np.pi * month / 12)
    
    # 3. Label encode categoricals
    if route_id not in encoders["Route_ID"].classes_:
        raise ValueError(f"Unknown Route_ID '{route_id}'. Valid: {list(encoders['Route_ID'].classes_)}")
    route_enc = int(encoders["Route_ID"].transform([route_id])[0])
    
    if weather not in encoders["Weather"].classes_:
        # Fallback to closest or clear
        weather = "Clear"
    weather_enc = int(encoders["Weather"].transform([weather])[0])
    
    if day_name not in encoders["Day_of_Week"].classes_:
        day_name = "Monday"
    day_enc = int(encoders["Day_of_Week"].transform([day_name])[0])
    
    # 4. Construct input feature row
    row_dict = {
        "Route_ID":    route_enc,
        "Hour":        hour,
        "Hour_sin":    hour_sin,
        "Hour_cos":    hour_cos,
        "Day_of_Week": day_enc,
        "Is_Weekend":  is_weekend,
        "Temperature": temperature,
        "Weather":     weather_enc,
        "Is_Holiday":  is_holiday,
        "Month":       month,
        "Month_sin":   month_sin,
        "Month_cos":   month_cos,
        "Quarter":     quarter,
        "WeekOfYear":  week_of_yr
    }
    
    X_input = pd.DataFrame([row_dict])[feature_cols]
    
    # 5. Predict
    raw_pred = float(model.predict(X_input)[0])
    pred_count = max(0, int(round(raw_pred)))
    
    # 6. Demand Classification
    if pred_count <= threshold_low:
        demand_level = "Low"
        badge = "🟢"
    elif pred_count <= threshold_mod:
        demand_level = "Moderate"
        badge = "🟡"
    elif pred_count <= threshold_high:
        demand_level = "High"
        badge = "🟠"
    else:
        demand_level = "Very High"
        badge = "🔴"
        
    # 7. Overcrowding and Operational Recommendations
    utilization_pct = round((pred_count / bus_capacity) * 100, 1)
    is_overcrowded  = pred_count > bus_capacity
    buses_needed    = max(1, int(np.ceil(pred_count / bus_capacity)))
    
    if is_overcrowded:
        extra_buses = buses_needed - 1
        recommendation = (f"⚠️ OVERCROWDING ALERT: Passenger demand ({pred_count}) exceeds standard "
                          f"capacity ({bus_capacity}). Deploy {extra_buses} additional bus(es) "
                          f"or reduce headway to 5-7 minutes.")
    elif utilization_pct < 40:
        recommendation = (f"💡 LOW OCCUPANCY: Utilization is at {utilization_pct}%. Consider deploying "
                          f"a mini-bus or merging scheduled trips to conserve fuel (SDG 13).")
    else:
        recommendation = (f"✅ OPTIMAL CAPACITY: Route is running at healthy utilization ({utilization_pct}%). "
                          f"Maintain current regular schedule.")
        
    return {
        "route_id": route_id,
        "date": date_str,
        "day_of_week": day_name,
        "hour": hour,
        "weather": weather,
        "temperature": temperature,
        "predicted_passengers": pred_count,
        "demand_level": demand_level,
        "demand_badge": badge,
        "bus_capacity": bus_capacity,
        "utilization_pct": utilization_pct,
        "is_overcrowded": is_overcrowded,
        "buses_needed": buses_needed,
        "recommendation": recommendation
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Public Transit Passenger Demand")
    parser.add_argument("--route", type=str, default="R01", help="Route ID (e.g., R01-R05)")
    parser.add_argument("--date", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Date (YYYY-MM-DD)")
    parser.add_argument("--hour", type=int, default=8, help="Hour of the day (0-23)")
    parser.add_argument("--weather", type=str, default="Clear", choices=["Clear", "Cloudy", "Foggy", "Rainy"], help="Weather")
    parser.add_argument("--temp", type=float, default=26.0, help="Temperature in °C")
    parser.add_argument("--holiday", type=int, default=0, choices=[0, 1], help="Is Holiday (0 or 1)")
    parser.add_argument("--capacity", type=int, default=60, help="Bus seating capacity")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal prompt")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("\n" + "="*50)
        print("  🚌 PUBLIC TRANSPORT DEMAND PREDICTOR (CLI)")
        print("="*50)
        route   = input("Enter Route ID [R01-R05] (default R01): ").strip() or "R01"
        date_in = input(f"Enter Date YYYY-MM-DD (default {datetime.today().strftime('%Y-%m-%d')}): ").strip() or datetime.today().strftime("%Y-%m-%d")
        hour_in = input("Enter Hour [6-23] (default 8): ").strip() or "8"
        weath   = input("Enter Weather [Clear/Cloudy/Foggy/Rainy] (default Clear): ").strip() or "Clear"
        temp_in = input("Enter Temperature °C (default 25.0): ").strip() or "25.0"
        
        res = predict_demand(
            route_id=route,
            date_str=date_in,
            hour=int(hour_in),
            weather=weath,
            temperature=float(temp_in),
            bus_capacity=args.capacity
        )
    else:
        res = predict_demand(
            route_id=args.route,
            date_str=args.date,
            hour=args.hour,
            weather=args.weather,
            temperature=args.temp,
            is_holiday=args.holiday,
            bus_capacity=args.capacity
        )
        
    print("\n" + "-"*50)
    print(f"  PREDICTION REPORT: Route {res['route_id']} @ {res['hour']:02d}:00 hrs")
    print("-"*50)
    print(f"  • Date & Day:           {res['date']} ({res['day_of_week']})")
    print(f"  • Weather & Temp:       {res['weather']} @ {res['temperature']}°C")
    print(f"  • Predicted Passengers: {res['predicted_passengers']} commuters")
    print(f"  • Demand Classification:{res['demand_badge']} {res['demand_level']}")
    print(f"  • Bus Capacity / Load:  {res['bus_capacity']} seats ({res['utilization_pct']}% utilized)")
    print(f"  • Recommended Fleet:    {res['buses_needed']} bus(es)")
    print(f"\n  [Advisory]: {res['recommendation']}")
    print("-"*50 + "\n")


if __name__ == "__main__":
    main()
