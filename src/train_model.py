"""
======================================================================
 Public Transport Demand Predictor — Model Training & Evaluation
======================================================================
 Models compared:
   1. Baseline         (mean predictor — sanity check)
   2. Linear Regression
   3. Decision Tree Regressor
   4. Random Forest Regressor

 Metrics:
   • MAE   — Mean Absolute Error
   • RMSE  — Root Mean Squared Error
   • R²    — Coefficient of Determination

 The best model is selected on RMSE and saved to disk.
======================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figs
import matplotlib.pyplot as plt

from sklearn.dummy       import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree        import DecisionTreeRegressor
from sklearn.ensemble    import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics     import (mean_absolute_error,
                                  mean_squared_error,
                                  r2_score)

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")
ASSET_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

# Import our preprocessing pipeline
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import run_pipeline


# ══════════════════════════════════════════════════════════════════════
#  Metric helpers
# ══════════════════════════════════════════════════════════════════════
def evaluate(name: str, model, X_test, y_test) -> dict:
    """Compute MAE, RMSE and R² for a fitted model."""
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    print(f"  {name:<28} MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}")
    return {"Model": name, "MAE": round(mae, 4),
            "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ══════════════════════════════════════════════════════════════════════
#  Model definitions
# ══════════════════════════════════════════════════════════════════════
def get_models() -> dict:
    """
    Return a dictionary of {name: unfitted_model} to train and compare.
    Hyperparameters are kept at sensible defaults; tuning is out of
    scope for this prototype.
    """
    return {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth   = 12,
            min_samples_leaf = 10,
            random_state= 42,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators      = 150,
            max_depth         = 15,
            min_samples_leaf  = 5,
            n_jobs            = -1,       # use all CPU cores
            random_state      = 42,
        ),

        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter          = 150,
            max_depth         = 12,
            min_samples_leaf  = 10,
            random_state      = 42,
        ),
    }


# ══════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ══════════════════════════════════════════════════════════════════════
def plot_metrics(results: list[dict], save_path: str) -> None:
    """Bar chart comparing MAE / RMSE / R² across all models."""
    df    = pd.DataFrame(results)
    names = df["Model"]
    x     = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    for ax, metric, colour in zip(axes,
                                   ["MAE", "RMSE", "R2"],
                                   ["#2196F3", "#4CAF50", "#FF9800"]):
        bars = ax.bar(x, df[metric], width=0.6, color=colour, alpha=0.85,
                      edgecolor="white")
        ax.set_title(metric, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_ylim(0, df[metric].max() * 1.25 if df[metric].max() > 0 else 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Saved model comparison → {save_path}")


def plot_actual_vs_predicted(y_test, preds, model_name: str,
                              save_path: str) -> None:
    """Scatter plot of actual vs predicted values for the best model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.3, s=12, color="#2196F3",
               label="Predictions")
    lims = [min(y_test.min(), preds.min()) - 10,
            max(y_test.max(), preds.max()) + 10]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Passenger Count",  fontsize=11)
    ax.set_ylabel("Predicted Passenger Count", fontsize=11)
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=12,
                 fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Saved actual vs predicted → {save_path}")


def plot_residuals(y_test, preds, model_name: str, save_path: str) -> None:
    """Plot residuals vs predicted and residual error distribution."""
    residuals = y_test - preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Residuals vs Predicted
    ax1.scatter(preds, residuals, alpha=0.3, s=12, color="#00897B")
    ax1.axhline(0, color="red", linestyle="--", lw=1.5)
    ax1.set_xlabel("Predicted Passenger Count", fontsize=11)
    ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
    ax1.set_title(f"Residuals vs Predicted — {model_name}", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # 2. Residual Distribution Histogram
    ax2.hist(residuals, bins=40, color="#1E88E5", edgecolor="white", alpha=0.85, density=True)
    mu, std = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    p_norm = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / std) ** 2)
    ax2.plot(x_norm, p_norm, "r-", lw=2, label=f"Normal Fit (μ={mu:.2f}, σ={std:.2f})")
    ax2.set_xlabel("Residual Error", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(f"Residual Distribution — {model_name}", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Saved residual analysis → {save_path}")


def plot_feature_importance(model, feature_names: list,
                             save_path: str) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#4CAF50" if importances[i] > np.median(importances)
               else "#90CAF9" for i in indices]
    ax.barh([feature_names[i] for i in indices],
            importances[indices], color=colours, edgecolor="white")
    ax.set_title("Feature Importance (Random Forest)", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  Saved feature importance → {save_path}")


# ══════════════════════════════════════════════════════════════════════
#  Main training routine
# ══════════════════════════════════════════════════════════════════════
def train_and_evaluate():
    # ── 1. Preprocess ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, encoders = run_pipeline()

    # ── 2. Train & evaluate all models ────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    models  = get_models()
    results = []

    for name, model in models.items():
        print(f"\n  Training: {name}")
        model.fit(X_train, y_train)
        metrics = evaluate(name, model, X_test, y_test)
        metrics["fitted_model"] = model
        results.append(metrics)

    # ── 3. Select best model (lowest RMSE, excluding baseline) ────────
    non_baseline = [r for r in results if r["Model"] != "Baseline (Mean)"]
    best_result  = min(non_baseline, key=lambda r: r["RMSE"])
    best_model   = best_result["fitted_model"]
    best_name    = best_result["Model"]

    print(f"\n  ✅  Best model: {best_name}  (RMSE={best_result['RMSE']})")
    print(f"      Selection rationale: lowest RMSE on held-out test set,")
    print(f"      which penalises large errors — critical for capacity planning.")

    # ── 4. Save best model ────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n[Save]  Best model → {model_path}")

    # ── 5. Save metrics for dashboard ─────────────────────────────────
    metrics_to_save = [{k: v for k, v in r.items() if k != "fitted_model"}
                       for r in results]
    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"results": metrics_to_save, "best_model": best_name}, f,
                  indent=2)
    print(f"[Save]  Metrics     → {metrics_path}")

    # ── 6. Save feature column list ───────────────────────────────────
    feature_path = os.path.join(MODEL_DIR, "feature_cols.json")
    with open(feature_path, "w") as f:
        json.dump(X_train.columns.tolist(), f)
    print(f"[Save]  Feature list → {feature_path}")

    # ── 7. Generate plots ─────────────────────────────────────────────
    plot_metrics(
        [{k: v for k, v in r.items() if k != "fitted_model"} for r in results],
        os.path.join(ASSET_DIR, "model_comparison.png"),
    )

    best_preds = best_model.predict(X_test)
    plot_actual_vs_predicted(
        y_test.values, best_preds, best_name,
        os.path.join(ASSET_DIR, "actual_vs_predicted.png"),
    )
    plot_residuals(
        y_test.values, best_preds, best_name,
        os.path.join(ASSET_DIR, "residual_distribution.png"),
    )

    rf_model = next((r["fitted_model"] for r in results if r["Model"] == "Random Forest"), None)
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(
            best_model, X_train.columns.tolist(),
            os.path.join(ASSET_DIR, "feature_importance.png"),
        )
    elif rf_model is not None and hasattr(rf_model, "feature_importances_"):
        plot_feature_importance(
            rf_model, X_train.columns.tolist(),
            os.path.join(ASSET_DIR, "feature_importance.png"),
        )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — all artefacts saved")
    print("=" * 60 + "\n")

    return best_model, results


if __name__ == "__main__":
    train_and_evaluate()
