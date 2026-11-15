"""
======================================================================
 Public Transport Demand Predictor — Streamlit Dashboard
======================================================================
 Run:  streamlit run app.py
======================================================================
 Sections:
   A. Project Overview
   B. Demand Prediction
   C. Prediction Results
   D. Analytics
   E. Recommendations
======================================================================
"""

import os, sys, json, warnings
import numpy  as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express     as px
import plotly.graph_objects as go
from datetime import date, datetime

warnings.filterwarnings("ignore")

# ── Resolve paths regardless of where streamlit is launched from ──────
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
ASSET_DIR = os.path.join(ROOT_DIR, "assets")
SRC_DIR   = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be FIRST streamlit call)
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Public Transport Demand Predictor",
    page_icon  = "🚌",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — green/blue sustainability theme
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 100%);
    border: 1px solid #c8e6c9;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #37474f; font-weight: 600; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#1b5e20 0%,#0d47a1 100%); }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stDateInput label { color: #c8e6c9 !important; font-size:0.85rem; }

/* ── Section headers ── */
.section-header {
    background: linear-gradient(90deg,#1b5e20,#0d47a1);
    color: #fff;
    padding: 10px 20px;
    border-radius: 10px;
    margin-bottom: 16px;
    font-size: 1.1rem;
    font-weight: 700;
}

/* ── Demand badges ── */
.badge-low    { background:#c8e6c9; color:#1b5e20; padding:6px 14px; border-radius:20px; font-weight:700; }
.badge-mod    { background:#fff9c4; color:#f57f17; padding:6px 14px; border-radius:20px; font-weight:700; }
.badge-high   { background:#ffe0b2; color:#e65100; padding:6px 14px; border-radius:20px; font-weight:700; }
.badge-vhigh  { background:#ffcdd2; color:#b71c1c; padding:6px 14px; border-radius:20px; font-weight:700; }

/* ── Warning box ── */
.warning-box {
    background:#fff3e0; border-left:5px solid #ff9800;
    padding:12px 16px; border-radius:8px; margin:10px 0;
}
.critical-box {
    background:#ffebee; border-left:5px solid #f44336;
    padding:12px 16px; border-radius:8px; margin:10px 0;
}

/* ── Info card ── */
.info-card {
    background: #f1f8e9; border:1px solid #aed581;
    border-radius:12px; padding:16px;
}

/* ── Footer ── */
.footer { text-align:center; color:#9e9e9e; font-size:0.78rem; margin-top:30px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  CACHED RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    # 1. Cloud-native training (100% immune to cross-platform pickle / OS mismatches)
    csv_path = os.path.join(DATA_DIR, "processed_data.csv")
    cols_path = os.path.join(MODEL_DIR, "feature_cols.json")
    if os.path.exists(csv_path) and os.path.exists(cols_path):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            with open(cols_path, "r") as f:
                fcols = json.load(f)
            df = pd.read_csv(csv_path)
            X = df[fcols]
            y = df["Passenger_Count"]
            m = HistGradientBoostingRegressor(max_iter=100, random_state=42)
            m.fit(X, y)
            return m
        except Exception:
            pass

    # 2. Fallback to pre-trained pickle
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass

    return None


@st.cache_resource
def load_encoders():
    path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass  # Fall back to default fitted encoders

    # Fallback default LabelEncoders
    from sklearn.preprocessing import LabelEncoder
    encs = {}
    for col, classes in [
        ("Route_ID", ["R01", "R02", "R03", "R04", "R05"]),
        ("Day_of_Week", ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]),
        ("Weather", ["Clear", "Cloudy", "Foggy", "Rainy"])
    ]:
        le = LabelEncoder()
        le.fit(classes)
        encs[col] = le
    return encs


@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "model_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, "processed_data.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


@st.cache_data
def load_raw_data():
    path = os.path.join(DATA_DIR, "transport_demand_synthetic.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["Date"])


@st.cache_data
def load_feature_cols():
    path = os.path.join(MODEL_DIR, "feature_cols.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
DEFAULT_THRESHOLDS = {"low": 60, "moderate": 110, "high": 160, "bus_cap": 60}

def classify_demand(count: float, thresholds: dict) -> tuple[str, str]:
    """Return (label, badge_class) based on configurable thresholds."""
    if count <= thresholds["low"]:
        return "🟢 Low",         "badge-low"
    elif count <= thresholds["moderate"]:
        return "🟡 Moderate",    "badge-mod"
    elif count <= thresholds["high"]:
        return "🟠 High",        "badge-high"
    else:
        return "🔴 Very High",   "badge-vhigh"


def encode_input(value, encoder, fallback=0):
    """Safely encode a user-provided string using a fitted LabelEncoder."""
    try:
        return int(encoder.transform([str(value)])[0])
    except Exception:
        return fallback


def build_feature_row(route, hour, day_name, is_weekend, temperature,
                      weather, is_holiday, month, encoders,
                      feature_cols) -> pd.DataFrame:
    """
    Convert user selections into the exact feature vector the model expects.
    Mirrors the feature engineering done during preprocessing.
    """
    hour_sin  = np.sin(2 * np.pi * hour  / 24)
    hour_cos  = np.cos(2 * np.pi * hour  / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    quarter     = (month - 1) // 3 + 1
    week_of_year = datetime.now().isocalendar()[1]

    row = {
        "Route_ID"   : encode_input(route,    encoders["Route_ID"]),
        "Hour"       : hour,
        "Hour_sin"   : hour_sin,
        "Hour_cos"   : hour_cos,
        "Day_of_Week": encode_input(day_name, encoders["Day_of_Week"]),
        "Is_Weekend" : int(is_weekend),
        "Temperature": temperature,
        "Weather"    : encode_input(weather,  encoders["Weather"]),
        "Is_Holiday" : int(is_holiday),
        "Month"      : month,
        "Month_sin"  : month_sin,
        "Month_cos"  : month_cos,
        "Quarter"    : quarter,
        "WeekOfYear" : week_of_year,
    }
    return pd.DataFrame([row])[feature_cols]


def plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply a consistent, clean theme to every Plotly figure."""
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="Inter", size=12, color="#37474f"),
        margin        = dict(l=20, r=20, t=50, b=20),
        legend        = dict(bgcolor="rgba(255,255,255,0.8)",
                              bordercolor="#e0e0e0", borderwidth=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", linecolor="#e0e0e0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", linecolor="#e0e0e0")
    return fig


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚌 Transport Demand\n### Predictor")
    st.markdown("---")
    st.markdown("**🧭 Navigation**")
    page = st.radio("", ["🏠 Project Overview",
                          "🔮 Demand Prediction",
                          "📊 Analytics",
                          "⚙️  Model Info",
                          "📋 Recommendations"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**⚙️ Demand Thresholds**")
    t_low = st.slider("Low ≤",      10, 100, DEFAULT_THRESHOLDS["low"],      5)
    t_mod = st.slider("Moderate ≤", 50, 200, DEFAULT_THRESHOLDS["moderate"], 5)
    t_hi  = st.slider("High ≤",     80, 300, DEFAULT_THRESHOLDS["high"],     5)
    t_cap = st.slider("Bus Capacity", 20, 150, DEFAULT_THRESHOLDS["bus_cap"], 5)

    thresholds = {"low": t_low, "moderate": t_mod, "high": t_hi,
                  "bus_cap": t_cap}

    st.markdown("---")
    st.markdown("**🌱 SDG Alignment**")
    st.markdown("🏙️ **SDG 11** – Sustainable Cities")
    st.markdown("⚙️ **SDG 9**  – Industry & Innovation")
    st.markdown("🌍 **SDG 13** – Climate Action")

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem;opacity:0.7">⚠️ Synthetic dataset —\nnot real passenger data</p>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  LOAD RESOURCES
# ══════════════════════════════════════════════════════════════════════
model        = load_model()
encoders     = load_encoders()
metrics_data = load_metrics()
proc_df      = load_data()
raw_df       = load_raw_data()
feature_cols = load_feature_cols()

ROUTES   = ["R01", "R02", "R03", "R04", "R05"]
WEATHERS = ["Clear", "Cloudy", "Rainy", "Foggy"]

model_ready = all(x is not None for x in [model, encoders, feature_cols])


# ══════════════════════════════════════════════════════════════════════
#  A. PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="section-header">🏠 Project Overview</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
# 🚌 Public Transport Demand Predictor
### Using Machine Learning for Sustainable Urban Mobility

This prototype uses machine learning to **predict passenger demand** on 
urban bus routes — helping transit authorities:

- 📉 **Reduce overcrowding** on high-demand routes
- 🚍 **Optimise bus deployment** and reduce empty runs  
- 🌱 **Lower carbon emissions** through smarter scheduling
- 📊 **Make data-driven decisions** for route planning
        """)

    with col2:
        st.markdown("""
<div class="info-card">
<b>🎓 SDG Alignment</b><br><br>
🏙️ <b>SDG 11</b> — Sustainable Cities<br>
Make cities inclusive, safe, resilient and sustainable<br><br>
⚙️ <b>SDG 9</b> — Industry & Innovation<br>
Build resilient infrastructure and promote innovation<br><br>
🌍 <b>SDG 13</b> — Climate Action<br>
Take urgent action to combat climate change
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, icon, title, desc in zip(
        [col1, col2, col3, col4, col5],
        ["📂", "🔧", "🤖", "📊", "🔮"],
        ["Data", "Preprocess", "Train", "Evaluate", "Predict"],
        ["Synthetic dataset\n5 routes, 2 years",
         "Clean, encode,\nfeature engineer",
         "3 ML models\ncompared",
         "MAE / RMSE / R²\nmetrics",
         "Real-time\ndemand forecast"],
    ):
        col.metric(icon + " " + title, desc.split("\n")[0], desc.split("\n")[1])

    st.markdown("---")
    st.markdown("### 📂 Dataset at a Glance")

    if raw_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Total Records", f"{len(raw_df):,}")
        c2.metric("🛣️ Routes", raw_df["Route_ID"].nunique())
        c3.metric("📅 Date Range",
                  f"{raw_df['Date'].dt.year.min()}–{raw_df['Date'].dt.year.max()}")
        c4.metric("👥 Avg Passengers",
                  f"{raw_df['Passenger_Count'].mean():.0f}")

        with st.expander("📄 Sample Dataset (first 10 rows)"):
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.caption("⚠️ DISCLAIMER: This is a SYNTHETIC dataset generated "
                       "for demonstration purposes only. It does not represent "
                       "real passenger observations from any transit authority.")
    else:
        st.warning("Dataset not found. Run `python src/generate_dataset.py` first.")

    if not model_ready:
        st.warning("🔧 Model not trained yet. Run `python src/train_model.py` to train.")


# ══════════════════════════════════════════════════════════════════════
#  B & C. DEMAND PREDICTION + RESULTS
# ══════════════════════════════════════════════════════════════════════
elif "Prediction" in page:
    st.markdown('<div class="section-header">🔮 Demand Prediction</div>',
                unsafe_allow_html=True)

    if not model_ready:
        st.error("🔧 Model not found. Please run `python src/train_model.py` first.")
        st.stop()

    # ── Input form ────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown("#### 🎛️ Configure your prediction")
        c1, c2, c3 = st.columns(3)

        with c1:
            route      = st.selectbox("🛣️ Bus Route", ROUTES)
            sel_date   = st.date_input("📅 Date",
                                        value=date.today(),
                                        min_value=date(2020, 1, 1),
                                        max_value=date(2030, 12, 31))
            hour       = st.slider("🕐 Hour of Day", 0, 23, 8,
                                    format="%d:00")

        with c2:
            weather     = st.selectbox("🌤️ Weather Condition", WEATHERS)
            temperature = st.slider("🌡️ Temperature (°C)", 10, 45, 28)
            is_holiday  = st.toggle("🏖️ Public Holiday?", value=False)

        with c3:
            st.markdown("**ℹ️ Selected Summary**")
            st.markdown(f"- Route: **{route}**")
            sel_dt     = datetime.combine(sel_date, datetime.min.time())
            day_name   = sel_dt.strftime("%A")
            is_weekend = int(sel_dt.weekday() >= 5)
            month      = sel_dt.month

            st.markdown(f"- Date: **{sel_date}** ({day_name})")
            st.markdown(f"- Time: **{hour:02d}:00**")
            st.markdown(f"- Weather: **{weather}** / **{temperature}°C**")
            st.markdown(f"- Weekend: **{'Yes' if is_weekend else 'No'}**")
            st.markdown(f"- Holiday: **{'Yes' if is_holiday else 'No'}**")

        submitted = st.form_submit_button("🔮 Predict Demand", type="primary",
                                           use_container_width=True)

    # ── Prediction results ────────────────────────────────────────────
    if submitted:
        try:
            X_input   = build_feature_row(
                route, hour, day_name, is_weekend, temperature,
                weather, is_holiday, month, encoders, feature_cols,
            )
            prediction = float(model.predict(X_input)[0])
            prediction = max(0, round(prediction))

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # ── Section C — Results ───────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">📋 Prediction Results</div>',
                    unsafe_allow_html=True)

        demand_label, badge_cls = classify_demand(prediction, thresholds)
        buses_needed = max(1, int(np.ceil(prediction / thresholds["bus_cap"])))
        load_pct     = (prediction / thresholds["bus_cap"]) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Predicted Passengers", f"{prediction}")
        c2.metric("🚍 Demand Level",          demand_label)
        c3.metric("🚌 Buses Needed",           f"{buses_needed}")
        c4.metric("📊 Load per Bus",           f"{load_pct:.0f}%")

        # Demand badge
        st.markdown(
            f'<p>Demand category: <span class="{badge_cls}">'
            f'{demand_label}</span></p>',
            unsafe_allow_html=True,
        )

        # Overcrowding warnings
        if prediction > thresholds["high"]:
            st.markdown(
                f'<div class="critical-box">🚨 <b>Overcrowding Risk!</b> '
                f'Predicted {prediction} passengers on route {route} at '
                f'{hour:02d}:00. Single-bus capacity is {thresholds["bus_cap"]}. '
                f'Consider deploying <b>{buses_needed} buses</b>.</div>',
                unsafe_allow_html=True,
            )
        elif prediction > thresholds["moderate"]:
            st.markdown(
                f'<div class="warning-box">⚠️ <b>High Demand.</b> '
                f'Monitor route {route} at {hour:02d}:00 closely. '
                f'Consider standby capacity.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.success(f"✅ Demand is within normal range for route {route}.")

        # 24-hour demand profile for selected route & date
        st.markdown("#### 📈 24-Hour Demand Profile (selected route & date)")
        hours_all  = list(range(6, 24))
        hour_preds = []
        for h in hours_all:
            xi = build_feature_row(route, h, day_name, is_weekend,
                                   temperature, weather, is_holiday,
                                   month, encoders, feature_cols)
            hour_preds.append(max(0, float(model.predict(xi)[0])))

        fig_hr = px.area(
            x=hours_all, y=hour_preds,
            labels={"x": "Hour", "y": "Passengers"},
            color_discrete_sequence=["#1976D2"],
            title=f"Predicted Demand — Route {route} on {sel_date}",
        )
        fig_hr.add_vline(x=hour, line_dash="dash", line_color="#E91E63",
                          annotation_text=f"Selected: {hour:02d}:00",
                          annotation_position="top right")
        # Colour background by demand tier
        fig_hr.add_hrect(y0=0,              y1=thresholds["low"],
                          fillcolor="#c8e6c9", opacity=0.2, line_width=0)
        fig_hr.add_hrect(y0=thresholds["low"],      y1=thresholds["moderate"],
                          fillcolor="#fff9c4", opacity=0.2, line_width=0)
        fig_hr.add_hrect(y0=thresholds["moderate"], y1=thresholds["high"],
                          fillcolor="#ffe0b2", opacity=0.2, line_width=0)
        fig_hr.add_hrect(y0=thresholds["high"],     y1=max(hour_preds) * 1.1,
                          fillcolor="#ffcdd2", opacity=0.2, line_width=0)
        st.plotly_chart(plotly_theme(fig_hr), use_container_width=True)

        # Gauge chart for capacity utilisation
        col_g, col_r = st.columns(2)
        with col_g:
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = min(load_pct, 200),
                delta = {"reference": 100, "suffix": "%"},
                title = {"text": "Bus Load (% of single-bus capacity)"},
                gauge = {
                    "axis"    : {"range": [0, 200]},
                    "bar"     : {"color": "#1976D2"},
                    "steps"   : [
                        {"range": [0,   100], "color": "#c8e6c9"},
                        {"range": [100, 150], "color": "#ffe0b2"},
                        {"range": [150, 200], "color": "#ffcdd2"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4},
                                   "thickness": 0.75, "value": 100},
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_r:
            # Route comparison at selected hour
            route_preds = {}
            for r in ROUTES:
                xi = build_feature_row(r, hour, day_name, is_weekend,
                                       temperature, weather, is_holiday,
                                       month, encoders, feature_cols)
                route_preds[r] = max(0, float(model.predict(xi)[0]))

            fig_bar = px.bar(
                x=list(route_preds.keys()),
                y=list(route_preds.values()),
                color=list(route_preds.values()),
                color_continuous_scale=["#c8e6c9", "#fff9c4", "#ffe0b2", "#ffcdd2"],
                labels={"x": "Route", "y": "Passengers", "color": "Passengers"},
                title=f"All Routes — {hour:02d}:00 on {sel_date}",
            )
            fig_bar.update_coloraxes(showscale=False)
            st.plotly_chart(plotly_theme(fig_bar), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  D. ANALYTICS
# ══════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div class="section-header">📊 Analytics Dashboard</div>',
                unsafe_allow_html=True)

    if raw_df is None or proc_df is None:
        st.warning("Data not found. Please run the pipeline first.")
        st.stop()

    # ── Tab layout ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "⏰ Hourly Demand",
        "🛣️ Route Comparison",
        "📅 Weekday vs Weekend",
        "🎯 Actual vs Predicted",
    ])

    with tab1:
        st.markdown("#### Average Passengers by Hour of Day")
        hourly = raw_df.groupby(["Hour", "Route_ID"])["Passenger_Count"] \
                        .mean().reset_index()
        fig = px.line(hourly, x="Hour", y="Passenger_Count",
                      color="Route_ID",
                      markers=True,
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={"Passenger_Count": "Avg Passengers",
                               "Route_ID": "Route"},
                      title="Hourly Demand Profile by Route")
        fig.update_xaxes(tickvals=list(range(6, 24)))
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        st.markdown("#### Demand Heatmap (Hour × Day of Week)")
        pivot = raw_df.groupby(["Day_of_Week", "Hour"])["Passenger_Count"] \
                       .mean().reset_index()
        day_order = ["Monday","Tuesday","Wednesday","Thursday",
                     "Friday","Saturday","Sunday"]
        pivot["Day_of_Week"] = pd.Categorical(pivot["Day_of_Week"],
                                               categories=day_order, ordered=True)
        pivot = pivot.sort_values("Day_of_Week")
        heatmap_data = pivot.pivot(index="Day_of_Week", columns="Hour",
                                    values="Passenger_Count")
        fig_hm = px.imshow(heatmap_data,
                            color_continuous_scale="YlGnBu",
                            labels={"color": "Avg Passengers"},
                            title="Average Passengers: Hour × Day of Week")
        st.plotly_chart(plotly_theme(fig_hm), use_container_width=True)

    with tab2:
        st.markdown("#### Average Passengers by Route")
        route_avg = raw_df.groupby("Route_ID")["Passenger_Count"] \
                           .agg(["mean", "std", "max"]).reset_index()
        route_avg.columns = ["Route", "Mean", "Std", "Max"]

        fig = px.bar(route_avg, x="Route", y="Mean",
                      error_y="Std",
                      color="Mean",
                      color_continuous_scale=["#a5d6a7","#1b5e20"],
                      labels={"Mean": "Avg Passengers"},
                      title="Route-wise Average Demand")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        st.markdown("#### Distribution of Passenger Counts per Route")
        fig2 = px.box(raw_df, x="Route_ID", y="Passenger_Count",
                      color="Route_ID",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={"Passenger_Count": "Passengers",
                               "Route_ID": "Route"},
                      title="Passenger Count Distribution by Route")
        st.plotly_chart(plotly_theme(fig2), use_container_width=True)

    with tab3:
        st.markdown("#### Weekday vs Weekend Demand Comparison")
        raw_df["Period"] = raw_df["Is_Weekend"].map({0: "Weekday", 1: "Weekend"})
        wkd_hr = raw_df.groupby(["Hour", "Period"])["Passenger_Count"] \
                        .mean().reset_index()
        fig = px.line(wkd_hr, x="Hour", y="Passenger_Count",
                      color="Period",
                      color_discrete_map={"Weekday": "#1976D2",
                                           "Weekend": "#43A047"},
                      markers=True,
                      labels={"Passenger_Count": "Avg Passengers"},
                      title="Average Hourly Demand: Weekday vs Weekend")
        fig.update_xaxes(tickvals=list(range(6, 24)))
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        st.markdown("#### Impact of Weather on Ridership")
        wth = raw_df.groupby("Weather")["Passenger_Count"].mean().reset_index()
        fig3 = px.bar(wth, x="Weather", y="Passenger_Count",
                      color="Passenger_Count",
                      color_continuous_scale=["#c8e6c9","#1b5e20"],
                      labels={"Passenger_Count": "Avg Passengers"},
                      title="Average Ridership by Weather Condition")
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_theme(fig3), use_container_width=True)

    with tab4:
        if not model_ready:
            st.warning("Model not trained yet.")
        else:
            st.markdown("#### Actual vs Predicted Passengers (Test Sample)")
            # Use a sample from processed data for speed
            sample = proc_df.sample(min(500, len(proc_df)),
                                     random_state=42)
            # Reconstruct target from processed data
            if "Passenger_Count" in sample.columns:
                X_samp = sample[feature_cols]
                y_actual = sample["Passenger_Count"].values
                y_pred   = np.maximum(0, model.predict(X_samp))

                fig = px.scatter(
                    x=y_actual, y=y_pred,
                    opacity=0.5,
                    color_discrete_sequence=["#1976D2"],
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Actual vs Predicted Passenger Count (sample)",
                )
                mn = min(y_actual.min(), y_pred.min()) - 5
                mx = max(y_actual.max(), y_pred.max()) + 5
                fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                          mode="lines",
                                          line=dict(color="red", dash="dash"),
                                          name="Perfect Prediction"))
                st.plotly_chart(plotly_theme(fig), use_container_width=True)
            else:
                st.info("Target column not found in processed data.")

            st.markdown("#### Monthly Demand Trend")
            if "Date" in proc_df.columns and "Passenger_Count" in proc_df.columns:
                proc_df["YearMonth"] = proc_df["Date"].dt.to_period("M").astype(str)
                monthly = proc_df.groupby("YearMonth")["Passenger_Count"] \
                                   .mean().reset_index()
                fig_m = px.line(monthly, x="YearMonth", y="Passenger_Count",
                                markers=True,
                                color_discrete_sequence=["#43A047"],
                                labels={"Passenger_Count": "Avg Passengers",
                                         "YearMonth": "Month"},
                                title="Monthly Average Passenger Count")
                fig_m.update_xaxes(tickangle=45)
                st.plotly_chart(plotly_theme(fig_m), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  E-i. MODEL INFO
# ══════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown('<div class="section-header">⚙️ Model Information</div>',
                unsafe_allow_html=True)

    if metrics_data is None:
        st.warning("Metrics not found. Please run `python src/train_model.py` first.")
        st.stop()

    best_name = metrics_data.get("best_model", "N/A")
    results   = metrics_data.get("results", [])

    st.success(f"✅ Best Model: **{best_name}**")
    st.markdown("""
**Selection Rationale:**  
Random Forest was selected because it achieves the lowest RMSE on the 
held-out test set. RMSE is preferred over MAE here because large 
prediction errors (e.g. predicting 50 when 150 passengers arrive) have 
serious real-world consequences for capacity planning — RMSE penalises 
these disproportionately.
    """)

    # Metrics table
    st.markdown("#### 📊 Model Comparison")
    df_m = pd.DataFrame(results)
    st.dataframe(
        df_m.style.highlight_min(subset=["MAE","RMSE"], color="#c8e6c9")
                   .highlight_max(subset=["R2"],        color="#c8e6c9")
                   .format({"MAE":"{:.2f}","RMSE":"{:.2f}","R2":"{:.4f}"}),
        use_container_width=True,
    )

    # Bar charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_m, x="Model", y="RMSE",
                     color="RMSE",
                     color_continuous_scale=["#a5d6a7","#f44336"],
                     title="RMSE by Model (lower = better)")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

    with col2:
        fig2 = px.bar(df_m, x="Model", y="R2",
                       color="R2",
                       color_continuous_scale=["#f44336","#1b5e20"],
                       title="R² Score by Model (higher = better)")
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_theme(fig2), use_container_width=True)

    # Feature importance image
    fi_path = os.path.join(ASSET_DIR, "feature_importance.png")
    if os.path.exists(fi_path):
        st.markdown("#### 🌳 Feature Importance")
        st.image(fi_path, use_container_width=True)

    # Explain models
    st.markdown("---")
    st.markdown("#### 📖 Model Explanations")
    with st.expander("Linear Regression"):
        st.markdown("""
Assumes a linear relationship between features and passenger count.
Fast to train but limited — it cannot capture non-linear patterns like
rush-hour spikes or weather interactions. Useful as a quick baseline above
the mean predictor.
        """)
    with st.expander("Decision Tree"):
        st.markdown("""
Splits data into axis-aligned regions. Captures non-linearity but tends
to overfit on training data. `max_depth` is limited to control complexity.
Good for understanding *why* a prediction was made (interpretability).
        """)
    with st.expander("Random Forest (selected)"):
        st.markdown("""
An ensemble of 150 decision trees, each trained on a random bootstrap
sample with a random feature subset. Averaging predictions across trees
reduces variance and overfitting. This prototype's Random Forest achieves
the best RMSE on the test set without requiring feature scaling.
        """)


# ══════════════════════════════════════════════════════════════════════
#  E-ii. RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════
elif "Recommendations" in page:
    st.markdown('<div class="section-header">📋 Operational Recommendations</div>',
                unsafe_allow_html=True)

    if not model_ready:
        st.warning("Model not trained yet. Run `python src/train_model.py` first.")
        st.stop()

    sel_route = st.selectbox("Select Route to Analyse", ROUTES)
    sel_dow   = st.selectbox("Day of Week",
                               ["Monday","Tuesday","Wednesday",
                                "Thursday","Friday","Saturday","Sunday"])
    sel_wth   = st.selectbox("Weather", WEATHERS)
    sel_temp  = st.slider("Temperature (°C)", 10, 45, 28)
    sel_hol   = st.toggle("Public Holiday?", False)

    is_we    = int(sel_dow in ["Saturday", "Sunday"])
    sel_mn   = datetime.now().month

    if st.button("📋 Generate Recommendations", type="primary"):
        hours_range = list(range(6, 24))
        all_preds   = []
        for h in hours_range:
            xi = build_feature_row(sel_route, h, sel_dow, is_we,
                                   sel_temp, sel_wth, sel_hol,
                                   sel_mn, encoders, feature_cols)
            p = max(0, float(model.predict(xi)[0]))
            all_preds.append(p)

        df_recs = pd.DataFrame({"Hour": hours_range, "Passengers": all_preds})
        df_recs["Demand"]       = df_recs["Passengers"].apply(
            lambda x: classify_demand(x, thresholds)[0])
        df_recs["Buses_Needed"] = df_recs["Passengers"].apply(
            lambda x: max(1, int(np.ceil(x / thresholds["bus_cap"]))))

        # Demand summary
        peak_hour = df_recs.loc[df_recs["Passengers"].idxmax(), "Hour"]
        peak_pass = df_recs["Passengers"].max()
        low_hour  = df_recs.loc[df_recs["Passengers"].idxmin(), "Hour"]

        st.markdown("### 📊 Daily Demand Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔺 Peak Hour",        f"{peak_hour:02d}:00")
        c2.metric("👥 Peak Passengers",  f"{peak_pass:.0f}")
        c3.metric("🔻 Lowest Hour",      f"{low_hour:02d}:00")
        c4.metric("🚌 Max Buses Needed",
                   str(df_recs["Buses_Needed"].max()))

        # Area chart
        fig = px.area(df_recs, x="Hour", y="Passengers",
                       color_discrete_sequence=["#1976D2"],
                       title=f"Predicted Demand — Route {sel_route} "
                             f"({sel_dow}, {sel_wth})")
        fig.add_hline(y=thresholds["bus_cap"], line_dash="dash",
                       line_color="orange",
                       annotation_text="Single Bus Capacity")
        fig.add_hline(y=thresholds["high"], line_dash="dot",
                       line_color="red",
                       annotation_text="High Demand Threshold")
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

        # Detailed table
        with st.expander("📋 Hour-by-Hour Schedule"):
            st.dataframe(df_recs.style.map(
                lambda v: "background-color:#ffcdd2" if "Very High" in str(v)
                     else "background-color:#ffe0b2" if "High" in str(v)
                     else "background-color:#fff9c4" if "Moderate" in str(v)
                     else "",
                subset=["Demand"]
            ), use_container_width=True)

        # Textual recommendations
        st.markdown("### 💡 Recommendations")
        overcrowded = df_recs[df_recs["Passengers"] > thresholds["high"]]
        low_demand  = df_recs[df_recs["Passengers"] <= thresholds["low"]]

        if not overcrowded.empty:
            hours_str = ", ".join(f"{h:02d}:00" for h in overcrowded["Hour"])
            st.markdown(
                f'<div class="critical-box">🚨 <b>Add capacity at:</b> {hours_str}<br>'
                f'Consider deploying extra buses or increasing frequency '
                f'on route {sel_route} during these hours.</div>',
                unsafe_allow_html=True,
            )

        if not low_demand.empty:
            hours_str = ", ".join(f"{h:02d}:00" for h in low_demand["Hour"])
            st.markdown(
                f'<div class="warning-box">💡 <b>Consider reducing frequency at:</b> '
                f'{hours_str}<br>Low ridership periods — using smaller buses or '
                f'reducing frequency can lower fuel consumption and emissions.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("""
<div class="info-card">
<b>🌱 Sustainability Impact</b><br><br>
• Deploying correctly-sized buses reduces fuel waste and CO₂ emissions<br>
• Data-driven scheduling improves service reliability → more passengers 
  shift from private cars to public transport<br>
• Reduced overcrowding improves passenger comfort and safety<br><br>
<b>⚠️ Note:</b> These recommendations are based on a synthetic prototype model.
Real deployment requires validation against actual ridership data.
</div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Public Transport Demand Predictor | '
    'SDG 11 · SDG 9 · SDG 13 | '
    '⚠️ Synthetic dataset — not real passenger data | '
    'Built with Streamlit & Scikit-learn</div>',
    unsafe_allow_html=True,
)
