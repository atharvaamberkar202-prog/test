# ============================================
# INDIA VIX VOLATILITY FORECAST DASHBOARD
# ============================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import plotly.graph_objects as go
from datetime import datetime

# --------------------------------------------
# CONFIG
# --------------------------------------------
st.set_page_config(page_title="India VIX Vol Dashboard", layout="wide")

st.title("📊 India VIX Volatility Forecast Dashboard")
st.caption("GARCH-based IV forecasting for Calendar & Diagonal Spread Optimization")

# --------------------------------------------
# FETCH DATA
# --------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data():
    vix = yf.download("^INDIAVIX", period="1y", interval="1d")
    nifty = yf.download("^NSEI", period="1y", interval="1d")

    df = pd.DataFrame()
    df["VIX"] = vix["Close"]
    df["NIFTY"] = nifty["Close"]

    df = df.dropna()
    df["returns"] = np.log(df["VIX"] / df["VIX"].shift(1)) * 100

    return df.dropna()

# --------------------------------------------
# GARCH MODEL
# --------------------------------------------
def run_garch(df, model_type="GARCH"):

    returns = df["returns"]

    if model_type == "EGARCH":
        model = arch_model(returns, vol='EGARCH', p=1, q=1)
    else:
        model = arch_model(returns, vol='GARCH', p=1, q=1)

    res = model.fit(disp="off")

    forecast = res.forecast(horizon=5)

    # Convert variance → volatility
    vol_forecast = np.sqrt(forecast.variance.values[-1]) * np.sqrt(252)

    return res, vol_forecast

# --------------------------------------------
# SIGNAL GENERATION
# --------------------------------------------
def generate_signal(current_vix, forecast_vol):

    avg_forecast = np.mean(forecast_vol)

    if current_vix < 13 and avg_forecast > current_vix:
        return "🟢 IV Expansion Expected → Favor Long Calendar / Diagonal Spreads"
    
    elif current_vix > 18 and avg_forecast < current_vix:
        return "🔴 IV Crush Risk → Avoid Long Vega / Consider Short Vega Structures"
    
    else:
        return "🟡 Neutral → Focus on Theta Decay Strategies"

# --------------------------------------------
# PLOT FUNCTIONS
# --------------------------------------------
def plot_vix(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["VIX"], name="India VIX"))
    fig.update_layout(title="India VIX Historical", template="plotly_dark")
    return fig

def plot_forecast(vol_forecast):
    days = np.arange(1, len(vol_forecast)+1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=vol_forecast, mode='lines+markers', name="Forecast Vol"))
    fig.update_layout(title="Volatility Forecast (Next 5 Days)", template="plotly_dark")
    return fig

# --------------------------------------------
# MAIN BUTTON
# --------------------------------------------
if st.button("🔄 Refresh & Run Model"):

    with st.spinner("Fetching data and running model..."):

        df = fetch_data()

        model_type = st.selectbox("Model Type", ["GARCH", "EGARCH"])

        res, vol_forecast = run_garch(df, model_type)

        current_vix = df["VIX"].iloc[-1]

        signal = generate_signal(current_vix, vol_forecast)

    # ----------------------------------------
    # DISPLAY METRICS
    # ----------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Current VIX", round(current_vix, 2))
    col2.metric("Avg Forecast Vol", round(np.mean(vol_forecast), 2))
    col3.metric("Model Used", model_type)

    # ----------------------------------------
    # SIGNAL
    # ----------------------------------------
    st.subheader("📌 Trading Signal")
    st.success(signal)

    # ----------------------------------------
    # PLOTS
    # ----------------------------------------
    st.plotly_chart(plot_vix(df), use_container_width=True)
    st.plotly_chart(plot_forecast(vol_forecast), use_container_width=True)

    # ----------------------------------------
    # TERM STRUCTURE TABLE
    # ----------------------------------------
    st.subheader("📈 Forecast Term Structure")

    term_df = pd.DataFrame({
        "Day": [f"T+{i}" for i in range(1,6)],
        "Forecast Vol": vol_forecast
    })

    st.dataframe(term_df)

    # ----------------------------------------
    # MODEL SUMMARY
    # ----------------------------------------
    with st.expander("📊 Model Summary"):
        st.text(res.summary())

# --------------------------------------------
# FOOTER
# --------------------------------------------
st.markdown("---")
st.caption("For educational purposes only. Not financial advice.")
