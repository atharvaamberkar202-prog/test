import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("NIFTY Monte Carlo Prediction Dashboard")

ticker = "^NSEI"

data = yf.download(ticker, period="5y")

# Fix for multi-index columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Use Adj Close if available otherwise Close
price_col = "Adj Close" if "Adj Close" in data.columns else "Close"

prices = data[price_col]

returns = np.log(prices / prices.shift(1)).dropna()

mu = returns.mean()
sigma = returns.std()

S0 = prices.iloc[-1]

simulations = 10000

Z = np.random.normal(size=simulations)

next_prices = S0 * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

mean_price = np.mean(next_prices)

prob_up = np.mean(next_prices > S0)

# =====================
# METRICS
# =====================

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(S0,2))
col2.metric("Expected Price", round(mean_price,2))
col3.metric("Probability Up", f"{prob_up*100:.2f}%")

# =====================
# HISTOGRAM
# =====================

fig = go.Figure()

fig.add_histogram(x=next_prices)

fig.update_layout(
    template="plotly_dark",
    title="Monte Carlo Distribution of Next Day Prices"
)

st.plotly_chart(fig, use_container_width=True)
