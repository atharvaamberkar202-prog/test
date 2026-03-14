import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("NIFTY Monte Carlo Prediction Dashboard")

ticker = "^NSEI"

data = yf.download(ticker, period="5y")

returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1)).dropna()

mu = returns.mean()
sigma = returns.std()

S0 = data["Adj Close"].iloc[-1]

simulations = 10000

Z = np.random.normal(size=simulations)

next_prices = S0 * np.exp((mu - 0.5*sigma**2) + sigma*Z)

mean_price = np.mean(next_prices)

prob_up = np.mean(next_prices > S0)

col1,col2,col3 = st.columns(3)

col1.metric("Current Price", round(S0,2))
col2.metric("Expected Price", round(mean_price,2))
col3.metric("Probability Up", f"{prob_up*100:.2f}%")

fig = go.Figure()

fig.add_histogram(x=next_prices)

st.plotly_chart(fig,use_container_width=True)