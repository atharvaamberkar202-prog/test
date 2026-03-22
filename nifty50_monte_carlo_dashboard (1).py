# ============================================
# INDIA VIX PROP MODEL DASHBOARD
# EGARCH + MULTI FACTOR + ROLLING WINDOW
# ============================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from arch import arch_model
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide")
st.title("📊 India VIX Prop Model (EGARCH + Multi-Factor + Sentiment)")

# -----------------------------------
# PARAMETERS
# -----------------------------------
ROLLING_WINDOW = 120  # days
NEWS_API_KEY = "1852d6efa58d42c0b7b9b8e7aabbd0e0"

# -----------------------------------
# BUTTON
# -----------------------------------
if st.button("🚀 Run Prop Model Pipeline"):

    with st.spinner("Fetching global market data..."):

        # -----------------------------------
        # FETCH DATA
        # -----------------------------------
        vix_india = yf.download("^INDIAVIX", period="1y")
        nifty = yf.download("^NSEI", period="1y")
        us_vix = yf.download("^VIX", period="1y")
        dxy = yf.download("DX-Y.NYB", period="1y")   # Dollar Index
        bonds = yf.download("^TNX", period="1y")     # US 10Y Yield

        df = pd.DataFrame({
            "INDIA_VIX": vix_india["Close"],
            "NIFTY": nifty["Close"],
            "US_VIX": us_vix["Close"],
            "DXY": dxy["Close"],
            "BOND": bonds["Close"]
        }).dropna()

        # -----------------------------------
        # RETURNS
        # -----------------------------------
        df["vix_ret"] = np.log(df["INDIA_VIX"] / df["INDIA_VIX"].shift(1))
        df["nifty_ret"] = np.log(df["NIFTY"] / df["NIFTY"].shift(1))
        df["usvix_ret"] = np.log(df["US_VIX"] / df["US_VIX"].shift(1))
        df["dxy_ret"] = np.log(df["DXY"] / df["DXY"].shift(1))
        df["bond_ret"] = np.log(df["BOND"] / df["BOND"].shift(1))

        df.dropna(inplace=True)

        # -----------------------------------
        # SENTIMENT
        # -----------------------------------
        analyzer = SentimentIntensityAnalyzer()

        def fetch_news():
            url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
            r = requests.get(url)
            data = r.json()
            return [a["title"] for a in data["articles"]]

        headlines = fetch_news()

        sentiment_scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        avg_sentiment = np.mean(sentiment_scores)

        # -----------------------------------
        # ROLLING EGARCH MODEL
        # -----------------------------------
        returns = df["vix_ret"] * 100

        rolling_forecasts = []

        for i in range(ROLLING_WINDOW, len(returns)):
            train = returns[i-ROLLING_WINDOW:i]

            model = arch_model(
                train,
                vol="EGARCH",
                p=1,
                q=1,
                o=1,  # asymmetry term
                dist="normal"
            )

            res = model.fit(disp="off")
            forecast = res.forecast(horizon=1)

            vol = np.sqrt(forecast.variance.values[-1][0])
            rolling_forecasts.append(vol)

        df = df.iloc[ROLLING_WINDOW:]
        df["EGARCH_VOL"] = rolling_forecasts

        # -----------------------------------
        # LATEST VALUES
        # -----------------------------------
        latest = df.iloc[-1]

        base_vol = latest["EGARCH_VOL"]

        # -----------------------------------
        # MULTI-FACTOR ADJUSTMENT
        # -----------------------------------
        adj = base_vol

        # NIFTY (inverse)
        adj += -2.5 * latest["nifty_ret"]

        # US VIX (global fear spillover)
        adj += 1.8 * latest["usvix_ret"]

        # Dollar strength (risk-off proxy)
        adj += 1.2 * latest["dxy_ret"]

        # Bond yields (macro stress)
        adj += 1.0 * latest["bond_ret"]

        # Sentiment (forward signal)
        adj += -1.5 * avg_sentiment

        predicted_vix = latest["INDIA_VIX"] * (1 + adj / 100)

        # -----------------------------------
        # OUTPUT
        # -----------------------------------
        st.subheader("📌 Model Output")

        col1, col2, col3 = st.columns(3)
        col1.metric("Current India VIX", round(latest["INDIA_VIX"], 2))
        col2.metric("EGARCH Vol", round(base_vol, 2))
        col3.metric("Predicted VIX", round(predicted_vix, 2))

        # -----------------------------------
        # PLOT
        # -----------------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["INDIA_VIX"],
            name="India VIX"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["EGARCH_VOL"],
            name="EGARCH Vol"
        ))

        fig.add_trace(go.Scatter(
            x=[df.index[-1]],
            y=[predicted_vix],
            mode="markers",
            name="Prediction",
            marker=dict(size=12)
        ))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # FACTOR BREAKDOWN
        # -----------------------------------
        st.subheader("🧠 Factor Contributions")

        st.write(f"NIFTY Effect: {-2.5 * latest['nifty_ret']:.4f}")
        st.write(f"US VIX Effect: {1.8 * latest['usvix_ret']:.4f}")
        st.write(f"DXY Effect: {1.2 * latest['dxy_ret']:.4f}")
        st.write(f"Bond Yield Effect: {1.0 * latest['bond_ret']:.4f}")
        st.write(f"Sentiment Effect: {-1.5 * avg_sentiment:.4f}")

        # -----------------------------------
        # NEWS DISPLAY
        # -----------------------------------
        st.subheader("📰 News Driving Sentiment")

        for h, s in zip(headlines, sentiment_scores):
            st.write(f"{h} → {round(s,3)}")

        st.write("### Avg Sentiment:", round(avg_sentiment, 3))
