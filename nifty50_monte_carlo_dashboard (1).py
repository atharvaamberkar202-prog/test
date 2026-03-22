# ============================================
# INDIA VIX PROP MODEL (PRODUCTION SAFE)
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
# USER INPUT
# -----------------------------------
NEWS_API_KEY = st.text_input("Enter News API Key", type="password")

ROLLING_WINDOW = 120

# -----------------------------------
# SAFE DATA FETCH
# -----------------------------------
def fetch_data(ticker):
    for _ in range(3):
        df = yf.download(ticker, period="1y", progress=False)
        if not df.empty:
            return df
    return pd.DataFrame()

def safe_close(df, name):
    if df is None or df.empty:
        st.warning(f"{name} data missing")
        return None

    if "Close" in df.columns:
        return df["Close"]
    elif "Adj Close" in df.columns:
        return df["Adj Close"]
    else:
        st.warning(f"{name} has no Close column")
        return None

# -----------------------------------
# MAIN BUTTON
# -----------------------------------
if st.button("🚀 Run Prop Model"):

    with st.spinner("Fetching market data..."):

        # Fetch all data
        vix_india = fetch_data("^INDIAVIX")
        nifty = fetch_data("^NSEI")
        us_vix = fetch_data("^VIX")
        dxy = fetch_data("DX=F")       # more stable proxy
        bonds = fetch_data("^TNX")

        # Extract close safely
        series_list = []

        def add_series(df, name):
            s = safe_close(df, name)
            if s is not None:
                series_list.append(s.rename(name))

        add_series(vix_india, "INDIA_VIX")
        add_series(nifty, "NIFTY")
        add_series(us_vix, "US_VIX")
        add_series(dxy, "DXY")
        add_series(bonds, "BOND")

        if len(series_list) < 3:
            st.error("Not enough data to run model")
            st.stop()

        # ALIGN DATA
        df = pd.concat(series_list, axis=1)
        df = df.dropna()

        if len(df) < ROLLING_WINDOW + 10:
            st.error("Not enough aligned data")
            st.stop()

        # -----------------------------------
        # RETURNS
        # -----------------------------------
        df["vix_ret"] = np.log(df["INDIA_VIX"] / df["INDIA_VIX"].shift(1))
        df["nifty_ret"] = np.log(df["NIFTY"] / df["NIFTY"].shift(1))
        
        if "US_VIX" in df:
            df["usvix_ret"] = np.log(df["US_VIX"] / df["US_VIX"].shift(1))
        else:
            df["usvix_ret"] = 0

        if "DXY" in df:
            df["dxy_ret"] = np.log(df["DXY"] / df["DXY"].shift(1))
        else:
            df["dxy_ret"] = 0

        if "BOND" in df:
            df["bond_ret"] = np.log(df["BOND"] / df["BOND"].shift(1))
        else:
            df["bond_ret"] = 0

        df = df.dropna()

        # -----------------------------------
        # SENTIMENT
        # -----------------------------------
        analyzer = SentimentIntensityAnalyzer()

        def fetch_news():
            if not NEWS_API_KEY:
                return [], 0

            try:
                url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
                r = requests.get(url)
                data = r.json()

                headlines = [a["title"] for a in data.get("articles", [])]

                scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
                avg = np.mean(scores) if scores else 0

                return headlines, avg

            except:
                return [], 0

        headlines, avg_sentiment = fetch_news()

        # -----------------------------------
        # EGARCH ROLLING
        # -----------------------------------
        returns = df["vix_ret"] * 100
        rolling_vol = []

        for i in range(ROLLING_WINDOW, len(returns)):
            train = returns.iloc[i-ROLLING_WINDOW:i]

            try:
                model = arch_model(train, vol="EGARCH", p=1, o=1, q=1)
                res = model.fit(disp="off")
                forecast = res.forecast(horizon=1)
                vol = np.sqrt(forecast.variance.values[-1][0])
            except:
                vol = np.nan

            rolling_vol.append(vol)

        df = df.iloc[ROLLING_WINDOW:]
        df["EGARCH_VOL"] = rolling_vol
        df = df.dropna()

        if df.empty:
            st.error("Model failed to generate output")
            st.stop()

        # -----------------------------------
        # LATEST VALUES
        # -----------------------------------
        latest = df.iloc[-1]

        base_vol = latest["EGARCH_VOL"]
        adj = base_vol

        # Factor adjustments
        adj += -2.5 * latest["nifty_ret"]
        adj += 1.8 * latest["usvix_ret"]
        adj += 1.2 * latest["dxy_ret"]
        adj += 1.0 * latest["bond_ret"]
        adj += -1.5 * avg_sentiment

        predicted_vix = latest["INDIA_VIX"] * (1 + adj / 100)

        # -----------------------------------
        # DISPLAY
        # -----------------------------------
        st.subheader("📌 Forecast")

        c1, c2, c3 = st.columns(3)
        c1.metric("Current VIX", round(latest["INDIA_VIX"], 2))
        c2.metric("EGARCH Vol", round(base_vol, 2))
        c3.metric("Predicted VIX", round(predicted_vix, 2))

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
            name="Prediction"
        ))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # FACTORS
        # -----------------------------------
        st.subheader("🧠 Factor Breakdown")

        st.write("NIFTY:", round(-2.5 * latest["nifty_ret"], 4))
        st.write("US VIX:", round(1.8 * latest["usvix_ret"], 4))
        st.write("DXY:", round(1.2 * latest["dxy_ret"], 4))
        st.write("BONDS:", round(1.0 * latest["bond_ret"], 4))
        st.write("Sentiment:", round(-1.5 * avg_sentiment, 4))

        # -----------------------------------
        # NEWS
        # -----------------------------------
        if headlines:
            st.subheader("📰 News Sentiment")

            for h in headlines:
                score = analyzer.polarity_scores(h)["compound"]
                st.write(f"{h} → {round(score,3)}")

            st.write("Avg Sentiment:", round(avg_sentiment, 3))
            
