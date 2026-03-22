import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from arch import arch_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(layout="wide", page_title="India VIX Volatility Dashboard")

NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # <-- Replace this

# -------------------------------
# STYLING (Bloomberg-like)
# -------------------------------
st.markdown("""
    <style>
    body { background-color: #0b0f1a; color: #e1e5ed; }
    .stApp { background-color: #0b0f1a; }
    h1, h2, h3 { color: #f5c542; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# DATA FETCHING
# -------------------------------
@st.cache_data(ttl=300)
def fetch_market_data():
    try:
        vix = yf.download("^INDIAVIX", period="6mo", interval="1d")
        nifty = yf.download("^NSEI", period="6mo", interval="1d")

        vix = vix[['Close']].rename(columns={'Close': 'VIX'})
        nifty = nifty[['Close']].rename(columns={'Close': 'NIFTY'})

        df = vix.join(nifty, how='inner')

        df['VIX_RET'] = np.log(df['VIX'] / df['VIX'].shift(1))
        df['NIFTY_RET'] = np.log(df['NIFTY'] / df['NIFTY'].shift(1))

        df.dropna(inplace=True)

        return df

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None


# -------------------------------
# NEWS + SENTIMENT
# -------------------------------
def fetch_news():
    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
        res = requests.get(url).json()

        headlines = [a['title'] for a in res['articles']]
        return headlines

    except Exception as e:
        st.warning(f"News fetch error: {e}")
        return []


def sentiment_score(headlines):
    analyzer = SentimentIntensityAnalyzer()

    scores = []
    for h in headlines:
        score = analyzer.polarity_scores(h)['compound']
        scores.append(score)

    if len(scores) == 0:
        return 0

    return np.mean(scores)


# -------------------------------
# EGARCH MODEL
# -------------------------------
def run_egarch(df, sentiment_factor=0):
    try:
        # Scale returns (important for stability)
        returns = df['VIX_RET'] * 100

        # Exogenous variable (NIFTY returns)
        exog = df[['NIFTY_RET']] * 100

        model = arch_model(
            returns,
            vol='EGARCH',
            p=1,
            q=1,
            x=exog
        )

        res = model.fit(disp="off")

        # Forecast 5 days
        forecasts = res.forecast(horizon=5, x=exog.iloc[-1:].values)

        vol_forecast = np.sqrt(forecasts.variance.values[-1])

        # Incorporate sentiment
        adjusted = vol_forecast * (1 + sentiment_factor)

        return vol_forecast, adjusted, res

    except Exception as e:
        st.error(f"Model error: {e}")
        return None, None, None


# -------------------------------
# PLOTTING
# -------------------------------
def plot_vix(df, forecast=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['VIX'],
        name="India VIX",
        line=dict(color='cyan')
    ))

    if forecast is not None:
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 6)]

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            name="Forecast",
            line=dict(color='yellow', dash='dash')
        ))

    fig.update_layout(
        template="plotly_dark",
        title="India VIX Forecast",
        height=500
    )

    return fig


def plot_nifty(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['NIFTY'],
        name="NIFTY",
        line=dict(color='orange')
    ))

    fig.update_layout(
        template="plotly_dark",
        title="NIFTY Trend",
        height=400
    )

    return fig


# -------------------------------
# UI
# -------------------------------
st.title("📊 India VIX Volatility Intelligence Dashboard")

if st.button("🚀 Run Analysis"):

    with st.spinner("Fetching data and running models..."):

        df = fetch_market_data()

        if df is not None:

            # NEWS + SENTIMENT
            headlines = fetch_news()
            sentiment = sentiment_score(headlines)

            st.subheader("🧠 Sentiment Factor")
            st.metric("News Sentiment Score", round(sentiment, 3))

            # MODEL
            forecast, adjusted, model = run_egarch(df, sentiment)

            if forecast is not None:

                st.subheader("📈 Volatility Forecast (Next 5 Days)")
                forecast_df = pd.DataFrame({
                    "Day": [f"T+{i}" for i in range(1, 6)],
                    "Raw Forecast": forecast,
                    "Sentiment Adjusted": adjusted
                })

                st.dataframe(forecast_df)

                # CHARTS
                st.plotly_chart(plot_vix(df, forecast), use_container_width=True)
                st.plotly_chart(plot_nifty(df), use_container_width=True)

            # NEWS DISPLAY
            st.subheader("📰 Global Financial Headlines (Used in Sentiment)")
            for h in headlines:
                st.markdown(f"- {h}")

else:
    st.info("Click 'Run Analysis' to start.")
