import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from arch import arch_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(layout="wide", page_title="India VIX Volatility Terminal")

NEWS_API_KEY = "YOUR_NEWSAPI_KEY"

# -------------------------------
# STYLE (Bloomberg-like)
# -------------------------------
st.markdown("""
<style>
.stApp { background-color: #0b0f1a; color: #e1e5ed; }
h1, h2, h3 { color: #f5c542; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# DATA FETCH
# -------------------------------
@st.cache_data(ttl=300)
def fetch_data():
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
        st.error(f"Data Error: {e}")
        return None

# -------------------------------
# NEWS + SENTIMENT
# -------------------------------
def fetch_news():
    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
        res = requests.get(url).json()
        return [a['title'] for a in res['articles']]
    except:
        return []

def sentiment_score(headlines):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    return np.mean(scores) if scores else 0

# -------------------------------
# EGARCH (1-day)
# -------------------------------
def run_egarch(df):
    try:
        returns = df['VIX_RET'] * 100
        exog = df[['NIFTY_RET']] * 100

        model = arch_model(returns, vol='EGARCH', p=1, q=1, x=exog)
        res = model.fit(disp="off")

        fc = res.forecast(horizon=1, x=exog.iloc[-1:].values)
        var = fc.variance.values[-1][0]

        return np.sqrt(var)

    except Exception as e:
        st.warning(f"EGARCH Error: {e}")
        return None

# -------------------------------
# GARCH (5-day term structure)
# -------------------------------
def run_garch(df):
    try:
        returns = df['VIX_RET'] * 100
        exog = df[['NIFTY_RET']] * 100

        model = arch_model(returns, vol='GARCH', p=1, q=1, x=exog)
        res = model.fit(disp="off")

        fc = res.forecast(horizon=5, x=exog.iloc[-1:].values)
        var = fc.variance.values[-1]

        vol = np.sqrt(var)

        # Smooth curve
        decay = np.exp(-0.15 * np.arange(5))
        vol = vol * (1 + 0.1 * decay)

        return vol

    except Exception as e:
        st.error(f"GARCH Error: {e}")
        return None

# -------------------------------
# PLOTS
# -------------------------------
def plot_vix(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name="India VIX"))
    fig.update_layout(template="plotly_dark", title="India VIX")
    return fig

def plot_nifty(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['NIFTY'], name="NIFTY"))
    fig.update_layout(template="plotly_dark", title="NIFTY")
    return fig

def plot_term_structure(raw, adj):
    fig = go.Figure()
    x = [1,2,3,4,5]

    fig.add_trace(go.Scatter(x=x, y=raw, mode='lines+markers', name="Raw"))
    fig.add_trace(go.Scatter(x=x, y=adj, mode='lines+markers', name="Adjusted", line=dict(dash='dash')))

    fig.update_layout(template="plotly_dark", title="Volatility Term Structure")
    return fig

# -------------------------------
# UI
# -------------------------------
st.title("📊 India VIX Volatility Terminal")

if st.button("🚀 Run Analysis"):

    with st.spinner("Running full pipeline..."):

        df = fetch_data()

        if df is not None:

            # Sentiment
            headlines = fetch_news()
            sentiment = sentiment_score(headlines)

            st.subheader("🧠 Sentiment")
            st.metric("Score", round(sentiment, 3))

            # Models
            egarch_1d = run_egarch(df)
            garch_5d = run_garch(df)

            if egarch_1d is not None and garch_5d is not None:

                # Hybrid curve
                garch_5d[0] = egarch_1d

                adjusted = garch_5d * (1 + sentiment)

                # Table
                st.subheader("📈 Forecast")
                forecast_df = pd.DataFrame({
                    "Day": [f"T+{i}" for i in range(1,6)],
                    "Volatility": garch_5d,
                    "Sentiment Adjusted": adjusted
                })
                st.dataframe(forecast_df)

                # Charts
                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(plot_vix(df), use_container_width=True)

                with col2:
                    st.plotly_chart(plot_nifty(df), use_container_width=True)

                st.plotly_chart(plot_term_structure(garch_5d, adjusted), use_container_width=True)

            # News
            st.subheader("📰 Headlines Used in Sentiment")
            for h in headlines:
                st.markdown(f"- {h}")

else:
    st.info("Click 'Run Analysis'")
