import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch import arch_model
import feedparser
from textblob import TextBlob
import datetime

# ==========================================
# PAGE CONFIG & BLOOMBERG-STYLE CSS
# ==========================================
st.set_page_config(page_title="Terminal | Volatility Forecaster", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Bloomberg Terminal Vibe */
    .stApp {
        background-color: #010101;
        color: #ffaa00;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3 {
        color: #ffaa00 !important;
    }
    .metric-container {
        background-color: #111111;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00ff00; /* Neon Green */
    }
    .metric-label {
        font-size: 14px;
        color: #aaaaaa;
    }
    .stButton>button {
        background-color: #000000;
        color: #00ff00;
        border: 1px solid #00ff00;
        font-family: 'Courier New', Courier, monospace;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00ff00;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA FETCHING MODULE
# ==========================================
def fetch_market_data():
    """Fetches ^INDIAVIX and ^NSEI (NIFTY 50) data."""
    try:
        vix_ticker = yf.Ticker("^INDIAVIX")
        nifty_ticker = yf.Ticker("^NSEI")
        
        vix_df = vix_ticker.history(period="2y")
        nifty_df = nifty_ticker.history(period="2y")
        
        if vix_df.empty or nifty_df.empty:
            raise ValueError("Empty dataframe returned from Yahoo Finance.")
            
        return vix_df, nifty_df
    except Exception as e:
        st.error(f"Market Data Error: {e}")
        return None, None

def fetch_financial_news():
    """Fetches and scores live financial news for India via Google News RSS."""
    try:
        url = "https://news.google.com/rss/search?q=India+financial+market+NIFTY+NSE&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        
        news_data = []
        sentiment_scores = []
        
        for entry in feed.entries[:15]:  # Top 15 headlines
            title = entry.title
            published = entry.published
            
            # Sentiment Analysis
            blob = TextBlob(title)
            score = blob.sentiment.polarity
            
            news_data.append({"Published": published, "Headline": title, "Sentiment Score": round(score, 3)})
            sentiment_scores.append(score)
            
        df_news = pd.DataFrame(news_data)
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        return df_news, avg_sentiment
    except Exception as e:
        st.error(f"News Fetching Error: {e}")
        return pd.DataFrame(), 0.0

# ==========================================
# MODELING MODULE
# ==========================================
def run_volatility_model(vix_df, nifty_df, avg_sentiment, forecast_horizon=5):
    """
    Fits an EGARCH model to VIX returns, extracts structural volatility trends,
    and applies a macro-sentiment adjustment to the multi-day forecast.
    """
    try:
        # Align data
        df = pd.concat([vix_df['Close'], nifty_df['Close']], axis=1).dropna()
        df.columns = ['VIX', 'NIFTY']
        
        # Calculate daily percentage returns
        df['VIX_Ret'] = df['VIX'].pct_change() * 100
        df['NIFTY_Ret'] = df['NIFTY'].pct_change() * 100
        df.dropna(inplace=True)
        
        # EGARCH Model (Asymmetric shocks: good for volatility)
        # We model the VIX returns. Note: Exogenous variables in multi-step forecasts 
        # require future values, so we use pure EGARCH for the forecast horizon.
        am = arch_model(df['VIX_Ret'], vol='EGARCH', p=1, o=1, q=1, dist='t')
        res = am.fit(disp='off', show_warning=False)
        
        # Generate Forecast
        forecasts = res.forecast(horizon=forecast_horizon)
        predicted_returns = forecasts.mean.iloc[-1].values  # Expected % change in VIX
        
        # Reconstruct forecasted VIX levels from predicted returns
        last_vix_level = df['VIX'].iloc[-1]
        projected_levels = []
        current_level = last_vix_level
        
        for ret in predicted_returns:
            current_level = current_level * (1 + (ret / 100))
            projected_levels.append(current_level)
            
        # SENTIMENT INTEGRATION
        # Negative Sentiment = Market Fear = Higher VIX. Positive Sentiment = Lower VIX.
        # We apply a heuristic shift based on the aggregated sentiment factor.
        sentiment_adjustment = -1 * (avg_sentiment * 2.5) # Scaling factor 
        
        adjusted_forecast = [max(1.0, val + sentiment_adjustment) for val in projected_levels]
        
        # Prepare Dates for Forecast
        last_date = df.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_horizon + 1)]
        
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted_VIX": adjusted_forecast
        })
        
        return df, forecast_df, res.summary()
    except Exception as e:
        st.error(f"Modeling Error: {e}")
        return None, None, None

# ==========================================
# UI & VISUALIZATION
# ==========================================
st.title("VOLATILITY FORECAST TERMINAL (VIX_IN)")

if st.button("RUN PIPELINE (FETCH & MODEL)"):
    with st.spinner("Fetching Market Data & Running EGARCH Pipeline..."):
        
        # 1. Fetch Data
        vix_df, nifty_df = fetch_market_data()
        news_df, avg_sentiment = fetch_financial_news()
        
        if vix_df is not None and nifty_df is not None:
            # 2. Run Model
            hist_df, forecast_df, model_summary = run_volatility_model(vix_df, nifty_df, avg_sentiment)
            
            if hist_df is not None:
                # Top Metrics Dashboard
                last_vix = hist_df['VIX'].iloc[-1]
                last_nifty = hist_df['NIFTY'].iloc[-1]
                vix_change = hist_df['VIX'].iloc[-1] - hist_df['VIX'].iloc[-2]
                nifty_change = hist_df['NIFTY'].iloc[-1] - hist_df['NIFTY'].iloc[-2]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">INDIA VIX (SPOT)</div>
                        <div class="metric-value" style="color: {'#ff3333' if vix_change > 0 else '#00ff00'};">
                            {last_vix:.2f} ({vix_change:+.2f})
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">NIFTY 50 (SPOT)</div>
                        <div class="metric-value" style="color: {'#00ff00' if nifty_change > 0 else '#ff3333'};">
                            {last_nifty:.2f} ({nifty_change:+.2f})
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    sentiment_color = "#00ff00" if avg_sentiment > 0.05 else "#ff3333" if avg_sentiment < -0.05 else "#aaaaaa"
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">MACRO SENTIMENT FACTOR</div>
                        <div class="metric-value" style="color: {sentiment_color};">
                            {avg_sentiment:+.3f}
                        </div>
                    </div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 3. Interactive Plotly Charts
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.1, 
                                    subplot_titles=("INDIA VIX & 5-Day EGARCH Forecast", "NIFTY 50 Spot Trend"))
                
                # Plot VIX Historical (Last 90 days for zoom)
                recent_hist = hist_df.tail(90)
                fig.add_trace(go.Scatter(x=recent_hist.index, y=recent_hist['VIX'], 
                                         mode='lines', name='Historical VIX', line=dict(color='#ffaa00', width=2)), row=1, col=1)
                
                # Plot VIX Forecast
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted_VIX'], 
                                         mode='lines+markers', name='Forecasted VIX (Sentiment Adj)', 
                                         line=dict(color='#00ff00', width=2, dash='dash')), row=1, col=1)
                
                # Plot NIFTY
                fig.add_trace(go.Scatter(x=recent_hist.index, y=recent_hist['NIFTY'], 
                                         mode='lines', name='NIFTY 50', line=dict(color='#00bbff', width=2)), row=2, col=1)
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111111",
                    plot_bgcolor="#000000",
                    height=600,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_xaxes(showgrid=True, gridcolor='#333333')
                fig.update_yaxes(showgrid=True, gridcolor='#333333')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. News & Sentiment Dataframe
                st.markdown("### 📰 MARKET HEADLINES & SENTIMENT DRIVERS")
                st.caption("Live headlines fetched via Google News RSS. NLP Sentiment scores (-1.0 to 1.0) are actively adjusting the baseline EGARCH Volatility Forecast.")
                if not news_df.empty:
                    # Apply custom styling to the dataframe for dark mode
                    st.dataframe(news_df.style.applymap(
                        lambda val: 'color: #00ff00' if val > 0 else 'color: #ff3333' if val < 0 else 'color: #aaaaaa',
                        subset=['Sentiment Score']
                    ), use_container_width=True)
                else:
                    st.warning("No news data retrieved.")
