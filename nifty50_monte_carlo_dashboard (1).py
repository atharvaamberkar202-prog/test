import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import ta
import datetime
from datetime import timedelta

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="Quant Dashboard: NIFTY 50 Predictor", layout="wide", page_icon="📈")

# Download NLTK data securely
@st.cache_resource
def load_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_vader()

TICKERS = {
    "NIFTY 50": "^NSEI",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "USD/INR": "INR=X"
}

# ==========================================
# MODULE 1: DATA PIPELINE
# ==========================================
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_market_data(start_date, end_date):
    data = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # Handle multi-index columns if yfinance returns them
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(ticker, level=1, axis=1)
                data[name] = df
        except Exception as e:
            st.warning(f"Could not fetch {name}: {e}")
    return data

@st.cache_data(ttl=3600)
def fetch_news_sentiment(ticker_symbol="^NSEI"):
    """Fetch latest news via RSS and compute sentiment."""
    # Using Yahoo Finance RSS for the specific ticker
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}&region=IN&lang=en-IN"
    feed = feedparser.parse(url)
    
    news_data = []
    for entry in feed.entries[:15]: # Get top 15 news
        score = sia.polarity_scores(entry.title)['compound']
        
        if score > 0.15: sentiment = "Bullish"
        elif score < -0.15: sentiment = "Bearish"
        else: sentiment = "Neutral"
            
        news_data.append({
            "Headline": entry.title,
            "Published": entry.published,
            "Score": round(score, 3),
            "Sentiment": sentiment
        })
    return pd.DataFrame(news_data)

# ==========================================
# MODULE 2: FEATURE ENGINEERING
# ==========================================
def engineer_features(nifty_df, sp500_df):
    df = nifty_df.copy()
    
    # 1. Technical Indicators
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['MA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['Vol_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) # Annualized
    
    # 2. Candle Classification (Threshold: 0.10% body)
    threshold = 0.001 
    body = (df['Close'] - df['Open']) / df['Open']
    
    conditions = [
        (body > threshold),
        (body < -threshold)
    ]
    choices = ['Up', 'Down']
    df['State'] = np.select(conditions, choices, default='Neutral')
    
    # Target Variable: Next Day's State (Lookahead Bias perfectly isolated here)
    df['Next_State'] = df['State'].shift(-1)
    
    # 3. Global Cues (S&P 500 T-1 Close influences NIFTY T Open)
    # Aligning dates: S&P500 date shifted by 1 day forward to match Nifty's "today"
    sp500_returns = np.log(sp500_df['Close'] / sp500_df['Close'].shift(1))
    sp500_returns.index = sp500_returns.index + pd.Timedelta(days=1)
    
    df['SP500_Overnight'] = sp500_returns
    df['SP500_Overnight'].fillna(method='ffill', inplace=True) # Carry forward Friday's US close to Monday's Nifty
    
    df['Global_Direction'] = np.where(df['SP500_Overnight'] > 0, 'Positive', 'Negative')
    
    df.dropna(inplace=True)
    return df

# ==========================================
# MODULE 3: MARKOV CHAIN MODEL
# ==========================================
def build_markov_model(df):
    """Builds a context-aware transition matrix."""
    # P(Next_State | Current_State, Global_Direction)
    transitions = pd.crosstab(
        [df['State'], df['Global_Direction']], 
        df['Next_State'], 
        normalize='index'
    )
    return transitions

def predict_next_candle(current_state, global_direction, transition_matrix, sentiment_bias):
    try:
        base_probs = transition_matrix.loc[(current_state, global_direction)].to_dict()
    except KeyError:
        # Fallback to equal probability if state combo never occurred (rare)
        base_probs = {'Up': 0.33, 'Down': 0.33, 'Neutral': 0.34}
        
    # Apply Sentiment Bias (Bayesian-inspired heuristic shift)
    # If news is strongly bullish, shift 5% probability from Down to Up
    if sentiment_bias > 0.2:
        base_probs['Up'] = min(1.0, base_probs.get('Up', 0) + 0.05)
        base_probs['Down'] = max(0.0, base_probs.get('Down', 0) - 0.05)
    elif sentiment_bias < -0.2:
        base_probs['Down'] = min(1.0, base_probs.get('Down', 0) + 0.05)
        base_probs['Up'] = max(0.0, base_probs.get('Up', 0) - 0.05)
        
    # Normalize
    total = sum(base_probs.values())
    return {k: v / total for k, v in base_probs.items()}

# ==========================================
# UI: STREAMLIT DASHBOARD
# ==========================================
def main():
    st.title("🏛️ Quant Dashboard: NIFTY 50 Markov Predictor")
    st.markdown("A probabilistic forecasting engine combining discrete-time Markov Chains, Global Market Cues, and NLP News Sentiment.")
    
    # Sidebar
    st.sidebar.header("Model Controls")
    if st.sidebar.button("🔄 Refresh Data & Recompute Model"):
        st.cache_data.clear()
        st.rerun()
        
    start_date = st.sidebar.date_input("Training Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("Training End Date", datetime.date.today())
    
    with st.spinner("Fetching global market data & calculating states..."):
        market_data = fetch_market_data(start_date, end_date)
        news_df = fetch_news_sentiment()
        
    if "NIFTY 50" not in market_data or "S&P 500" not in market_data:
        st.error("Failed to fetch core index data. Please check your internet connection or Yahoo Finance status.")
        return
        
    nifty_df = engineer_features(market_data["NIFTY 50"], market_data["S&P 500"])
    tm = build_markov_model(nifty_df)
    
    # Get Current Context
    latest_nifty = nifty_df.iloc[-1]
    current_state = latest_nifty['State']
    current_global = latest_nifty['Global_Direction']
    avg_sentiment = news_df['Score'].mean() if not news_df.empty else 0
    
    probs = predict_next_candle(current_state, current_global, tm, avg_sentiment)
    predicted_class = max(probs, key=probs.get)
    confidence = probs[predicted_class] * 100

    # UI: Top KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last NIFTY Close", f"₹{latest_nifty['Close']:,.2f}", f"{latest_nifty['Returns']*100:.2f}%")
    col2.metric("Today's State", current_state)
    col3.metric("Global Cue (S&P 500)", current_global)
    
    sentiment_color = "green" if avg_sentiment > 0.1 else "red" if avg_sentiment < -0.1 else "gray"
    col4.markdown(f"**News Sentiment:** <span style='color:{sentiment_color}; font-size:24px'><b>{avg_sentiment:.2f}</b></span>", unsafe_allow_html=True)
    
    st.divider()
    
    # UI: Prediction Banner
    pred_color = "#28a745" if predicted_class == 'Up' else "#dc3545" if predicted_class == 'Down' else "#ffc107"
    st.markdown(f"""
    <div style="background-color:{pred_color}; padding:20px; border-radius:10px; text-align:center; color:white;">
        <h2>Next Day Prediction: {predicted_class.upper()}</h2>
        <p style="font-size: 18px;">Model Confidence: {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # UI: Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Market Data & Probs", "🧠 Markov Details", "📰 Global Sentiment"])
    
    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            # Plotly Candlestick
            fig = go.Figure(data=[go.Candlestick(x=nifty_df.index[-100:],
                open=nifty_df['Open'].iloc[-100:],
                high=nifty_df['High'].iloc[-100:],
                low=nifty_df['Low'].iloc[-100:],
                close=nifty_df['Close'].iloc[-100:],
                name="NIFTY")])
            
            fig.add_trace(go.Scatter(x=nifty_df.index[-100:], y=nifty_df['MA_20'].iloc[-100:], line=dict(color='blue', width=1), name='20 MA'))
            fig.update_layout(title="NIFTY 50 - Last 100 Days", height=450, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame(list(probs.items()), columns=['State', 'Probability'])
            fig_pie = px.pie(prob_df, values='Probability', names='State', 
                             color='State', color_discrete_map={'Up':'#28a745', 'Down':'#dc3545', 'Neutral':'#ffc107'},
                             hole=0.4)
            fig_pie.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.metric("RSI (14)", f"{latest_nifty['RSI_14']:.2f}")
            st.metric("Annualized Volatility", f"{latest_nifty['Vol_20']*100:.2f}%")

    with tab2:
        st.subheader("Context-Aware Transition Matrix")
        st.markdown("This matrix shows $P(S_{t+1} | S_t, G_t)$. How likely the market is to move Up, Down, or Neutral tomorrow, given today's candle and last night's US market direction.")
        
        # Format the MultiIndex for Heatmap
        tm_reset = tm.reset_index()
        tm_reset['Context (State + Global)'] = tm_reset['State'] + " + " + tm_reset['Global_Direction']
        tm_reset.set_index('Context (State + Global)', inplace=True)
        tm_heat = tm_reset[['Up', 'Down', 'Neutral']]
        
        fig_heat = px.imshow(tm_heat, text_auto=".2%", aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.subheader("Latest Financial News & Sentiment Bias")
        if not news_df.empty:
            def highlight_sentiment(val):
                color = 'green' if val == 'Bullish' else 'red' if val == 'Bearish' else 'gray'
                return f'color: {color}'
            
            st.dataframe(news_df.style.applymap(highlight_sentiment, subset=['Sentiment']), use_container_width=True)
        else:
            st.write("No news data currently available.")
            
if __name__ == "__main__":
    main()
