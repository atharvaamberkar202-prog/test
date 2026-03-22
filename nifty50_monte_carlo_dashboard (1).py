"""
NIFTY 50 Markov Chain Prediction Dashboard
==========================================
Production-grade Streamlit app for next-day NIFTY candle prediction
using Markov Chains, global market cues, and sentiment analysis.
"""

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NIFTY 50 · Markov Prediction Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

from data_collector   import fetch_all_data
from feature_engineer import engineer_features
from markov_model     import MarkovModel
from sentiment        import fetch_sentiment
from ui_components    import (
    apply_theme, render_header, render_global_cues,
    render_prediction_panel, render_transition_heatmap,
    render_price_chart, render_sentiment_box,
    render_indicators, render_correlation_matrix,
    render_footer,
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
apply_theme()

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "data_loaded"  not in st.session_state:
    st.session_state.data_loaded  = False
if "raw_data"     not in st.session_state:
    st.session_state.raw_data     = {}
if "features"     not in st.session_state:
    st.session_state.features     = None
if "model"        not in st.session_state:
    st.session_state.model        = None
if "sentiment"    not in st.session_state:
    st.session_state.sentiment    = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(neutral_threshold: float = 0.001):
    """Fetch data → engineer features → build model → fetch sentiment."""
    progress = st.progress(0, text="⏳ Fetching market data…")

    raw = fetch_all_data()
    st.session_state.raw_data = raw
    progress.progress(25, text="⚙️ Engineering features…")

    features = engineer_features(raw, neutral_threshold=neutral_threshold)
    st.session_state.features = features
    progress.progress(55, text="🔗 Building Markov model…")

    model = MarkovModel()
    model.fit(features)
    st.session_state.model = model
    progress.progress(75, text="📰 Fetching sentiment…")

    sentiment = fetch_sentiment()
    st.session_state.sentiment = sentiment
    progress.progress(100, text="✅ Done!")
    time.sleep(0.4)
    progress.empty()

    st.session_state.data_loaded  = True
    st.session_state.last_refresh = datetime.now()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    neutral_threshold = st.slider(
        "Neutral band (|return| < threshold → Neutral)",
        min_value=0.0005,
        max_value=0.005,
        value=0.001,
        step=0.0005,
        format="%.4f",
    )

    st.divider()
    refresh_btn = st.button("🔄 Refresh Model", use_container_width=True, type="primary")
    st.caption("Re-fetches data, recomputes features, reruns model & sentiment.")

    st.divider()
    st.markdown("### 📌 About")
    st.info(
        "**Markov Chain model** conditioned on:\n"
        "- Current NIFTY candle state\n"
        "- Global market direction\n"
        "- News sentiment bias\n\n"
        "_Probabilistic, not guaranteed._"
    )

    if st.session_state.last_refresh:
        st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%d %b %Y  %H:%M:%S')}")

# ─────────────────────────────────────────────────────────────────────────────
# Auto-load on first run or manual refresh
# ─────────────────────────────────────────────────────────────────────────────
if refresh_btn or not st.session_state.data_loaded:
    with st.spinner("Loading…"):
        try:
            run_pipeline(neutral_threshold)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.data_loaded:
    st.info("Click **Refresh Model** in the sidebar to load data.")
    st.stop()

features  = st.session_state.features
model     = st.session_state.model
sentiment = st.session_state.sentiment
raw_data  = st.session_state.raw_data

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
render_header()

# ── Row 1: Global cues ────────────────────────────────────────────────────────
render_global_cues(features, raw_data)

st.divider()

# ── Row 2: Prediction + heatmap ───────────────────────────────────────────────
col_pred, col_heat = st.columns([1, 1.4], gap="large")
with col_pred:
    render_prediction_panel(model, features, sentiment)
with col_heat:
    render_transition_heatmap(model)

st.divider()

# ── Row 3: Price chart ────────────────────────────────────────────────────────
render_price_chart(features)

st.divider()

# ── Row 4: Indicators + Sentiment ─────────────────────────────────────────────
col_ind, col_sent = st.columns([1.2, 1], gap="large")
with col_ind:
    render_indicators(features)
with col_sent:
    render_sentiment_box(sentiment)

st.divider()

# ── Row 5: Correlation matrix ────────────────────────────────────────────────
render_correlation_matrix(features)

render_footer()
