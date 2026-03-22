"""
India VIX GARCH Volatility Forecasting Dashboard
=================================================
Streamlit app that combines GARCH(1,1) modelling, NIFTY spot data,
and global news sentiment to forecast India VIX volatility.

Run with:  streamlit run india_vix_dashboard.py
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import datetime
import traceback
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import yfinance as yf
from arch import arch_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="India VIX · GARCH Volatility Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS – Bloomberg-terminal dark theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d0d0d;
        color: #e0e0e0;
        font-family: 'Courier New', monospace;
    }
    [data-testid="stSidebar"] {
        background-color: #111418;
        border-right: 1px solid #1f2937;
    }
    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"]  { color: #6b7280; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { color: #38bdf8; font-size: 1.55rem; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 0.82rem; }
    /* ── Headers ── */
    h1 { color: #38bdf8; letter-spacing: 2px; font-size: 1.6rem; }
    h2 { color: #93c5fd; font-size: 1.15rem; border-bottom: 1px solid #1e3a5f; padding-bottom: 4px; }
    h3 { color: #7dd3fc; font-size: 0.98rem; }
    /* ── Primary button ── */
    .stButton > button {
        background: linear-gradient(135deg, #0369a1, #0ea5e9);
        color: #fff;
        border: none;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 10px 28px;
        font-size: 0.95rem;
    }
    .stButton > button:hover { opacity: 0.85; }
    /* ── Alerts / news ── */
    .news-card {
        background: #111827;
        border-left: 3px solid #0ea5e9;
        border-radius: 4px;
        padding: 8px 14px;
        margin-bottom: 8px;
        font-size: 0.82rem;
    }
    .bullish  { border-left-color: #22c55e; }
    .bearish  { border-left-color: #ef4444; }
    .neutral  { border-left-color: #6b7280; }
    .tag {
        display: inline-block;
        padding: 1px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 6px;
    }
    .tag-bullish { background:#166534; color:#4ade80; }
    .tag-bearish { background:#7f1d1d; color:#fca5a5; }
    .tag-neutral { background:#1f2937; color:#9ca3af; }
    /* ── Scrollable news box ── */
    .news-box { max-height: 420px; overflow-y: auto; padding-right: 4px; }
    /* ── Dividers ── */
    hr { border-color: #1e3a5f; }
    /* ── Expanders ── */
    [data-testid="stExpander"] { background: #111827; border: 1px solid #1e3a5f; border-radius: 6px; }
    /* ── Plotly figs get transparent bg ── */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# ── 1. DATA FETCHING
# ─────────────────────────────────────────────

def fetch_india_vix(period: str = "2y") -> pd.DataFrame:
    """Fetch India VIX OHLC data from yfinance."""
    ticker = yf.Ticker("^INDIAVIX")
    df = ticker.history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError("India VIX data returned empty. Check your internet connection.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Close"]].rename(columns={"Close": "VIX"}).dropna()
    df["VIX_Returns"] = np.log(df["VIX"] / df["VIX"].shift(1))
    return df.dropna()


def fetch_nifty(period: str = "2y") -> pd.DataFrame:
    """Fetch NIFTY 50 spot price data."""
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError("NIFTY data returned empty.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Close"]].rename(columns={"Close": "NIFTY"}).dropna()
    df["NIFTY_Returns"] = np.log(df["NIFTY"] / df["NIFTY"].shift(1))
    return df.dropna()


# ─────────────────────────────────────────────
# ── 2. NEWS & SENTIMENT
# ─────────────────────────────────────────────

_NEWS_QUERIES = [
    "India stock market",
    "NIFTY 50 outlook",
    "RBI monetary policy",
    "global financial markets",
    "US Federal Reserve interest rates",
    "crude oil prices India",
]

def fetch_news_headlines(api_key: Optional[str], n_headlines: int = 30) -> list[dict]:
    """
    Attempt to fetch headlines via NewsAPI.
    Falls back to curated placeholder headlines if key absent / call fails.
    """
    if api_key:
        try:
            query = " OR ".join(_NEWS_QUERIES[:3])
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q={requests.utils.quote(query)}"
                f"&language=en&sortBy=publishedAt&pageSize={n_headlines}"
                f"&apiKey={api_key}"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            articles = r.json().get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "url": a.get("url", "#"),
                    "publishedAt": a.get("publishedAt", ""),
                }
                for a in articles
                if a.get("title") and "[Removed]" not in a.get("title", "")
            ][:n_headlines]
        except Exception as e:
            st.warning(f"NewsAPI call failed ({e}). Using fallback headlines.")

    # ── Fallback curated headlines (no API key needed) ──
    fallback = [
        {"title": "RBI holds repo rate steady, signals cautious stance on inflation", "source": "Economic Times", "url": "#", "publishedAt": ""},
        {"title": "FII outflows from Indian equities surge amid global risk-off", "source": "Mint", "url": "#", "publishedAt": ""},
        {"title": "US Fed signals two rate cuts in 2025, boosting emerging market sentiment", "source": "Bloomberg", "url": "#", "publishedAt": ""},
        {"title": "Crude oil climbs 2% on Middle East tensions, adding to inflation fears", "source": "Reuters", "url": "#", "publishedAt": ""},
        {"title": "NIFTY 50 hits resistance at 22,800; technical analysts urge caution", "source": "Business Standard", "url": "#", "publishedAt": ""},
        {"title": "India GDP growth forecast revised upward to 7.1% by IMF", "source": "IMF", "url": "#", "publishedAt": ""},
        {"title": "Global markets rally on easing US-China trade tensions", "source": "Financial Times", "url": "#", "publishedAt": ""},
        {"title": "India VIX spikes above 14 as options buyers hedge election risk", "source": "NSE Insights", "url": "#", "publishedAt": ""},
        {"title": "Bank Nifty underperforms as NPA concerns resurface in PSU banks", "source": "Moneycontrol", "url": "#", "publishedAt": ""},
        {"title": "Rupee weakens past 84 mark amid dollar strength and trade deficit data", "source": "LiveMint", "url": "#", "publishedAt": ""},
        {"title": "Sensex soars 600 points as IT sector leads broad-based rally", "source": "NDTV Profit", "url": "#", "publishedAt": ""},
        {"title": "Inflation in India eases to 4.8%, within RBI comfort zone", "source": "Economic Times", "url": "#", "publishedAt": ""},
        {"title": "US tech stocks drag global equities lower on earnings disappointment", "source": "WSJ", "url": "#", "publishedAt": ""},
        {"title": "Foreign reserves rise to record high, signalling RBI intervention", "source": "Hindu BusinessLine", "url": "#", "publishedAt": ""},
        {"title": "SEBI tightens F&O rules; volatility may rise short-term", "source": "Moneycontrol", "url": "#", "publishedAt": ""},
    ]
    return fallback[:n_headlines]


def analyse_sentiment(headlines: list[dict]) -> Tuple[list[dict], float]:
    """Run VADER sentiment on each headline; return enriched list + composite score."""
    analyser = SentimentIntensityAnalyzer()
    scores = []
    enriched = []
    for h in headlines:
        vs = analyser.polarity_scores(h["title"])
        compound = vs["compound"]          # –1 (bearish) → +1 (bullish)
        scores.append(compound)
        sentiment_label = (
            "Bullish" if compound >= 0.05
            else "Bearish" if compound <= -0.05
            else "Neutral"
        )
        enriched.append({**h, "compound": compound, "sentiment": sentiment_label})

    composite = float(np.mean(scores)) if scores else 0.0
    return enriched, composite


# ─────────────────────────────────────────────
# ── 3. GARCH MODELLING
# ─────────────────────────────────────────────

def build_garch_forecast(
    vix_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    sentiment_score: float,
    forecast_horizon: int = 5,
    use_nifty_exog: bool = True,
) -> Tuple[pd.DataFrame, dict, object]:
    """
    Fit GARCH(1,1) on VIX log-returns (scaled to %).
    Optionally includes NIFTY returns as exogenous regressor.
    Incorporates sentiment as a mean-equation regressor.

    Returns
    -------
    forecast_df : DataFrame with forecast dates + predicted annualised vol
    metrics     : dict of model diagnostics
    res         : fitted arch ModelResult
    """
    # ── Align indices ──
    combined = vix_df[["VIX", "VIX_Returns"]].join(
        nifty_df[["NIFTY", "NIFTY_Returns"]], how="inner"
    ).dropna()

    returns = combined["VIX_Returns"] * 100   # scale to %

    # ── Exogenous regressors for mean equation ──
    exog = None
    if use_nifty_exog:
        nifty_ret = combined["NIFTY_Returns"] * 100
        # Sentiment broadcasted as a constant series (daily factor)
        sentiment_col = pd.Series(sentiment_score * 10, index=returns.index, name="Sentiment")
        exog = pd.concat([nifty_ret.rename("NIFTY_ret"), sentiment_col], axis=1)

    # ── Fit GARCH(1,1) ──
    am = arch_model(
        returns,
        x=exog,
        vol="GARCH",
        p=1,
        q=1,
        mean="ARX",
        dist="skewt",   # skewed-t for fat tails in VIX returns
        rescale=False,
    )
    res = am.fit(
        disp="off",
        show_warning=False,
        options={"maxiter": 500},
    )

    # ── Forecast ──
    # arch requires a 3-D array of shape (n_steps, horizon, n_regressors)
    # when there are multiple exogenous regressors.
    # We use the last observed values held constant across the forecast horizon.
    if use_nifty_exog and exog is not None:
        n_regressors = exog.shape[1]
        last_row = exog.iloc[-1].values          # shape: (n_regressors,)
        # Build (1, forecast_horizon, n_regressors) — one prediction step,
        # repeated across the horizon
        fcast_exog = np.tile(last_row, (1, forecast_horizon, 1))
        # shape is now (1, forecast_horizon, n_regressors) ✓
        forecasts = res.forecast(horizon=forecast_horizon, x=fcast_exog, reindex=False)
    else:
        forecasts = res.forecast(horizon=forecast_horizon, reindex=False)

    # Variance forecasts → annualised vol (GARCH outputs % returns variance)
    var_fc = forecasts.variance.values[-1]        # shape: (horizon,)
    vol_fc = np.sqrt(var_fc * 252) / 100          # annualised, back to decimal

    # ── Build forecast dates (business days) ──
    last_date = combined.index[-1]
    fc_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    forecast_df = pd.DataFrame({
        "Date": fc_dates,
        "Forecasted_Vol_Annualised": vol_fc,
        "Forecasted_VIX_Equiv": vol_fc * 100,     # approximate VIX-equivalent
    })

    # ── Diagnostics ──
    cond_vol_last = res.conditional_volatility.iloc[-1]
    metrics = {
        "Log-Likelihood": round(res.loglikelihood, 2),
        "AIC": round(res.aic, 2),
        "BIC": round(res.bic, 2),
        "alpha (ARCH)": round(res.params.get("alpha[1]", float("nan")), 4),
        "beta (GARCH)": round(res.params.get("beta[1]", float("nan")), 4),
        "Persistence": round(
            res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0), 4
        ),
        "Last Cond. Vol (%)": round(float(cond_vol_last), 4),
        "Sentiment Score": round(sentiment_score, 4),
    }

    return forecast_df, metrics, res


# ─────────────────────────────────────────────
# ── 4. PLOTTING HELPERS
# ─────────────────────────────────────────────

_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.95)",
    font=dict(family="Courier New", size=11, color="#c9d1d9"),
    xaxis=dict(gridcolor="#1e3a5f", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e3a5f", showgrid=True, zeroline=False),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f", borderwidth=1),
    hovermode="x unified",
)


def plot_vix_history(vix_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vix_df.index, y=vix_df["VIX"],
        mode="lines", name="India VIX",
        line=dict(color="#38bdf8", width=1.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
    ))
    # Danger zone band
    fig.add_hrect(y0=20, y1=vix_df["VIX"].max() * 1.05,
                  fillcolor="rgba(239,68,68,0.06)", line_width=0)
    fig.add_hline(y=20, line_dash="dot", line_color="#ef4444",
                  annotation_text="Stress Level (20)", annotation_position="top left",
                  annotation_font_color="#ef4444")
    fig.update_layout(**_PLOTLY_LAYOUT, title="📊 India VIX — Historical Close")
    fig.update_yaxes(title_text="VIX Level")
    return fig


def plot_nifty(nifty_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nifty_df.index, y=nifty_df["NIFTY"],
        mode="lines", name="NIFTY 50",
        line=dict(color="#4ade80", width=1.5),
        fill="tozeroy", fillcolor="rgba(74,222,128,0.07)",
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, title="📈 NIFTY 50 — Spot Price Trend")
    fig.update_yaxes(title_text="Index Level")
    return fig


def plot_garch_forecast(
    vix_df: pd.DataFrame,
    res,
    forecast_df: pd.DataFrame,
) -> go.Figure:
    """Combined chart: historical VIX + conditional vol + forecast."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
        subplot_titles=("India VIX Level + Forecast", "GARCH Conditional Volatility (annualised %)"),
    )

    # ── Panel 1: Historical VIX ──
    fig.add_trace(go.Scatter(
        x=vix_df.index, y=vix_df["VIX"],
        name="India VIX",
        line=dict(color="#38bdf8", width=1.5),
    ), row=1, col=1)

    # ── Forecast band ──
    fc_vol = forecast_df["Forecasted_VIX_Equiv"].values
    fc_dates = forecast_df["Date"].values
    upper = fc_vol * 1.10
    lower = fc_vol * 0.90

    fig.add_trace(go.Scatter(
        x=list(fc_dates) + list(fc_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(251,191,36,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±10% Confidence Band",
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_vol,
        mode="lines+markers",
        name="Forecasted VIX-Equiv",
        line=dict(color="#fbbf24", width=2.5, dash="dash"),
        marker=dict(size=8, color="#fbbf24"),
    ), row=1, col=1)

    # Vertical line at last historical date
    last_date = str(vix_df.index[-1].date())
    fig.add_vline(x=last_date, line_dash="dot", line_color="#6b7280",
                  annotation_text="Today", annotation_position="top right",
                  row=1, col=1)

    # ── Panel 2: Conditional Volatility ──
    cond_vol = res.conditional_volatility * np.sqrt(252)   # annualise
    fig.add_trace(go.Scatter(
        x=vix_df.index[-len(cond_vol):], y=cond_vol,
        name="GARCH Cond. Vol",
        line=dict(color="#c084fc", width=1.5),
        fill="tozeroy", fillcolor="rgba(192,132,252,0.08)",
    ), row=2, col=1)

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title="🔮 GARCH(1,1) Forecast — India VIX",
        height=580,
    )
    fig.update_yaxes(title_text="VIX Level", row=1, col=1, gridcolor="#1e3a5f")
    fig.update_yaxes(title_text="Ann. Vol (%)", row=2, col=1, gridcolor="#1e3a5f")
    return fig


def plot_sentiment_gauge(score: float) -> go.Figure:
    """Render a gauge for composite sentiment score."""
    color = "#22c55e" if score > 0.05 else "#ef4444" if score < -0.05 else "#6b7280"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 0, "valueformat": ".3f"},
        number={"font": {"color": color, "size": 28}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": "#6b7280"},
            "bar": {"color": color},
            "bgcolor": "#0d1117",
            "bordercolor": "#1e3a5f",
            "steps": [
                {"range": [-1, -0.05], "color": "rgba(239,68,68,0.15)"},
                {"range": [-0.05, 0.05], "color": "rgba(107,114,128,0.10)"},
                {"range": [0.05, 1], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
        title={"text": "News Sentiment Score", "font": {"color": "#93c5fd", "size": 13}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        margin=dict(l=20, r=20, t=40, b=10),
        height=200,
    )
    return fig


def plot_returns_histogram(vix_df: pd.DataFrame) -> go.Figure:
    ret = vix_df["VIX_Returns"].dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ret, nbinsx=60, name="VIX Log-Returns",
        marker_color="#38bdf8", opacity=0.75,
    ))
    # Normal overlay
    mu, sigma = ret.mean(), ret.std()
    x_range = np.linspace(ret.min(), ret.max(), 200)
    from scipy.stats import norm
    pdf = norm.pdf(x_range, mu, sigma) * len(ret) * (ret.max() - ret.min()) / 60
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf, mode="lines",
        name="Normal Fit", line=dict(color="#fbbf24", width=2),
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, title="📉 VIX Log-Return Distribution",
                      showlegend=True, height=300)
    fig.update_xaxes(title_text="Log-Return (%)")
    fig.update_yaxes(title_text="Count")
    return fig


# ─────────────────────────────────────────────
# ── 5. STREAMLIT APP
# ─────────────────────────────────────────────

def render_metric_row(vix_df, nifty_df, forecast_df, sentiment_score):
    col1, col2, col3, col4, col5 = st.columns(5)
    last_vix = vix_df["VIX"].iloc[-1]
    prev_vix = vix_df["VIX"].iloc[-2]
    last_nifty = nifty_df["NIFTY"].iloc[-1]
    prev_nifty = nifty_df["NIFTY"].iloc[-2]
    next_fc = forecast_df["Forecasted_VIX_Equiv"].iloc[0]
    fc5_mean = forecast_df["Forecasted_VIX_Equiv"].mean()

    with col1:
        st.metric("India VIX (Last)", f"{last_vix:.2f}",
                  delta=f"{last_vix - prev_vix:+.2f}")
    with col2:
        st.metric("NIFTY 50 (Last)", f"{last_nifty:,.0f}",
                  delta=f"{last_nifty - prev_nifty:+.0f}")
    with col3:
        st.metric("VIX Forecast D+1", f"{next_fc:.2f}",
                  delta=f"{next_fc - last_vix:+.2f}")
    with col4:
        st.metric("Avg Forecast (5D)", f"{fc5_mean:.2f}",
                  delta=f"{fc5_mean - last_vix:+.2f}")
    with col5:
        label = "Bullish 🟢" if sentiment_score > 0.05 else "Bearish 🔴" if sentiment_score < -0.05 else "Neutral ⚪"
        st.metric("Sentiment", label, delta=f"{sentiment_score:+.3f}")


def render_news(enriched_headlines: list[dict]):
    st.markdown("### 🗞️ Global Financial Headlines · Sentiment Analysis Input")
    st.markdown(
        "_These headlines were fetched in real-time and processed by VADER "
        "to compute the composite sentiment score used in the GARCH model._",
        unsafe_allow_html=False,
    )

    bull = sum(1 for h in enriched_headlines if h["sentiment"] == "Bullish")
    bear = sum(1 for h in enriched_headlines if h["sentiment"] == "Bearish")
    neut = len(enriched_headlines) - bull - bear

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Bullish", bull)
    c2.metric("🔴 Bearish", bear)
    c3.metric("⚪ Neutral", neut)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="news-box">', unsafe_allow_html=True)
    for h in enriched_headlines:
        s = h["sentiment"]
        css_cls = "bullish" if s == "Bullish" else "bearish" if s == "Bearish" else "neutral"
        tag_cls = f"tag-{css_cls}"
        date_str = h.get("publishedAt", "")[:10] or "—"
        url = h.get("url", "#")
        link = f'<a href="{url}" target="_blank" style="color:#38bdf8;text-decoration:none;">{h["title"]}</a>'
        st.markdown(
            f'<div class="news-card {css_cls}">'
            f'{link}'
            f'<span class="tag {tag_cls}">{s}</span>'
            f'<br><span style="color:#4b5563;font-size:0.7rem;">{h["source"]} · {date_str} · '
            f'compound: {h["compound"]:+.3f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_metrics(metrics: dict):
    st.markdown("### 🔬 GARCH Model Diagnostics")
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, v)


def render_forecast_table(forecast_df: pd.DataFrame):
    st.markdown("### 📋 Volatility Forecast Table")
    display = forecast_df.copy()
    display["Date"] = display["Date"].dt.strftime("%a %d %b %Y")
    display["Forecasted_Vol_Annualised"] = display["Forecasted_Vol_Annualised"].map(
        lambda x: f"{x*100:.2f}%"
    )
    display["Forecasted_VIX_Equiv"] = display["Forecasted_VIX_Equiv"].map(
        lambda x: f"{x:.2f}"
    )
    display.columns = ["Date", "Annualised Vol", "VIX-Equivalent"]
    st.dataframe(display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ── SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    st.sidebar.markdown("## ⚙️ Configuration")
    st.sidebar.markdown("---")

    cfg = {}
    cfg["period"] = st.sidebar.selectbox(
        "Historical Data Window",
        ["6mo", "1y", "2y", "5y"],
        index=2,
        help="How far back to fetch India VIX and NIFTY data.",
    )
    cfg["forecast_horizon"] = st.sidebar.slider(
        "Forecast Horizon (Days)", min_value=1, max_value=10, value=5
    )
    cfg["use_nifty_exog"] = st.sidebar.checkbox(
        "Use NIFTY Returns as Exogenous Variable", value=True
    )
    cfg["n_headlines"] = st.sidebar.slider(
        "Number of News Headlines", min_value=5, max_value=50, value=20, step=5
    )
    cfg["newsapi_key"] = st.sidebar.text_input(
        "NewsAPI Key (optional)",
        type="password",
        help="Get a free key at https://newsapi.org — leave blank to use curated fallback headlines.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model**: GARCH(1,1) · ARX mean · Skewed-t errors\n\n"
        "**Sentiment**: VADER compound score\n\n"
        "**Data**: yfinance (India VIX + NIFTY)\n\n"
        "**News**: NewsAPI / Fallback"
    )
    return cfg


# ─────────────────────────────────────────────
# ── MAIN
# ─────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown(
        "<h1>📉 INDIA VIX · GARCH VOLATILITY FORECASTER</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Real-time volatility forecasting powered by **GARCH(1,1)** · "
        "NIFTY 50 exogenous signals · Global news sentiment analysis",
        unsafe_allow_html=False,
    )
    st.markdown("---")

    cfg = render_sidebar()

    # ── Session state defaults ──
    for key in ("vix_df", "nifty_df", "forecast_df", "metrics", "res",
                "headlines", "sentiment_score", "ran"):
        if key not in st.session_state:
            st.session_state[key] = None
    if "ran" not in st.session_state:
        st.session_state["ran"] = False

    # ── Run Analysis button ──
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run = st.button("▶ Run Analysis", use_container_width=True)

    if run:
        with st.spinner("🔄 Fetching data & running GARCH model …"):
            errors = []

            # 1. VIX
            try:
                vix_df = fetch_india_vix(cfg["period"])
                st.session_state["vix_df"] = vix_df
            except Exception as e:
                errors.append(f"VIX fetch error: {e}")
                vix_df = None

            # 2. NIFTY
            try:
                nifty_df = fetch_nifty(cfg["period"])
                st.session_state["nifty_df"] = nifty_df
            except Exception as e:
                errors.append(f"NIFTY fetch error: {e}")
                nifty_df = None

            # 3. News & Sentiment
            try:
                raw_headlines = fetch_news_headlines(
                    cfg.get("newsapi_key") or None, cfg["n_headlines"]
                )
                enriched, sentiment_score = analyse_sentiment(raw_headlines)
                st.session_state["headlines"] = enriched
                st.session_state["sentiment_score"] = sentiment_score
            except Exception as e:
                errors.append(f"News/sentiment error: {e}")
                enriched, sentiment_score = [], 0.0
                st.session_state["headlines"] = []
                st.session_state["sentiment_score"] = 0.0

            # 4. GARCH model
            if vix_df is not None and nifty_df is not None:
                try:
                    forecast_df, metrics, res = build_garch_forecast(
                        vix_df, nifty_df, sentiment_score,
                        forecast_horizon=cfg["forecast_horizon"],
                        use_nifty_exog=cfg["use_nifty_exog"],
                    )
                    st.session_state["forecast_df"] = forecast_df
                    st.session_state["metrics"] = metrics
                    st.session_state["res"] = res
                except Exception as e:
                    errors.append(f"GARCH model error: {e}\n{traceback.format_exc()}")

            st.session_state["ran"] = True

            if errors:
                for err in errors:
                    st.error(err)
            else:
                st.success("✅ Analysis complete.")

    # ── Render results if available ──
    if st.session_state.get("ran") and st.session_state.get("vix_df") is not None:
        vix_df     = st.session_state["vix_df"]
        nifty_df   = st.session_state["nifty_df"]
        forecast_df= st.session_state["forecast_df"]
        metrics    = st.session_state["metrics"]
        res        = st.session_state["res"]
        headlines  = st.session_state["headlines"]
        sentiment_score = st.session_state["sentiment_score"]

        if forecast_df is None:
            st.warning("Model did not produce forecasts. Check the errors above.")
            return

        # ── KPI Row ──
        st.markdown("### 📊 Key Metrics")
        render_metric_row(vix_df, nifty_df, forecast_df, sentiment_score)
        st.markdown("---")

        # ── GARCH forecast chart ──
        st.plotly_chart(
            plot_garch_forecast(vix_df, res, forecast_df),
            use_container_width=True,
        )

        # ── Forecast table + sentiment gauge side-by-side ──
        left, right = st.columns([3, 2])
        with left:
            render_forecast_table(forecast_df)
        with right:
            st.plotly_chart(plot_sentiment_gauge(sentiment_score), use_container_width=True)

        st.markdown("---")

        # ── History charts ──
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_vix_history(vix_df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_nifty(nifty_df), use_container_width=True)

        st.plotly_chart(plot_returns_histogram(vix_df), use_container_width=True)

        # ── Model diagnostics ──
        render_model_metrics(metrics)
        st.markdown("---")

        # ── Model summary expandable ──
        with st.expander("📄 Full GARCH Model Summary"):
            st.text(str(res.summary()))

        # ── News section ──
        st.markdown("---")
        if headlines:
            render_news(headlines)
        else:
            st.info("No headlines available.")

        # ── Footer ──
        st.markdown("---")
        st.markdown(
            f"<p style='color:#4b5563;font-size:0.75rem;text-align:center;'>"
            f"Last updated: {datetime.datetime.now().strftime('%d %b %Y %H:%M:%S IST')} · "
            f"Data: yfinance (NSE) · Model: GARCH(1,1) ARX Skewed-t · "
            f"Sentiment: VADER NLP"
            f"</p>",
            unsafe_allow_html=True,
        )

    elif not st.session_state.get("ran"):
        st.info("👈 Configure settings in the sidebar, then click **▶ Run Analysis** to start.")


if __name__ == "__main__":
    main()
