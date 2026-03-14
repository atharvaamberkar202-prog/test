"""
=============================================================================
  NIFTY 50 — Next-Day Monte Carlo Prediction Dashboard
  Streamlit App · Bloomberg Terminal Style · Dark Theme

  Run:
      streamlit run nifty50_monte_carlo_dashboard.py

  requirements.txt:
      streamlit
      yfinance
      pandas
      numpy
      scipy
      requests
      plotly
      vaderSentiment
      textblob
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, math, warnings, datetime
import numpy  as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── scipy KDE (optional, numpy fallback if missing) ──────────────────────────
try:
    from scipy.stats import gaussian_kde
    SCIPY_OK = True
except ImportError:
    SCIPY_OK     = False
    gaussian_kde = None

# ── Sentiment libraries (VADER > TextBlob > keyword) ─────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False

try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except ImportError:
    TEXTBLOB_OK = False

# ── yfinance ──────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    st.error("yfinance not installed. Add `yfinance` to requirements.txt")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be the FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "NIFTY 50 · Monte Carlo Dashboard",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #050e1c; }
  section[data-testid="stSidebar"] { background-color: #071428; }
  #MainMenu, footer, header { visibility: hidden; }

  .section-hdr {
    font-size: 11px; font-weight: 700; letter-spacing: 2.5px;
    color: #3a7bd5; text-transform: uppercase;
    border-bottom: 1px solid #1a3050;
    padding-bottom: 6px; margin: 28px 0 14px;
  }

  .kpi-card {
    background: #0a1628; border: 1px solid #1a2f4a;
    border-radius: 8px; padding: 16px 18px; min-height: 90px;
  }
  .kpi-label {
    font-size: 10px; color: #4a6a8a; text-transform: uppercase;
    letter-spacing: 1.2px; margin-bottom: 4px;
  }
  .kpi-value { font-size: 21px; font-weight: 700;
               font-variant-numeric: tabular-nums; }
  .kpi-sub   { font-size: 11px; color: #4a7a9a; margin-top: 3px; }

  .prob-card {
    background: #0a1628; border: 1px solid #1a2f4a;
    border-radius: 10px; padding: 24px 20px; text-align: center;
  }
  .prob-icon  { font-size: 30px; }
  .prob-title { font-size: 11px; color: #5a7a9a; text-transform: uppercase;
                letter-spacing: 1px; margin: 8px 0; }
  .prob-val   { font-size: 38px; font-weight: 700;
                font-variant-numeric: tabular-nums; }
  .prob-bar-bg {
    height: 7px; background: #1e2a3a; border-radius: 4px; margin-top: 10px;
  }

  .interp {
    background: #071428; border: 1px solid #1a3a5a;
    border-left: 4px solid #3a7bd5; border-radius: 6px;
    padding: 18px 22px; font-size: 13px; line-height: 1.85;
    color: #a0bcd8;
  }

  .disclaimer {
    text-align: center; font-size: 11px; color: #2a4a6a;
    border-top: 1px solid #0f1e30; padding-top: 16px; margin-top: 28px;
  }

  /* Streamlit native metric override */
  [data-testid="metric-container"] {
    background: #0a1628 !important;
    border: 1px solid #1a2f4a !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
  }
  [data-testid="metric-container"] label {
    color: #4a6a8a !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1px;
  }
  [data-testid="stMetricValue"] {
    color: #e0ecff !important; font-size: 20px !important;
    font-weight: 700 !important;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_TICKER  = "^NSEI"

# ── SGX / GIFT Nifty — ticker candidates tried in order ──────────────────────
# GIFT Nifty (formerly SGX Nifty) trades on NSE IFSC / SGX.
# yfinance exposes NSE monthly futures; we try current & next two months then
# fall back to S&P 500 / NASDAQ E-mini futures as globally-traded proxies.
def _sgx_ticker_candidates() -> list[str]:
    cands = []
    today = datetime.date.today()
    for delta in [0, 1, 2]:
        y   = today.year + (today.month - 1 + delta) // 12
        m   = (today.month - 1 + delta) % 12 + 1
        mon = datetime.date(y, m, 1).strftime("%b").upper()
        yr2 = str(y)[2:]
        cands.append(f"NIFTY{yr2}{mon}FUT.NS")    # e.g. NIFTY26MARFUT.NS
    cands += ["ES=F", "NQ=F"]                      # global futures fallback
    return cands

# ── Weekend-aware news sentiment weights (annualised %) ──────────────────────
# On weekends 2 days of news accumulate with no intraday price release valve,
# so sentiment historically has stronger predictive power for Monday's open.
NEWS_WEIGHT_WEEKDAY = 0.010   # baseline        → max ±1.0% p.a. contribution
NEWS_WEIGHT_FRIDAY  = 0.018   # pre-weekend gap → max ±1.8%
NEWS_WEIGHT_WEEKEND = 0.030   # 2-day news pile → max ±3.0%

GREEN  = "#00e5b0"
RED    = "#ff4560"
AMBER  = "#f0c040"
BLUE   = "#00c8ff"
PURPLE = "#7b61ff"
CARD   = "#0a1628"
GRID   = "#162033"

GLOBAL_INDICES = {
    "S&P 500"            : "^GSPC",
    "NASDAQ 100"         : "^NDX",
    "Dow Jones"          : "^DJI",
    "FTSE 100"           : "^FTSE",
    "DAX"                : "^GDAXI",
    "Nikkei 225"         : "^N225",
    "Hang Seng"          : "^HSI",
    "Shanghai Composite" : "000001.SS",
}

MOCK_HEADLINES = [
    "Asian markets slip as Fed signals higher-for-longer rate stance",
    "India GDP growth beats estimates; RBI holds rates steady",
    "Oil prices surge on OPEC production cut announcement",
    "US inflation data shows sticky core CPI; equities under pressure",
    "China stimulus hopes boost emerging market sentiment",
    "Tech stocks rally as NVIDIA earnings exceed forecasts",
    "Global bond yields rise amid debt ceiling uncertainty",
    "Foreign institutional investors net buyers in Indian equities",
    "Rupee strengthens against dollar on positive trade balance data",
    "IMF upgrades India growth forecast for current fiscal year",
]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    lookback = st.selectbox("Lookback period",
                            [3, 5, 7], index=1,
                            format_func=lambda x: f"{x} years")
    n_sims = st.selectbox("Monte Carlo paths",
                          [1_000, 5_000, 10_000, 20_000], index=2,
                          format_func=lambda x: f"{x:,} paths")
    news_key = st.text_input(
        "NewsAPI Key (optional)",
        value=os.getenv("NEWS_API_KEY", ""),
        type="password",
        help="Free key from newsapi.org — leave blank to use mock headlines")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

    st.markdown("---")
    st.markdown("""
**Sources**
- NIFTY 50 + 8 global indices via `yfinance`
- SGX / GIFT Nifty futures (NSE or proxy)
- Headlines via NewsAPI (mock fallback)

**Drift Formula**
```
adj_mu = mu + alpha/252
alpha  = GF×3% + SGX×w% + NS×nw%
```

**Weekend Weighting**
- Mon–Thu: news `nw=1.0%`, SGX `w=1.5%`
- Friday:  news `nw=1.8%`, SGX `w=2.5%`
- Weekend: news `nw=3.0%`, SGX `w=4.0%`

**Sentiment**
- VADER → TextBlob → Keyword
""")
    st.caption("⚠️ Educational use only. Not investment advice.")


# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE  (cached 1 hour)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(lookback_yr: int) -> dict:
    end   = datetime.date.today().strftime("%Y-%m-%d")
    start = (datetime.date.today() -
             datetime.timedelta(days=365 * lookback_yr + 30)).strftime("%Y-%m-%d")

    def _dl(ticker: str) -> pd.Series:
        """
        Download a single ticker and return a clean named pd.Series of
        daily Close prices.  Handles both yfinance ≤0.1.x (flat columns)
        and ≥0.2.x (MultiIndex columns like ('Close', '^NSEI')).
        """
        try:
            df = yf.download(
                ticker,
                start=start, end=end,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,   # request flat columns when possible
            )
        except TypeError:
            # older yfinance that doesn't accept multi_level_index kwarg
            try:
                df = yf.download(ticker, start=start, end=end,
                                 auto_adjust=True, progress=False)
            except Exception:
                return pd.Series(dtype=float, name=ticker)
        except Exception:
            return pd.Series(dtype=float, name=ticker)

        if df is None or df.empty:
            return pd.Series(dtype=float, name=ticker)

        # ── Flatten MultiIndex columns if present ─────────────────────────────
        if isinstance(df.columns, pd.MultiIndex):
            # columns look like: [('Close','^NSEI'), ('Open','^NSEI'), ...]
            df.columns = [c[0] if isinstance(c, tuple) else c
                          for c in df.columns]

        if "Close" not in df.columns:
            # Last-resort: take the first numeric column
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                return pd.Series(dtype=float, name=ticker)
            col = numeric_cols[0]
        else:
            col = "Close"

        s = df[col].copy()
        # squeeze() can still return a DataFrame if there are duplicate cols
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

        s = s.squeeze()
        if not isinstance(s, pd.Series):
            return pd.Series(dtype=float, name=ticker)

        s = s.dropna()
        s.name = ticker          # always force the ticker as the series name
        s.index = pd.to_datetime(s.index)
        return s

    nifty = _dl(NIFTY_TICKER)
    if nifty.empty:
        return {}

    global_raw = {nm: _dl(tk) for nm, tk in GLOBAL_INDICES.items()}

    # ── Align on a shared date index using name-based concatenation ───────────
    # Key fix: use the series name as the column key, not s.name after concat
    valid_globals = {nm: s for nm, s in global_raw.items() if not s.empty}

    all_series = {"__NIFTY__": nifty}
    for nm, s in valid_globals.items():
        all_series[nm] = s          # key = friendly name, not ticker

    combined = pd.concat(all_series.values(), axis=1, keys=all_series.keys())
    combined.index = pd.to_datetime(combined.index)
    combined.sort_index(inplace=True)

    nifty_a = combined["__NIFTY__"].dropna()

    global_a = {}
    for nm in valid_globals:
        if nm in combined.columns:
            g = combined[nm].dropna()
            if not g.empty:
                global_a[nm] = g

    def log_ret(s: pd.Series) -> pd.Series:
        return np.log(s / s.shift(1)).dropna()

    return {
        "nifty"          : nifty_a,
        "nifty_returns"  : log_ret(nifty_a),
        "global_closes"  : global_a,
        "global_returns" : {k: log_ret(v) for k, v in global_a.items()},
        "last_date"      : str(nifty_a.index[-1].date()),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def get_sentiment(api_key: str) -> dict:
    headlines = MOCK_HEADLINES[:]
    if api_key:
        try:
            url = (
                "https://newsapi.org/v2/everything"
                "?q=stock+market+india+nifty+economy+finance"
                "&sortBy=publishedAt&language=en&pageSize=20"
                f"&apiKey={api_key}"
            )
            r    = requests.get(url, timeout=8)
            r.raise_for_status()
            arts = r.json().get("articles", [])
            hl   = [a["title"] for a in arts if a.get("title")]
            if hl:
                headlines = hl
        except Exception:
            pass

    scores = []
    method = "Keyword"
    if VADER_OK:
        sia    = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(h)["compound"] for h in headlines]
        method = "VADER"
    elif TEXTBLOB_OK:
        scores = [TextBlob(h).sentiment.polarity for h in headlines]
        method = "TextBlob"
    else:
        POS = {"bull","surges","rally","gains","upgrade","positive","beats",
               "strong","growth","rise","buyers","stimulus","optimism","boom"}
        NEG = {"bear","slips","falls","decline","cut","downgrade","pressure",
               "inflation","risk","fear","sticky","uncertainty","recession",
               "selloff","plunge"}
        for h in headlines:
            w = set(h.lower().split())
            p, n = len(w & POS), len(w & NEG)
            scores.append((p - n) / (p + n or 1))

    agg = float(np.clip(np.mean(scores) if scores else 0, -1, 1))
    return {"score": agg, "headlines": headlines[:10],
            "method": method, "n": len(scores)}


@st.cache_data(ttl=900, show_spinner=False)   # 15-min cache — futures move fast
def fetch_sgx_nifty(nifty_spot: float) -> dict:
    """
    Fetch the most recent GIFT / SGX Nifty futures price and compute:
      - futures price & source ticker
      - premium / discount vs NIFTY spot (%)
      - directional signal in [-1, +1]

    Tries NSE monthly futures tickers first, then falls back to S&P 500
    E-mini futures (ES=F) rescaled as a proxy directional signal.
    """
    def _get_close_series(df: pd.DataFrame) -> pd.Series:
        """Flatten any MultiIndex columns and return the Close series."""
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c
                          for c in df.columns]
        col = "Close" if "Close" in df.columns else df.select_dtypes("number").columns[0]
        s   = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.squeeze().dropna()

    candidates = _sgx_ticker_candidates()
    two_days   = (datetime.date.today() -
                  datetime.timedelta(days=6)).strftime("%Y-%m-%d")
    tomorrow   = (datetime.date.today() +
                  datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in candidates:
        try:
            try:
                df = yf.download(ticker, start=two_days, end=tomorrow,
                                 auto_adjust=True, progress=False,
                                 interval="1d", multi_level_index=False)
            except TypeError:
                df = yf.download(ticker, start=two_days, end=tomorrow,
                                 auto_adjust=True, progress=False,
                                 interval="1d")

            close_s = _get_close_series(df)
            if len(close_s) < 1:
                continue
            fut_price = float(close_s.iloc[-1])
            if fut_price <= 0:
                continue

            is_proxy   = ticker in ("ES=F", "NQ=F")
            source_lbl = f"GIFT/NSE Futures ({ticker})"

            if is_proxy:
                # For global futures proxies we use their 1-day return to
                # infer an equivalent NIFTY premium/discount.
                if len(close_s) >= 2:
                    proxy_ret = float(close_s.iloc[-1] / close_s.iloc[-2]) - 1.0
                else:
                    proxy_ret = 0.0
                premium_pct = proxy_ret * 100
                source_lbl  = f"Global Futures Proxy ({ticker})"
            else:
                premium_pct = (fut_price / nifty_spot - 1.0) * 100

            # Signal: cap premium at ±2% → normalise to [-1, +1]
            signal = float(np.clip(premium_pct / 2.0, -1.0, 1.0))

            return {
                "price"      : round(fut_price, 2),
                "premium_pct": round(premium_pct, 3),
                "signal"     : signal,
                "ticker"     : ticker,
                "source"     : source_lbl,
                "available"  : True,
            }
        except Exception:
            continue

    # Hard fallback — no SGX signal available
    return {
        "price"      : nifty_spot,
        "premium_pct": 0.0,
        "signal"     : 0.0,
        "ticker"     : "N/A",
        "source"     : "Unavailable",
        "available"  : False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOAD  &  COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("📡 Fetching market data …"):
    data = fetch_prices(lookback)

if not data:
    st.error("Failed to fetch NIFTY 50 data. Check your internet connection.")
    st.stop()

with st.spinner("📰 Analysing news sentiment …"):
    sentiment = get_sentiment(news_key)

# ── Base NIFTY price ──────────────────────────────────────────────────────────
S0 = float(data["nifty"].iloc[-1])

# ── SGX / GIFT Nifty signal ───────────────────────────────────────────────────
with st.spinner("🌐 Fetching SGX / GIFT Nifty futures …"):
    sgx = fetch_sgx_nifty(S0)

# ── Weekend detection ─────────────────────────────────────────────────────────
_today      = datetime.date.today()
_weekday    = _today.weekday()          # 0=Mon … 4=Fri, 5=Sat, 6=Sun
IS_WEEKEND  = _weekday >= 5
IS_FRIDAY   = _weekday == 4

if IS_WEEKEND:
    session_label  = "🗓️ WEEKEND — Enhanced News & SGX Weighting Active"
    session_colour = AMBER
    news_weight    = NEWS_WEIGHT_WEEKEND   # 3× — 2-day news accumulation
    sgx_weight_ann = 0.04   # SGX is the primary price discovery on weekends
elif IS_FRIDAY:
    session_label  = "📅 FRIDAY — Pre-Weekend Elevated Weighting"
    session_colour = "#f0a040"
    news_weight    = NEWS_WEIGHT_FRIDAY    # 1.8× — gap risk
    sgx_weight_ann = 0.025
else:
    session_label  = f"📈 WEEKDAY ({_today.strftime('%A')})"
    session_colour = GREEN
    news_weight    = NEWS_WEIGHT_WEEKDAY   # baseline
    sgx_weight_ann = 0.015

# ── Global signal ─────────────────────────────────────────────────────────────
prev_returns = {}
for nm, ret in data["global_returns"].items():
    if len(ret) >= 2:
        prev_returns[nm] = float(ret.iloc[-1])

if prev_returns:
    raw_gf = np.mean(list(prev_returns.values()))
    std_gf = np.std(list(prev_returns.values())) or 1e-9
    gf     = float(np.clip(raw_gf / std_gf, -3, 3))
else:
    gf = 0.0

# ── GBM parameters ────────────────────────────────────────────────────────────
rets   = data["nifty_returns"].dropna()
mu     = float(rets.mean())
sigma  = float(rets.std())
ns     = sentiment["score"]
sgx_s  = sgx["signal"]   # in [-1, +1]

# ── Quant-grade signal composition ───────────────────────────────────────────
#
#   Each term contributes a bounded annualised alpha converted to daily:
#
#   Signal          Weight (ann%)   Rationale
#   ─────────────── ─────────────── ─────────────────────────────────────────
#   Global indices  gf/3 × 3 %     Cross-sectional Z-score of 8 indices
#   SGX/GIFT Nifty  sgx × sgx_w%   Futures premium/discount vs spot
#   News sentiment  ns  × news_w%  Weekend: 3×, Friday: 1.8×, Weekday: 1×
#
gf_unit   = gf / 3.0                               # Z-score → [-1, +1]
alpha_ann = (gf_unit * 0.03
             + sgx_s  * sgx_weight_ann
             + ns     * news_weight)               # total annualised alpha
alpha_day = alpha_ann / 252                        # convert to daily
adj_mu    = mu + alpha_day

ann_vol    = sigma  * math.sqrt(252) * 100
ann_mu     = mu     * 252 * 100
ann_adjmu  = adj_mu * 252 * 100
drift_adj  = alpha_ann * 100   # display as annualised %

# ── Component breakdown for display ──────────────────────────────────────────
gf_contrib  = gf_unit * 0.03 * 100        # ann %
sgx_contrib = sgx_s   * sgx_weight_ann * 100
ns_contrib  = ns      * news_weight    * 100


# ── Monte Carlo ───────────────────────────────────────────────────────────────
rng    = np.random.default_rng(42)
Z      = rng.standard_normal(n_sims)
S_next = S0 * np.exp((adj_mu - 0.5 * sigma**2) + sigma * Z)

# ── Statistics ────────────────────────────────────────────────────────────────
pcts = {
    "P1"    : float(np.percentile(S_next,  1)),
    "P5"    : float(np.percentile(S_next,  5)),
    "P25"   : float(np.percentile(S_next, 25)),
    "Median": float(np.median(S_next)),
    "Mean"  : float(np.mean(S_next)),
    "P75"   : float(np.percentile(S_next, 75)),
    "P95"   : float(np.percentile(S_next, 95)),
    "P99"   : float(np.percentile(S_next, 99)),
}
chgs = {k: (v / S0 - 1) * 100 for k, v in pcts.items()}
probs = {
    "up"       : float(np.mean(S_next > S0)         * 100),
    "down"     : float(np.mean(S_next < S0)         * 100),
    "above_p05": float(np.mean(S_next > S0 * 1.005) * 100),
    "below_m05": float(np.mean(S_next < S0 * 0.995) * 100),
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def val_colour(v: float) -> str:
    return GREEN if v >= 0 else RED

def kpi(label: str, value: str, sub: str = "", color: str = "#e0ecff"):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color};">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def prob_card(icon: str, title: str, value: float, invert: bool = False):
    bar_col = GREEN if (value >= 50) != invert else RED
    val_col = GREEN if (value >= 50) != invert else RED
    fill    = f"width:{value:.1f}%;background:{bar_col};height:100%;border-radius:4px;"
    st.markdown(f"""
    <div class="prob-card">
      <div class="prob-icon">{icon}</div>
      <div class="prob-title">{title}</div>
      <div class="prob-val" style="color:{val_col};">{value:.1f}%</div>
      <div class="prob-bar-bg">
        <div style="{fill}"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def section_hdr(letter: str, title: str):
    st.markdown(
        f'<div class="section-hdr">{letter} &nbsp;—&nbsp; {title}</div>',
        unsafe_allow_html=True)

def base_layout(title: str, h: int = 360) -> dict:
    return dict(
        title       = dict(text=title, font=dict(color="#e0e6f0", size=13)),
        template    = "plotly_dark",
        paper_bgcolor = CARD,
        plot_bgcolor  = "#0d1f35",
        height      = h,
        margin      = dict(l=52, r=18, t=44, b=42),
        font        = dict(color="#7a8fa6"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_paths():
    rng2  = np.random.default_rng(99)
    dt    = 1 / 252
    Z_fan = rng2.standard_normal((100, 5))
    fig   = go.Figure()
    for i in range(100):
        path = [S0]
        for z in Z_fan[i]:
            path.append(path[-1] * np.exp(
                (adj_mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z))
        col = GREEN if path[-1] >= S0 else RED
        fig.add_trace(go.Scatter(
            x=list(range(6)), y=path, mode="lines",
            line=dict(color=col, width=0.6 if i > 0 else 1.8),
            opacity=0.14 if i > 0 else 0.9,
            showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=S0, line_dash="dash", line_color=AMBER, line_width=1.5,
                  annotation_text=f"  Current: {S0:,.2f}",
                  annotation_font_color=AMBER)
    fig.update_layout(**base_layout("Monte Carlo Sample Paths (100 shown)"),
        xaxis=dict(title="Days Ahead", gridcolor=GRID, tickvals=[0,1,2,3,4,5]),
        yaxis=dict(title="NIFTY Level", gridcolor=GRID, tickformat=",.0f"))
    return fig


def build_histogram():
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=S_next, nbinsx=120,
        marker_color=GREEN, opacity=0.75, name="Simulated Prices"))
    for lbl, val, col in [
        ("P5",  pcts["P5"],    RED),
        ("P25", pcts["P25"],   AMBER),
        ("Med", pcts["Median"],BLUE),
        ("P75", pcts["P75"],   AMBER),
        ("P95", pcts["P95"],   RED),
    ]:
        fig.add_vline(x=val, line_color=col, line_dash="dot", line_width=1.5,
                      annotation_text=f" {lbl}", annotation_font_color=col,
                      annotation_font_size=11)
    fig.add_vline(x=S0, line_color="#ffffff", line_dash="dash", line_width=2,
                  annotation_text="  Current", annotation_font_color="#ffffff")
    fig.update_layout(**base_layout("Distribution of Simulated Next-Day Prices"),
        xaxis=dict(title="Price Level", gridcolor=GRID, tickformat=",.0f"),
        yaxis=dict(title="Frequency",   gridcolor=GRID),
        bargap=0.02)
    return fig


def build_confidence():
    fig = go.Figure()
    for lo_k, hi_k, fill, nm in [
        ("P1",  "P99", "rgba(255,69,96,0.10)",  "P1–P99"),
        ("P5",  "P95", "rgba(240,192,64,0.14)", "P5–P95"),
        ("P25", "P75", "rgba(0,200,255,0.20)",  "IQR P25–P75"),
    ]:
        lo_v, hi_v = pcts[lo_k], pcts[hi_k]
        fig.add_trace(go.Scatter(
            x=[0,1,1,0], y=[S0, hi_v, lo_v, S0],
            fill="toself", fillcolor=fill,
            line=dict(width=0), name=nm, mode="lines"))
    for lbl, val, col in [
        ("P5",   pcts["P5"],    RED),  ("P25",  pcts["P25"],   AMBER),
        ("Med",  pcts["Median"],BLUE), ("Mean", pcts["Mean"],  PURPLE),
        ("P75",  pcts["P75"],   AMBER),("P95",  pcts["P95"],   RED),
    ]:
        fig.add_trace(go.Scatter(
            x=[0,1], y=[S0, val], mode="lines+markers",
            line=dict(color=col, width=1.4, dash="dot"),
            marker=dict(size=7, color=col), name=lbl))
    fig.update_layout(**base_layout("Confidence Interval Bands (Today → Tomorrow)"),
        xaxis=dict(tickvals=[0,1], ticktext=["Today","Next Day"], gridcolor=GRID),
        yaxis=dict(title="NIFTY Level", gridcolor=GRID, tickformat=",.0f"),
        legend=dict(bgcolor=CARD, font=dict(color="#a0aec0", size=11)))
    return fig


def build_kde():
    x_arr = np.linspace(S_next.min(), S_next.max(), 500)
    if SCIPY_OK and gaussian_kde is not None:
        _k   = gaussian_kde(S_next)
        y    = _k(x_arr)
        y0   = float(_k(np.array([S0]))[0])
    else:
        cnt, edges = np.histogram(S_next, bins=200, density=True)
        ctr = (edges[:-1] + edges[1:]) / 2
        y   = np.interp(x_arr, ctr, cnt)
        y0  = float(np.interp(S0, ctr, cnt))

    fig  = go.Figure()
    m_up = x_arr >= S0
    m_dn = x_arr <  S0
    fig.add_trace(go.Scatter(
        x=np.concatenate([[S0], x_arr[m_up]]),
        y=np.concatenate([[y0],     y[m_up]]),
        fill="tozeroy", fillcolor="rgba(0,229,176,0.20)",
        line=dict(color=GREEN, width=1.5), name="Upside"))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_arr[m_dn], [S0]]),
        y=np.concatenate([    y[m_dn],  [y0]]),
        fill="tozeroy", fillcolor="rgba(255,69,96,0.20)",
        line=dict(color=RED, width=1.5), name="Downside"))
    for lbl, val, col in [
        ("P1",  pcts["P1"],    RED),  ("P5",  pcts["P5"],    "#ff7c96"),
        ("P25", pcts["P25"],   AMBER),("Med", pcts["Median"],BLUE),
        ("Mean",pcts["Mean"],  PURPLE),("P75", pcts["P75"],  AMBER),
        ("P95", pcts["P95"],   "#ff7c96"),("P99",pcts["P99"],RED),
    ]:
        fig.add_vline(x=val, line_color=col, line_dash="dot", line_width=1.2,
                      annotation_text=f" {lbl}", annotation_font_color=col,
                      annotation_font_size=10)
    fig.add_vline(x=S0, line_color="#ffffff", line_width=2, line_dash="dash",
                  annotation_text="  Current", annotation_font_color="#ffffff")
    fig.update_layout(**base_layout("KDE Distribution with Percentile Markers"),
        xaxis=dict(title="Price Level", gridcolor=GRID, tickformat=",.0f"),
        yaxis=dict(title="Density",     gridcolor=GRID),
        legend=dict(bgcolor=CARD, font=dict(color="#a0aec0", size=11)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#071428,#0d1f38,#071a30);
            border-bottom:1px solid #1a3050;padding:18px 24px;
            border-radius:10px;margin-bottom:8px;
            display:flex;justify-content:space-between;align-items:center;">
  <div>
    <span style="background:#ff4560;color:#fff;font-size:10px;font-weight:700;
                 letter-spacing:1.5px;padding:3px 8px;border-radius:3px;
                 margin-right:12px;">LIVE SIM</span>
    <span style="font-size:22px;font-weight:700;color:#e8f0ff;">
      NIFTY 50 <span style="color:#00e5b0;">Monte Carlo</span>
      Prediction Dashboard
    </span>
  </div>
  <div style="text-align:right;">
    <div style="color:#5a7a9a;font-size:12px;">
      Bloomberg-Style · Quant Analytics · {data['last_date']}
    </div>
    <div style="color:#00e5b0;font-size:24px;font-weight:700;">
      &#8377; {S0:,.2f}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — MARKET SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section_hdr("A", "Market Summary")

# ── Session mode badge ────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:#0a1628;border:1px solid #1a2f4a;border-radius:8px;'
    f'padding:10px 18px;margin-bottom:12px;font-size:13px;font-weight:600;'
    f'color:{session_colour};">'
    f'{session_label} &nbsp;·&nbsp; '
    f'News weight: <strong>{news_weight*100:.1f}% p.a.</strong> &nbsp;·&nbsp; '
    f'SGX weight: <strong>{sgx_weight_ann*100:.1f}% p.a.</strong>'
    f'</div>',
    unsafe_allow_html=True)

# ── Row 1: core metrics ───────────────────────────────────────────────────────
c = st.columns(5)
with c[0]: kpi("Current NIFTY",  f"&#8377; {S0:,.2f}",
               f"Last close · {data['last_date']}", GREEN)
with c[1]: kpi("Ann. Volatility", f"{ann_vol:.2f}%",
               f"&#963;&#xd7;&#x221a;252 · {lookback}yr", AMBER)
with c[2]: kpi("Baseline Drift &#956;", f"{ann_mu:+.2f}%",
               "Annualised mean return", val_colour(ann_mu))
with c[3]: kpi("Adjusted Drift", f"{ann_adjmu:+.2f}%",
               "After all signals", val_colour(ann_adjmu))
with c[4]: kpi("Simulations", f"{n_sims:,}",
               "GBM · 1-step · seed 42", PURPLE)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Row 2: signal breakdown ───────────────────────────────────────────────────
c2 = st.columns(5)
with c2[0]:
    sgx_col = val_colour(sgx["premium_pct"])
    sgx_avail = "✓ " + sgx["source"] if sgx["available"] else "⚠ Fallback proxy"
    kpi("SGX / GIFT Nifty", f"&#8377; {sgx['price']:,.2f}",
        f"{sgx_avail}", sgx_col)
with c2[1]:
    sgx_prem_col = val_colour(sgx["premium_pct"])
    kpi("SGX Premium", f"{sgx['premium_pct']:+.3f}%",
        f"Signal: {sgx['signal']:+.3f} · Contrib: {sgx_contrib:+.2f}% p.a.",
        sgx_prem_col)
with c2[2]: kpi("Global Factor",  f"{gf:+.4f}",
               f"Z-score &#177;3 · Contrib: {gf_contrib:+.2f}% p.a.",
               val_colour(gf))
with c2[3]: kpi("News Sentiment", f"{ns:+.4f}",
               f"{sentiment['method']} · Contrib: {ns_contrib:+.2f}% p.a.",
               val_colour(ns))
with c2[4]: kpi("Total Drift Adj.", f"{drift_adj:+.2f}%",
               f"GF + SGX + News · annualised", val_colour(drift_adj))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — SIMULATION RESULTS
# ═════════════════════════════════════════════════════════════════════════════
section_hdr("B", "Simulation Results")

col_l, col_r = st.columns([1, 1.6])

with col_l:
    st.caption("PERCENTILE TABLE")
    rows = []
    for lbl in ["P1","P5","P25","Median","Mean","P75","P95","P99"]:
        chg   = chgs[lbl]
        arrow = "▲" if chg >= 0 else "▼"
        rows.append({
            "Percentile"      : lbl,
            "Predicted Price" : f"₹ {pcts[lbl]:>10,.2f}",
            "Change %"        : f"{arrow} {abs(chg):.3f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True, height=308)

with col_r:
    st.caption("GLOBAL INDEX — PREV-DAY RETURNS")
    gr_rows = []
    for nm, ret in prev_returns.items():
        arrow = "▲" if ret >= 0 else "▼"
        gr_rows.append({"Index": nm,
                         "Return": f"{arrow} {abs(ret)*100:.3f}%"})
    st.dataframe(pd.DataFrame(gr_rows), use_container_width=True,
                 hide_index=True, height=308)

st.caption("LATEST HEADLINES")
hl_items = "".join(
    f'<span style="color:#8a9bb0;font-size:12px;">&#x2022; {h}</span><br>'
    for h in sentiment["headlines"][:8]
)
st.markdown(
    f'<div style="background:{CARD};border:1px solid #1a2f4a;'
    f'border-radius:8px;padding:12px 16px;">{hl_items}</div>',
    unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — PROBABILITY GRID
# ═════════════════════════════════════════════════════════════════════════════
section_hdr("C", "Probability Grid")

p1, p2, p3, p4 = st.columns(4)
with p1: prob_card("🔼", "Price Up Tomorrow",    probs["up"],        invert=False)
with p2: prob_card("🔽", "Price Down Tomorrow",  probs["down"],      invert=True)
with p3: prob_card("🚀", "Move > +0.5%",         probs["above_p05"], invert=False)
with p4: prob_card("📉", "Move < &#8722;0.5%",   probs["below_m05"], invert=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — CHARTS
# ═════════════════════════════════════════════════════════════════════════════
section_hdr("D", "Charts — Plotly Dark Theme")

CFG = {"displayModeBar": False}

r1a, r1b = st.columns(2)
with r1a:
    st.plotly_chart(build_paths(),     use_container_width=True, config=CFG)
with r1b:
    st.plotly_chart(build_histogram(), use_container_width=True, config=CFG)

r2a, r2b = st.columns(2)
with r2a:
    st.plotly_chart(build_confidence(),use_container_width=True, config=CFG)
with r2b:
    st.plotly_chart(build_kde(),       use_container_width=True, config=CFG)


# ═════════════════════════════════════════════════════════════════════════════
# INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════
section_hdr("✦", "Quantitative Interpretation")

mean_chg   = chgs["Mean"]
bias       = ("bullish 📈" if mean_chg >  0.15
              else "bearish 📉" if mean_chg < -0.15
              else "neutral ⚖️")
vol_regime = ("elevated" if ann_vol > 22
              else "moderate" if ann_vol > 14
              else "subdued")
gf_desc    = ("positive global tailwinds" if gf >  0.5
              else "negative global headwinds" if gf < -0.5
              else "mixed global cues")
ns_desc    = ("bullish" if ns >  0.1 else "bearish" if ns < -0.1 else "neutral")
sgx_desc   = ("pricing a gap-up" if sgx["premium_pct"] >  0.10
              else "pricing a gap-down" if sgx["premium_pct"] < -0.10
              else "broadly flat")
weekend_note = (
    f" News sentiment receives a <strong>{news_weight/NEWS_WEIGHT_WEEKDAY:.0f}× "
    f"weekend multiplier</strong> ({news_weight*100:.1f}% p.a. vs the "
    f"{NEWS_WEIGHT_WEEKDAY*100:.1f}% weekday baseline) reflecting 2-day "
    f"news accumulation with no intraday price release. "
    if IS_WEEKEND else
    (f" Friday pre-weekend elevated news weighting ({news_weight*100:.1f}% p.a.) "
     f"reflects heightened gap risk into the weekend. " if IS_FRIDAY else "")
)

st.markdown(f"""
<div class="interp">
The model carries a <strong>{bias}</strong> bias for the next NIFTY session,
with the mean simulated return of <strong>{mean_chg:+.3f}%</strong> driven by
a three-signal adjusted drift (global indices, SGX futures, news sentiment).
Annualised volatility is <strong>{ann_vol:.2f}%</strong> —
a <strong>{vol_regime}</strong> regime — with the P5–P95 band spanning
<strong>&#8377;{pcts['P5']:,.0f} – &#8377;{pcts['P95']:,.0f}</strong>.
SGX / GIFT Nifty futures are <strong>{sgx_desc}</strong>
({sgx["premium_pct"]:+.3f}% premium, contributing {sgx_contrib:+.2f}% p.a.);
global indices signal <strong>{gf_desc}</strong>
(factor {gf:+.3f}, contributing {gf_contrib:+.2f}% p.a.);
and news flow is <strong>{ns_desc}</strong>
(score {ns:+.3f}, contributing {ns_contrib:+.2f}% p.a. at current session weight).{weekend_note}
Total drift adjustment above the historical baseline: <strong>{drift_adj:+.2f}% annualised</strong>.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  NIFTY 50 Monte Carlo Dashboard &nbsp;·&nbsp;
  Powered by yfinance · Plotly · GBM &nbsp;·&nbsp;
  <strong style="color:#3a7bd5;">For Educational &amp; Research Use Only
  — Not Investment Advice</strong>
</div>
""", unsafe_allow_html=True)
