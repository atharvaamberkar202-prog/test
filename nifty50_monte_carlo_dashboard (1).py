"""
=============================================================================
  NIFTY 50 — Next-Day Monte Carlo Prediction Dashboard
  Streamlit App · Bloomberg Terminal Style · Dark Theme

  Run:
      streamlit run nifty50_monte_carlo_dashboard.py

  requirements.txt:
      streamlit
      streamlit-autorefresh
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
import os, math, warnings, datetime, time
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

# ── streamlit-autorefresh ─────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

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
# AUTO-REFRESH — must be near the TOP so it triggers a full fresh rerun
# Every 60 seconds; cache TTL is set to 55s so data is always fresh.
# ─────────────────────────────────────────────────────────────────────────────
REFRESH_INTERVAL_MS = 60_000   # 60 seconds
CACHE_TTL_SEC       = 55       # slightly less than refresh interval

if AUTOREFRESH_OK:
    _refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="auto_refresh")
else:
    st.warning("⚠️  `streamlit-autorefresh` not installed — auto-refresh disabled. "
               "Install with: pip install streamlit-autorefresh")
    _refresh_count = 0

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
# CONSTANTS & COLOURS
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_TICKER = "^NSEI"

GREEN  = "#00e5b0"
RED    = "#ff4560"
AMBER  = "#f0c040"
BLUE   = "#00c8ff"
PURPLE = "#7b61ff"
CARD   = "#0a1628"
GRID   = "#162033"

NEWS_WEIGHT_WEEKDAY = 0.010
NEWS_WEIGHT_FRIDAY  = 0.018
NEWS_WEIGHT_WEEKEND = 0.030

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

    if st.button("🔄 Force Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
**Auto-Refresh:** every {REFRESH_INTERVAL_MS // 1000}s · Refresh #{_refresh_count}

**GIFT Nifty Source Priority**
1. `GIFT NIFTY` — direct yfinance `NIFTY.NS` / `BANKNIFTY.NS`
2. NSE near-month futures `NIFTYYYMMM.NS`
3. S&P 500 E-mini `ES=F` directional proxy

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
# HELPERS — yfinance download wrapper
# ─────────────────────────────────────────────────────────────────────────────
def _yf_download(ticker: str, **kwargs) -> pd.DataFrame:
    """Robust yfinance downloader that handles both old and new API."""
    try:
        df = yf.download(ticker, progress=False,
                         multi_level_index=False, **kwargs)
    except TypeError:
        try:
            df = yf.download(ticker, progress=False, **kwargs)
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def _extract_close(df: pd.DataFrame, ticker: str = "") -> pd.Series:
    """Extract a clean Close price Series from a yfinance DataFrame."""
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    # Flatten MultiIndex columns (yfinance ≥0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Close" in df.columns:
        col = "Close"
    else:
        nums = df.select_dtypes("number").columns
        if len(nums) == 0:
            return pd.Series(dtype=float, name=ticker)
        col = nums[0]
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.squeeze().dropna()
    if not isinstance(s, pd.Series):
        return pd.Series(dtype=float, name=ticker)
    s.name = ticker
    s.index = pd.to_datetime(s.index)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# GIFT / SGX NIFTY FUTURES FETCHER  (NO calculations — real data only)
# ─────────────────────────────────────────────────────────────────────────────
def _nse_futures_tickers() -> list:
    """Return up to 3 NSE near-month futures tickers in yfinance format."""
    today = datetime.date.today()
    tickers = []
    for delta in range(3):
        yr    = today.year  + (today.month - 1 + delta) // 12
        mo    = (today.month - 1 + delta) % 12 + 1
        mon3  = datetime.date(yr, mo, 1).strftime("%b").upper()   # e.g. MAR
        yr2   = str(yr)[2:]                                        # e.g. 25
        tickers.append(f"NIFTY{yr2}{mon3}FUT.NS")
    return tickers


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def fetch_gift_nifty(nifty_spot: float, _cache_bust: int) -> dict:
    """
    Fetch GIFT Nifty / SGX Nifty futures price in REAL Nifty-point units.

    Strategy (tried in order):
      1. GIFT NIFTY direct — yfinance ticker  'GIFTNIFTY.NS'  (NSE listed)
      2. NIFTY spot intraday tick (5-min) — best pre-open/live proxy
         when GIFT Nifty data is unavailable
      3. NSE near-month futures  NIFTY25MARFUT.NS  etc.
      4. ES=F / NQ=F *return* applied to Nifty spot as a directional proxy
         (raw price is NEVER shown; only the implied Nifty-equivalent)
      5. Neutral hard-fallback (signal = 0)

    _cache_bust: pass an integer that changes each refresh cycle to force
                 Streamlit to bypass the cache and fetch fresh data.
    """
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

    # ── 1. GIFT NIFTY direct ──────────────────────────────────────────────────
    # NSE listed GIFT Nifty continuous futures ticker
    for gift_ticker in ("GIFTNIFTY.NS"):
        try:
            df = _yf_download(gift_ticker, period="5d", interval="5m")
            s  = _extract_close(df, gift_ticker)
            if len(s) >= 1:
                gift_price = float(s.iloc[-1])
                # GIFT Nifty trades in Nifty-equivalent points; validate range
                if 0.85 * nifty_spot <= gift_price <= 1.15 * nifty_spot:
                    premium_pct = (gift_price / nifty_spot - 1.0) * 100
                    signal      = float(np.clip(premium_pct / 2.0, -1.0, 1.0))
                    ts          = str(s.index[-1])
                    return {
                        "price"      : round(gift_price, 2),
                        "premium_pct": round(premium_pct, 3),
                        "signal"     : signal,
                        "ticker"     : gift_ticker,
                        "source"     : f"GIFT NIFTY ({gift_ticker})",
                        "timestamp"  : ts,
                        "available"  : True,
                        "is_proxy"   : False,
                    }
        except Exception:
            pass

    # ── 2. NIFTY intraday live tick (best real-time during market hours) ──────
    try:
        df    = _yf_download("^NSEI", period="1d", interval="1m")
        s     = _extract_close(df, "^NSEI")
        if len(s) >= 2:
            live  = float(s.iloc[-1])
            # Use latest intraday vs previous session close as the signal
            if 0.90 * nifty_spot <= live <= 1.10 * nifty_spot:
                premium_pct = (live / nifty_spot - 1.0) * 100
                signal      = float(np.clip(premium_pct / 2.0, -1.0, 1.0))
                ts          = str(s.index[-1])
                return {
                    "price"      : round(live, 2),
                    "premium_pct": round(premium_pct, 3),
                    "signal"     : signal,
                    "ticker"     : "^NSEI (1-min)",
                    "source"     : "NIFTY Live Intraday (1-min tick)",
                    "timestamp"  : ts,
                    "available"  : True,
                    "is_proxy"   : False,
                }
    except Exception:
        pass

    # ── 3. NSE near-month futures ─────────────────────────────────────────────
    for ticker in _nse_futures_tickers():
        try:
            df = _yf_download(ticker, start=week_ago, end=tomorrow)
            s  = _extract_close(df, ticker)
            if len(s) < 1:
                continue
            fut_price = float(s.iloc[-1])
            # Strict sanity: must be within 5% of spot (genuine NIFTY-unit futures)
            if fut_price <= 0 or abs(fut_price / nifty_spot - 1) > 0.05:
                continue
            premium_pct = (fut_price / nifty_spot - 1.0) * 100
            signal      = float(np.clip(premium_pct / 2.0, -1.0, 1.0))
            ts          = str(s.index[-1])
            return {
                "price"      : round(fut_price, 2),
                "premium_pct": round(premium_pct, 3),
                "signal"     : signal,
                "ticker"     : ticker,
                "source"     : f"NSE Near-Month Futures ({ticker})",
                "timestamp"  : ts,
                "available"  : True,
                "is_proxy"   : False,
            }
        except Exception:
            continue

    # ── 4. ES=F / NQ=F directional proxy ─────────────────────────────────────
    # IMPORTANT: we use the proxy's *return* only — the raw ES/NQ price
    # is NEVER shown.  The implied Nifty-equivalent price is derived from
    # applying the proxy's daily return to the last NIFTY close.
    for ticker in ("ES=F", "NQ=F"):
        try:
            df = _yf_download(ticker, start=week_ago, end=tomorrow)
            s  = _extract_close(df, ticker)
            if len(s) < 2:
                continue
            proxy_ret     = float(s.iloc[-1] / s.iloc[-2]) - 1.0
            implied_nifty = nifty_spot * (1.0 + proxy_ret)
            premium_pct   = proxy_ret * 100
            signal        = float(np.clip(premium_pct / 2.0, -1.0, 1.0))
            ts            = str(s.index[-1])
            return {
                "price"      : round(implied_nifty, 2),  # NIFTY units, NOT ES price
                "premium_pct": round(premium_pct, 3),
                "signal"     : signal,
                "ticker"     : ticker,
                "source"     : f"Directional Proxy via {ticker} return (NIFTY-equivalent)",
                "timestamp"  : ts,
                "available"  : True,
                "is_proxy"   : True,
            }
        except Exception:
            continue

    # ── 5. Hard fallback ──────────────────────────────────────────────────────
    return {
        "price"      : nifty_spot,
        "premium_pct": 0.0,
        "signal"     : 0.0,
        "ticker"     : "N/A",
        "source"     : "Unavailable — signal set to neutral",
        "timestamp"  : "N/A",
        "available"  : False,
        "is_proxy"   : False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRICE DATA PIPELINE  (short TTL = always fresh on auto-refresh)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def fetch_prices(lookback_yr: int, _cache_bust: int) -> dict:
    """
    Download NIFTY 50 + 8 global indices.
    _cache_bust changes each auto-refresh cycle to force a fresh download.
    """
    end   = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.date.today() -
             datetime.timedelta(days=365 * lookback_yr + 30)).strftime("%Y-%m-%d")

    def _dl(ticker: str) -> pd.Series:
        df = _yf_download(ticker, start=start, end=end, auto_adjust=True)
        s  = _extract_close(df, ticker)
        s.index = pd.to_datetime(s.index)
        return s

    nifty = _dl(NIFTY_TICKER)
    if nifty.empty:
        return {}

    global_raw = {nm: _dl(tk) for nm, tk in GLOBAL_INDICES.items()}
    valid_globals = {nm: s for nm, s in global_raw.items() if not s.empty}

    all_series = {"__NIFTY__": nifty}
    all_series.update(valid_globals)

    combined = pd.concat(all_series.values(), axis=1, keys=all_series.keys())
    combined.index = pd.to_datetime(combined.index)
    combined.sort_index(inplace=True)

    nifty_a  = combined["__NIFTY__"].dropna()
    global_a = {nm: combined[nm].dropna() for nm in valid_globals
                if nm in combined.columns and not combined[nm].dropna().empty}

    def log_ret(s: pd.Series) -> pd.Series:
        return np.log(s / s.shift(1)).dropna()

    return {
        "nifty"          : nifty_a,
        "nifty_returns"  : log_ret(nifty_a),
        "global_closes"  : global_a,
        "global_returns" : {k: log_ret(v) for k, v in global_a.items()},
        "last_date"      : str(nifty_a.index[-1].date()),
        "fetch_time"     : datetime.datetime.now().strftime("%H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEWS SENTIMENT  (short TTL)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def get_sentiment(api_key: str, _cache_bust: int) -> dict:
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


# ─────────────────────────────────────────────────────────────────────────────
# CACHE-BUST KEY — changes every REFRESH_INTERVAL_MS milliseconds
# This forces @st.cache_data to treat each refresh as a new call even if
# the function arguments are unchanged.
# ─────────────────────────────────────────────────────────────────────────────
_cache_bust_key = int(time.time() // (REFRESH_INTERVAL_MS / 1000))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("📡 Fetching market data …"):
    data = fetch_prices(lookback, _cache_bust=_cache_bust_key)

if not data:
    st.error("❌ Failed to fetch NIFTY 50 data. Check your internet connection.")
    st.stop()

with st.spinner("📰 Analysing news sentiment …"):
    sentiment = get_sentiment(news_key, _cache_bust=_cache_bust_key)

# ── Base NIFTY price ──────────────────────────────────────────────────────────
S0 = float(data["nifty"].iloc[-1])

# ── GIFT / SGX Nifty futures ─────────────────────────────────────────────────
with st.spinner("🌐 Fetching GIFT Nifty futures …"):
    sgx = fetch_gift_nifty(S0, _cache_bust=_cache_bust_key)

# ── Weekend detection ─────────────────────────────────────────────────────────
_today     = datetime.date.today()
_weekday   = _today.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun
IS_WEEKEND = _weekday >= 5
IS_FRIDAY  = _weekday == 4

if IS_WEEKEND:
    session_label  = "🗓️ WEEKEND — Enhanced News & GIFT Nifty Weighting Active"
    session_colour = AMBER
    news_weight    = NEWS_WEIGHT_WEEKEND
    sgx_weight_ann = 0.04
elif IS_FRIDAY:
    session_label  = "📅 FRIDAY — Pre-Weekend Elevated Weighting"
    session_colour = "#f0a040"
    news_weight    = NEWS_WEIGHT_FRIDAY
    sgx_weight_ann = 0.025
else:
    session_label  = f"📈 WEEKDAY ({_today.strftime('%A')})"
    session_colour = GREEN
    news_weight    = NEWS_WEIGHT_WEEKDAY
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
sgx_s  = sgx["signal"]

# ── Signal composition ────────────────────────────────────────────────────────
gf_unit   = gf / 3.0
alpha_ann = (gf_unit * 0.03
             + sgx_s  * sgx_weight_ann
             + ns     * news_weight)
alpha_day = alpha_ann / 252
adj_mu    = mu + alpha_day

ann_vol   = sigma  * math.sqrt(252) * 100
ann_mu    = mu     * 252 * 100
ann_adjmu = adj_mu * 252 * 100
drift_adj = alpha_ann * 100

gf_contrib  = gf_unit * 0.03 * 100
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
chgs  = {k: (v / S0 - 1) * 100 for k, v in pcts.items()}
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
        title         = dict(text=title, font=dict(color="#e0e6f0", size=13)),
        template      = "plotly_dark",
        paper_bgcolor = CARD,
        plot_bgcolor  = "#0d1f35",
        height        = h,
        margin        = dict(l=52, r=18, t=44, b=42),
        font          = dict(color="#7a8fa6"),
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
                (adj_mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z))
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
        fig.add_trace(go.Scatter(
            x=[0,1,1,0], y=[S0, pcts[hi_k], pcts[lo_k], S0],
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

    fig   = go.Figure()
    m_up  = x_arr >= S0
    m_dn  = x_arr <  S0
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
now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
      Bloomberg-Style · Quant Analytics
    </div>
    <div style="color:#5a7a9a;font-size:11px;">
      Data as of: {data['last_date']} &nbsp;·&nbsp;
      Fetched: <span style="color:#00e5b0;">{data['fetch_time']}</span>
      &nbsp;·&nbsp; Page refresh #{_refresh_count}
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

st.markdown(
    f'<div style="background:#0a1628;border:1px solid #1a2f4a;border-radius:8px;'
    f'padding:10px 18px;margin-bottom:12px;font-size:13px;font-weight:600;'
    f'color:{session_colour};">'
    f'{session_label} &nbsp;·&nbsp; '
    f'News weight: <strong>{news_weight*100:.1f}% p.a.</strong> &nbsp;·&nbsp; '
    f'GIFT Nifty weight: <strong>{sgx_weight_ann*100:.1f}% p.a.</strong>'
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
    sgx_col  = val_colour(sgx["premium_pct"])
    is_proxy = sgx.get("is_proxy", False)
    if not sgx["available"]:
        sgx_status = "⚠ Unavailable"
    elif is_proxy:
        sgx_status = f"⚠ Proxy ({sgx['ticker']}) — directional est."
    else:
        sgx_status = f"✓ {sgx['source']}"
    price_label = f"&#8377; {sgx['price']:,.2f}" if sgx["available"] else "N/A"
    kpi("GIFT Nifty (actual/implied)", price_label,
        f"{sgx_status} · as of {sgx.get('timestamp','N/A')[:19]}", sgx_col)
with c2[1]:
    sgx_prem_col = val_colour(sgx["premium_pct"])
    kpi("GIFT Nifty Premium", f"{sgx['premium_pct']:+.3f}%",
        f"Signal: {sgx['signal']:+.3f} · Contrib: {sgx_contrib:+.2f}% p.a.",
        sgx_prem_col)
with c2[2]: kpi("Global Factor",  f"{gf:+.4f}",
               f"Z-score ±3 · Contrib: {gf_contrib:+.2f}% p.a.",
               val_colour(gf))
with c2[3]: kpi("News Sentiment", f"{ns:+.4f}",
               f"{sentiment['method']} · Contrib: {ns_contrib:+.2f}% p.a.",
               val_colour(ns))
with c2[4]: kpi("Total Drift Adj.", f"{drift_adj:+.2f}%",
               "GF + GIFT Nifty + News · ann.", val_colour(drift_adj))


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
section_hdr("✦", "What Does This All Mean?")

gf_signal_unit  = gf / 3.0
sgx_signal_unit = sgx["signal"]
ns_signal_unit  = ns
composite       = (gf_signal_unit + sgx_signal_unit + ns_signal_unit) / 3.0

if   composite >  0.25:  verdict = "BULLISH"       ; v_col = GREEN  ; v_icon = "🟢"
elif composite >  0.05:  verdict = "MILDLY BULLISH" ; v_col = "#70e0b0" ; v_icon = "🟡"
elif composite < -0.25:  verdict = "BEARISH"        ; v_col = RED    ; v_icon = "🔴"
elif composite < -0.05:  verdict = "MILDLY BEARISH" ; v_col = "#ff8c9a" ; v_icon = "🟠"
else:                    verdict = "NEUTRAL"        ; v_col = AMBER  ; v_icon = "⚪"

mean_chg = chgs["Mean"]

def _arrow(v, threshold=0.05):
    if   v >  threshold: return "🟢 Positive"
    elif v < -threshold: return "🔴 Negative"
    else:                return "⚪ Neutral"

gf_label  = _arrow(gf_signal_unit,  0.10)
sgx_label = _arrow(sgx_signal_unit, 0.05)
ns_label  = _arrow(ns_signal_unit,  0.05)

we_note = ""
if IS_WEEKEND:
    we_note = (f"📅 It's the weekend — news headlines carry extra weight "
               f"({news_weight*100:.0f}% vs the normal {NEWS_WEIGHT_WEEKDAY*100:.0f}%) "
               f"because markets haven't reacted yet.")
elif IS_FRIDAY:
    we_note = (f"📅 It's Friday — news weight is slightly elevated "
               f"({news_weight*100:.0f}%) to account for weekend gap risk.")

# ── Big verdict banner ───────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#0a1628;border:2px solid {v_col};border-radius:12px;
            padding:22px 28px;margin-bottom:16px;text-align:center;">
  <div style="font-size:13px;color:#7a9abf;text-transform:uppercase;
              letter-spacing:2px;margin-bottom:6px;">Overall Direction for Next Session</div>
  <div style="font-size:42px;font-weight:800;color:{v_col};
              letter-spacing:1px;">{v_icon} {verdict}</div>
  <div style="font-size:15px;color:#a0bcd8;margin-top:8px;">
    Average of all signals: <strong style="color:{v_col};">{composite:+.2f}</strong>
    &nbsp;·&nbsp;
    Expected move: <strong style="color:{v_col};">{mean_chg:+.3f}%</strong>
    &nbsp;·&nbsp;
    Price range (90%): <strong>&#8377;{pcts['P5']:,.0f} – &#8377;{pcts['P95']:,.0f}</strong>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Three signal cards ───────────────────────────────────────────────────────
ia, ib, ic = st.columns(3)

with ia:
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1a2f4a;border-radius:10px;padding:18px;">
      <div style="font-size:11px;color:#4a6a8a;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:6px;">🌍 Global Markets</div>
      <div style="font-size:18px;font-weight:700;color:#e0ecff;margin-bottom:4px;">
        {gf_label}
      </div>
      <div style="font-size:13px;color:#7a9abf;line-height:1.6;">
        The last session across major world indices
        (S&amp;P 500, DAX, Nikkei, etc.) was
        <strong>{"mostly up" if gf_signal_unit > 0.05 else "mostly down" if gf_signal_unit < -0.05 else "mixed"}</strong>.
        This {"adds upward" if gf_signal_unit >= 0 else "adds downward"} momentum
        to NIFTY's opening.
      </div>
    </div>""", unsafe_allow_html=True)

with ib:
    premium_word = "above" if sgx["premium_pct"] > 0 else "below"
    proxy_note   = " (estimated via global futures return)" if sgx.get("is_proxy") else ""
    source_badge = (f'<span style="font-size:10px;color:#4a8a4a;background:#0a200a;'
                    f'padding:2px 6px;border-radius:3px;border:1px solid #2a5a2a;">'
                    f'✓ REAL DATA</span>'
                    if not sgx.get("is_proxy") and sgx["available"]
                    else f'<span style="font-size:10px;color:#8a6a20;background:#1a1400;'
                         f'padding:2px 6px;border-radius:3px;border:1px solid #4a3a10;">'
                         f'⚠ ESTIMATED</span>')
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1a2f4a;border-radius:10px;padding:18px;">
      <div style="font-size:11px;color:#4a6a8a;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:6px;">
        📊 GIFT / SGX Nifty Futures &nbsp; {source_badge}
      </div>
      <div style="font-size:18px;font-weight:700;color:#e0ecff;margin-bottom:4px;">
        {sgx_label}
      </div>
      <div style="font-size:13px;color:#7a9abf;line-height:1.6;">
        GIFT Nifty is at <strong>&#8377;{sgx['price']:,.0f}</strong>
        — <strong>{abs(sgx["premium_pct"]):.2f}% {premium_word}</strong>
        the last NIFTY close of &#8377;{S0:,.0f}{proxy_note}.
        Source: <em>{sgx['source']}</em>.
        This signals a {"higher" if sgx["premium_pct"] > 0 else "lower"} open.
      </div>
    </div>""", unsafe_allow_html=True)

with ic:
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1a2f4a;border-radius:10px;padding:18px;">
      <div style="font-size:11px;color:#4a6a8a;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:6px;">📰 News Sentiment</div>
      <div style="font-size:18px;font-weight:700;color:#e0ecff;margin-bottom:4px;">
        {ns_label}
      </div>
      <div style="font-size:13px;color:#7a9abf;line-height:1.6;">
        The tone of recent financial headlines is
        <strong>{"positive" if ns > 0.05 else "negative" if ns < -0.05 else "broadly neutral"}</strong>
        (score: {ns:+.2f}).
        {"Weekend weighting means this signal carries more influence today." if IS_WEEKEND or IS_FRIDAY else "Standard weekday weighting applied."}
      </div>
    </div>""", unsafe_allow_html=True)

if we_note:
    st.markdown(
        f'<div style="background:#071e10;border:1px solid #1a4a2a;border-radius:8px;'
        f'padding:12px 18px;margin-top:12px;font-size:13px;color:#80c8a0;">'
        f'{we_note}</div>',
        unsafe_allow_html=True)

vol_plain = ("higher than usual" if ann_vol > 22
             else "around normal levels" if ann_vol > 14
             else "lower than usual")
st.markdown(
    f'<div style="background:#0a1628;border:1px solid #1a2f4a;border-radius:8px;'
    f'padding:12px 18px;margin-top:10px;font-size:13px;color:#7a9abf;">'
    f'📉 <strong style="color:#e0ecff;">Expected volatility</strong> is '
    f'<strong>{vol_plain}</strong> at {ann_vol:.1f}% annualised — the market '
    f'could realistically close anywhere between '
    f'<strong style="color:{RED};">&#8377;{pcts["P5"]:,.0f}</strong> and '
    f'<strong style="color:{GREEN};">&#8377;{pcts["P95"]:,.0f}</strong> '
    f'in the next session (90% of simulated outcomes).'
    f'</div>',
    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="disclaimer">
  NIFTY 50 Monte Carlo Dashboard &nbsp;·&nbsp;
  Powered by yfinance · Plotly · GBM &nbsp;·&nbsp;
  Last fetched: {data['fetch_time']} &nbsp;·&nbsp; Auto-refresh every {REFRESH_INTERVAL_MS // 1000}s
  <br>
  <strong style="color:#3a7bd5;">For Educational &amp; Research Use Only
  — Not Investment Advice</strong>
</div>
""", unsafe_allow_html=True)
