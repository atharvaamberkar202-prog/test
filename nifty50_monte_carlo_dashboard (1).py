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
      feedparser
      lxml
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, math, warnings, datetime, time, re, html as html_module
import numpy  as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── feedparser for RSS news ───────────────────────────────────────────────────
try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False

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
# AUTO-REFRESH — top of script so every 60s triggers a full fresh rerun
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
        help="Free key from newsapi.org — leave blank to use live RSS feeds (ET/Moneycontrol)")

    if st.button("🔄 Force Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
**Auto-Refresh:** every {REFRESH_INTERVAL_MS // 1000}s · Refresh #{_refresh_count}

**GIFT Nifty Source Priority**
1. NSE India public API (`allIndices`) — official live data
2. Stooq.com GIFT Nifty scrape
3. NSE near-month futures `NIFTYYYMMM.NS`
4. `ES=F` S&P 500 futures directional proxy

**News Sources (no API key needed)**
- Economic Times Markets RSS
- Moneycontrol Markets RSS
- LiveMint RSS
- NewsAPI (if key provided)

**Drift Formula**
```
adj_mu = mu + alpha/252
alpha  = GF×3% + SGX×w% + NS×nw%
```

**Weekend Weighting**
- Mon–Thu: news `nw=1.0%`, SGX `w=1.5%`
- Friday:  news `nw=1.8%`, SGX `w=2.5%`
- Weekend: news `nw=3.0%`, SGX `w=4.0%`
""")
    st.caption("⚠️ Educational use only. Not investment advice.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — yfinance download wrapper
# ─────────────────────────────────────────────────────────────────────────────
def _yf_download(ticker: str, **kwargs) -> pd.DataFrame:
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
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    col = "Close" if "Close" in df.columns else (
        df.select_dtypes("number").columns[0]
        if len(df.select_dtypes("number").columns) else None)
    if col is None:
        return pd.Series(dtype=float, name=ticker)
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
# GIFT NIFTY FETCHER — real data sources, no calculation
# ─────────────────────────────────────────────────────────────────────────────
_NSE_HEADERS = {
    "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0.0.0 Safari/537.36",
    "Accept"         : "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer"        : "https://www.nseindia.com/",
    "Connection"     : "keep-alive",
}


def _nse_session() -> requests.Session:
    """Create an NSE-cookie-primed session (NSE requires cookies from homepage)."""
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com/", timeout=8)
    except Exception:
        pass
    return s


def _nse_futures_tickers() -> list:
    today = datetime.date.today()
    tickers = []
    for delta in range(3):
        yr   = today.year  + (today.month - 1 + delta) // 12
        mo   = (today.month - 1 + delta) % 12 + 1
        mon3 = datetime.date(yr, mo, 1).strftime("%b").upper()
        yr2  = str(yr)[2:]
        tickers.append(f"NIFTY{yr2}{mon3}FUT.NS")
    return tickers


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def fetch_gift_nifty(nifty_spot: float, _cache_bust: int) -> dict:
    """
    Fetch GIFT Nifty live price from real data sources.

    Priority:
      1. NSE India public JSON API  (nseindia.com/api/allIndices)
         → returns "GIFT NIFTY" with last, open, high, low, change%
      2. Stooq.com NIFTY1! scrape (covers GIFT Nifty futures)
      3. NSE near-month futures via yfinance  NIFTYYYMMM.NS
      4. ES=F return applied to Nifty spot as directional proxy
      5. Neutral hard-fallback
    """
    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # ── 1. NSE India official allIndices API ──────────────────────────────────
    try:
        sess = _nse_session()
        resp = sess.get(
            "https://www.nseindia.com/api/allIndices",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        indices = data.get("data", [])
        for idx in indices:
            name = str(idx.get("index", "")).upper()
            if "GIFT" in name and "NIFTY" in name:
                last = float(idx.get("last", 0) or idx.get("indexSymbol", 0) or 0)
                if last <= 0:
                    # try alternate field names
                    last = float(idx.get("currentValue", 0) or 0)
                if last > 0 and 0.80 * nifty_spot <= last <= 1.20 * nifty_spot:
                    prem = (last / nifty_spot - 1.0) * 100
                    sig  = float(np.clip(prem / 2.0, -1.0, 1.0))
                    ts   = str(idx.get("timeVal", datetime.datetime.now().strftime("%H:%M")))
                    chg  = float(idx.get("percentChange", 0) or 0)
                    return {
                        "price"      : round(last, 2),
                        "premium_pct": round(prem, 3),
                        "signal"     : sig,
                        "ticker"     : "GIFT NIFTY",
                        "source"     : "NSE India API (allIndices)",
                        "timestamp"  : ts,
                        "pct_change" : round(chg, 3),
                        "available"  : True,
                        "is_proxy"   : False,
                    }
    except Exception:
        pass

    # ── 2. NSE India allIndices — GIFT NIFTY 50 alternative field check ───────
    try:
        sess = _nse_session()
        resp = sess.get(
            "https://www.nseindia.com/api/allIndices",
            timeout=10
        )
        if resp.ok:
            raw = resp.text
            # regex fallback: find any JSON object where index contains GIFT
            matches = re.findall(
                r'"index"\s*:\s*"([^"]*GIFT[^"]*)".*?"last"\s*:\s*([\d.]+)',
                raw, re.IGNORECASE
            )
            for name_m, val_m in matches:
                last = float(val_m)
                if 0.80 * nifty_spot <= last <= 1.20 * nifty_spot:
                    prem = (last / nifty_spot - 1.0) * 100
                    sig  = float(np.clip(prem / 2.0, -1.0, 1.0))
                    return {
                        "price"      : round(last, 2),
                        "premium_pct": round(prem, 3),
                        "signal"     : sig,
                        "ticker"     : "GIFT NIFTY",
                        "source"     : f"NSE India API ({name_m})",
                        "timestamp"  : datetime.datetime.now().strftime("%H:%M:%S"),
                        "pct_change" : 0.0,
                        "available"  : True,
                        "is_proxy"   : False,
                    }
    except Exception:
        pass

    # ── 3. Stooq.com — GIFT Nifty futures (GNF) ──────────────────────────────
    for stooq_sym in ("gnf.f", "nifty1!.ix"):
        try:
            url = f"https://stooq.com/q/l/?s={stooq_sym}&f=sd2t2ohlcv&h&e=csv"
            r   = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if r.ok and "Date" in r.text:
                lines = r.text.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split(",")
                    close_val = float(parts[5]) if len(parts) > 5 else 0
                    if close_val > 0 and 0.80 * nifty_spot <= close_val <= 1.20 * nifty_spot:
                        prem = (close_val / nifty_spot - 1.0) * 100
                        sig  = float(np.clip(prem / 2.0, -1.0, 1.0))
                        return {
                            "price"      : round(close_val, 2),
                            "premium_pct": round(prem, 3),
                            "signal"     : sig,
                            "ticker"     : stooq_sym.upper(),
                            "source"     : f"Stooq.com ({stooq_sym.upper()})",
                            "timestamp"  : parts[1] if len(parts) > 1 else "N/A",
                            "pct_change" : 0.0,
                            "available"  : True,
                            "is_proxy"   : False,
                        }
        except Exception:
            continue

    # ── 4. NSE near-month futures via yfinance ────────────────────────────────
    for ticker in _nse_futures_tickers():
        try:
            df = _yf_download(ticker, start=week_ago, end=tomorrow)
            s  = _extract_close(df, ticker)
            if len(s) < 1:
                continue
            fut_price = float(s.iloc[-1])
            if fut_price <= 0 or abs(fut_price / nifty_spot - 1) > 0.05:
                continue
            prem = (fut_price / nifty_spot - 1.0) * 100
            sig  = float(np.clip(prem / 2.0, -1.0, 1.0))
            return {
                "price"      : round(fut_price, 2),
                "premium_pct": round(prem, 3),
                "signal"     : sig,
                "ticker"     : ticker,
                "source"     : f"NSE Near-Month Futures ({ticker})",
                "timestamp"  : str(s.index[-1]),
                "pct_change" : 0.0,
                "available"  : True,
                "is_proxy"   : False,
            }
        except Exception:
            continue

    # ── 5. ES=F / NQ=F directional proxy ─────────────────────────────────────
    for ticker in ("ES=F", "NQ=F"):
        try:
            df = _yf_download(ticker, start=week_ago, end=tomorrow)
            s  = _extract_close(df, ticker)
            if len(s) < 2:
                continue
            proxy_ret     = float(s.iloc[-1] / s.iloc[-2]) - 1.0
            implied_nifty = nifty_spot * (1.0 + proxy_ret)
            prem          = proxy_ret * 100
            sig           = float(np.clip(prem / 2.0, -1.0, 1.0))
            return {
                "price"      : round(implied_nifty, 2),
                "premium_pct": round(prem, 3),
                "signal"     : sig,
                "ticker"     : ticker,
                "source"     : f"Proxy via {ticker} return (NIFTY-equivalent estimate)",
                "timestamp"  : str(s.index[-1]),
                "pct_change" : round(prem, 3),
                "available"  : True,
                "is_proxy"   : True,
            }
        except Exception:
            continue

    # ── 6. Hard fallback ──────────────────────────────────────────────────────
    return {
        "price"      : nifty_spot,
        "premium_pct": 0.0,
        "signal"     : 0.0,
        "ticker"     : "N/A",
        "source"     : "All sources unavailable — signal set to neutral",
        "timestamp"  : "N/A",
        "pct_change" : 0.0,
        "available"  : False,
        "is_proxy"   : False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRICE DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def fetch_prices(lookback_yr: int, _cache_bust: int) -> dict:
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

    global_raw    = {nm: _dl(tk) for nm, tk in GLOBAL_INDICES.items()}
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
# NEWS — Real RSS
