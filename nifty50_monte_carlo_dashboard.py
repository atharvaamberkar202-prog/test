"""
=============================================================================
  NIFTY 50 — Next-Day Monte Carlo Prediction Dashboard
  Quant-Grade | Bloomberg Terminal Style | Dark Theme
  
  Author  : Senior Quantitative Analyst
  Version : 2.0.0
  
  Data Sources:
    - yfinance  → NIFTY 50, S&P 500, NASDAQ 100, DJIA, FTSE 100,
                   DAX, Nikkei 225, Hang Seng, Shanghai Composite
    - NewsAPI   → Latest global financial headlines
                  (fallback: curated mock headlines if key absent)
  
  Pipeline:
    Step 1 → Data Preparation  (prices, log-returns, alignment)
    Step 2 → Global Market Signal (weighted factor, normalised)
    Step 3 → News Sentiment Analysis (VADER / TextBlob)
    Step 4 → Adjust GBM Drift
    Step 5 → Monte Carlo Simulation (GBM, 10 000 paths)
    Step 6 → Prediction Statistics & Probabilities
    Step 7 → Self-Contained HTML Dashboard (4 sections, Plotly dark)

  Dependencies (install once):
    pip install yfinance pandas numpy scipy requests
                plotly textblob vaderSentiment
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, json, math, warnings, datetime, textwrap
import numpy  as np
import pandas as pd
import requests

# ── scipy (KDE for Chart 4) ───────────────────────────────────────────────────
try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    gaussian_kde    = None          # handled gracefully in step7

warnings.filterwarnings("ignore")

# ── Optional rich sentiment library (VADER preferred, TextBlob fallback) ─────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# ── yfinance ──────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance not installed.  Run:  pip install yfinance")
    sys.exit(1)

# ── Plotly ────────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    print("[ERROR] plotly not installed.  Run:  pip install plotly")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")   # set env var or paste key
N_SIMULATIONS  = 10_000
LOOKBACK_YEARS = 5
OUTPUT_HTML    = "nifty50_montecarlo_dashboard.html"

# ─────────────────────────────────────────────────────────────────────────────
# TICKER MAP
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_TICKER = "^NSEI"

GLOBAL_INDICES = {
    "S&P 500"             : "^GSPC",
    "NASDAQ 100"          : "^NDX",
    "Dow Jones"           : "^DJI",
    "FTSE 100"            : "^FTSE",
    "DAX"                 : "^GDAXI",
    "Nikkei 225"          : "^N225",
    "Hang Seng"           : "^HSI",
    "Shanghai Composite"  : "000001.SS",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]  {msg}")

def safe_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV; return Close series.  Retry once on failure."""
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("empty frame")
        close = df["Close"].squeeze()
        close.name = ticker
        return close.dropna()
    except Exception as exc:
        log(f"  ⚠  {ticker} fetch failed: {exc}")
        return pd.Series(dtype=float, name=ticker)

def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def step1_fetch_data() -> dict:
    log("STEP 1 ▸ Fetching market data …")

    end   = datetime.date.today().strftime("%Y-%m-%d")
    start = (datetime.date.today() -
             datetime.timedelta(days=365 * LOOKBACK_YEARS + 30)).strftime("%Y-%m-%d")

    # ── NIFTY 50 ──────────────────────────────────────────────────────────────
    log(f"  → NIFTY 50  ({NIFTY_TICKER})")
    nifty_close = safe_fetch(NIFTY_TICKER, start, end)
    if nifty_close.empty:
        log("  [FATAL] Cannot fetch NIFTY 50.  Check internet / yfinance.")
        sys.exit(1)

    # ── GLOBAL INDICES ────────────────────────────────────────────────────────
    global_closes = {}
    for name, ticker in GLOBAL_INDICES.items():
        log(f"  → {name:25s} ({ticker})")
        s = safe_fetch(ticker, start, end)
        if not s.empty:
            global_closes[name] = s

    # ── Align on common business dates ───────────────────────────────────────
    all_series = [nifty_close] + list(global_closes.values())
    combined   = pd.concat(all_series, axis=1).dropna(how="all")
    combined.index = pd.to_datetime(combined.index)
    combined.sort_index(inplace=True)

    nifty_aligned  = combined.iloc[:, 0].dropna()
    global_aligned = {
        name: combined[ser.name].dropna()
        for name, ser in zip(global_closes.keys(), global_closes.values())
        if ser.name in combined.columns
    }

    log(f"  ✓ NIFTY rows: {len(nifty_aligned)} | "
        f"global indices loaded: {len(global_aligned)}")

    return {
        "nifty"         : nifty_aligned,
        "nifty_returns" : log_returns(nifty_aligned),
        "global_closes" : global_aligned,
        "global_returns": {k: log_returns(v) for k, v in global_aligned.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GLOBAL MARKET SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
def step2_global_signal(data: dict) -> dict:
    log("STEP 2 ▸ Computing global market signal …")

    prev_returns = {}
    for name, ret in data["global_returns"].items():
        if len(ret) >= 2:
            prev_returns[name] = float(ret.iloc[-1])   # most-recent log return

    if not prev_returns:
        log("  ⚠  No global returns available.  Signal = 0.")
        return {"global_factor": 0.0, "prev_returns": {}}

    # Equal-weighted average (simple; can extend to risk-parity weights)
    raw_factor   = np.mean(list(prev_returns.values()))

    # Normalise by cross-sectional std so factor is Z-score–like
    cross_std    = np.std(list(prev_returns.values())) or 1e-9
    norm_factor  = raw_factor / cross_std

    # Soft-cap at ±3σ
    norm_factor  = float(np.clip(norm_factor, -3, 3))

    log(f"  ✓ Raw factor: {raw_factor:.6f} | "
        f"Normalised factor: {norm_factor:.4f}")
    for n, r in prev_returns.items():
        log(f"      {n:25s} prev-day return: {r*100:+.3f}%")

    return {"global_factor": norm_factor, "prev_returns": prev_returns}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — NEWS SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
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

def fetch_headlines(api_key: str) -> list[str]:
    """Fetch from NewsAPI; fall back to mock headlines."""
    if not api_key:
        log("  ℹ  NEWS_API_KEY not set — using curated mock headlines.")
        return MOCK_HEADLINES

    url = (
        "https://newsapi.org/v2/everything"
        "?q=stock+market+india+nifty+economy+finance"
        "&sortBy=publishedAt"
        "&language=en"
        "&pageSize=20"
        f"&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        headlines = [a["title"] for a in articles if a.get("title")]
        if not headlines:
            raise ValueError("no headlines returned")
        log(f"  ✓ Fetched {len(headlines)} live headlines from NewsAPI.")
        return headlines
    except Exception as exc:
        log(f"  ⚠  NewsAPI error: {exc}  → using mock headlines.")
        return MOCK_HEADLINES

def analyse_sentiment(headlines: list[str]) -> dict:
    """Score each headline; return aggregate score in [−1, +1]."""
    scores = []

    if VADER_AVAILABLE:
        analyser = SentimentIntensityAnalyzer()
        for h in headlines:
            s = analyser.polarity_scores(h)["compound"]   # already [−1, +1]
            scores.append(s)
        method = "VADER"
    elif TEXTBLOB_AVAILABLE:
        for h in headlines:
            s = TextBlob(h).sentiment.polarity             # [−1, +1]
            scores.append(s)
        method = "TextBlob"
    else:
        # Lightweight keyword-based fallback
        POS = {"bull", "surges", "rally", "gains", "upgrade",
               "positive", "beats", "strong", "growth", "rise",
               "buys", "buyers", "stimulus", "optimism", "boom"}
        NEG = {"bear", "slips", "falls", "decline", "cut", "downgrade",
               "pressure", "inflation", "risk", "fear", "sticky",
               "uncertainty", "recession", "selloff", "plunge"}
        for h in headlines:
            words = set(h.lower().split())
            pos_c = len(words & POS)
            neg_c = len(words & NEG)
            total = pos_c + neg_c or 1
            scores.append((pos_c - neg_c) / total)
        method = "Keyword fallback"

    agg_score = float(np.mean(scores)) if scores else 0.0
    agg_score = float(np.clip(agg_score, -1, 1))          # ensure bounds

    log(f"  ✓ Sentiment [{method}]: {agg_score:+.4f}  "
        f"(headlines analysed: {len(scores)})")

    return {
        "score"     : agg_score,
        "scores"    : scores,
        "headlines" : headlines,
        "method"    : method,
    }

def step3_sentiment() -> dict:
    log("STEP 3 ▸ Fetching and analysing news sentiment …")
    headlines = fetch_headlines(NEWS_API_KEY)
    return analyse_sentiment(headlines)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ADJUST DRIFT
# ─────────────────────────────────────────────────────────────────────────────
def step4_adjust_drift(data: dict, signal: dict,
                       sentiment: dict) -> dict:
    log("STEP 4 ▸ Adjusting GBM drift …")

    rets   = data["nifty_returns"].dropna()
    mu     = float(rets.mean())
    sigma  = float(rets.std())

    gf     = signal["global_factor"]
    ns     = sentiment["score"]

    adj_mu = mu + (gf * 0.3) + (ns * 0.2)

    ann_vol   = sigma * math.sqrt(252) * 100
    ann_mu    = mu    * 252  * 100
    ann_adjmu = adj_mu * 252 * 100

    log(f"  ✓ mu={mu:.6f}  sigma={sigma:.6f}")
    log(f"  ✓ Adjusted mu={adj_mu:.6f}  "
        f"(global_factor={gf:+.4f}, news={ns:+.4f})")
    log(f"  ✓ Annualised vol={ann_vol:.2f}%  "
        f"baseline drift={ann_mu:.2f}%  "
        f"adjusted drift={ann_adjmu:.2f}%")

    return {
        "mu"        : mu,
        "sigma"     : sigma,
        "adj_mu"    : adj_mu,
        "ann_vol"   : ann_vol,
        "ann_mu"    : ann_mu,
        "ann_adjmu" : ann_adjmu,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MONTE CARLO SIMULATION  (GBM)
# ─────────────────────────────────────────────────────────────────────────────
def step5_simulate(data: dict, params: dict,
                   n_sims: int = N_SIMULATIONS) -> dict:
    log(f"STEP 5 ▸ Running Monte Carlo  ({n_sims:,} paths) …")

    S0    = float(data["nifty"].iloc[-1])
    mu    = params["adj_mu"]
    sigma = params["sigma"]

    rng   = np.random.default_rng(seed=42)
    Z     = rng.standard_normal(n_sims)

    # GBM single-step
    drift     = (mu - 0.5 * sigma ** 2)
    S_next    = S0 * np.exp(drift + sigma * Z)
    sim_rets  = np.log(S_next / S0)

    log(f"  ✓ S0={S0:,.2f}  "
        f"E[S_next]={S_next.mean():,.2f}  "
        f"σ[S_next]={S_next.std():,.2f}")

    return {
        "S0"       : S0,
        "S_next"   : S_next,
        "sim_rets" : sim_rets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PREDICTION STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def step6_statistics(sim: dict) -> dict:
    log("STEP 6 ▸ Computing prediction statistics …")

    S0     = sim["S0"]
    S_next = sim["S_next"]

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
    changes = {k: (v / S0 - 1) * 100 for k, v in pcts.items()}

    probs = {
        "up"          : float(np.mean(S_next > S0)         * 100),
        "down"        : float(np.mean(S_next < S0)         * 100),
        "above_p05"   : float(np.mean(S_next > S0 * 1.005) * 100),
        "below_m05"   : float(np.mean(S_next < S0 * 0.995) * 100),
    }

    for k, v in pcts.items():
        log(f"  {k:8s} → {v:>10,.2f}  ({changes[k]:+.3f}%)")
    log(f"  P(up)={probs['up']:.1f}%  "
        f"P(down)={probs['down']:.1f}%  "
        f"P(>+0.5%)={probs['above_p05']:.1f}%  "
        f"P(<-0.5%)={probs['below_m05']:.1f}%")

    return {"percentiles": pcts, "changes": changes, "probs": probs}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BUILD HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def _pct_bar(value: float, low_green: bool = False) -> str:
    """Mini progress bar HTML for probability cards."""
    colour = "#00e5b0" if (value >= 50) != low_green else "#ff4560"
    return (
        f'<div style="height:6px;background:#1e2a3a;border-radius:3px;'
        f'margin-top:8px;">'
        f'<div style="height:100%;width:{value:.1f}%;'
        f'background:{colour};border-radius:3px;'
        f'transition:width .5s ease;"></div></div>'
    )

def _interp_colour(value: float, v_min: float,
                   v_max: float, invert: bool = False) -> str:
    """Interpolate between red→amber→green for a heatmap effect."""
    if v_max == v_min:
        t = 0.5
    else:
        t = (value - v_min) / (v_max - v_min)
    if invert:
        t = 1 - t
    if t < 0.5:
        r = 255;  g = int(t * 2 * 200);  b = 60
    else:
        r = int((1 - t) * 2 * 255);  g = 200;  b = 60
    return f"rgb({r},{g},{b})"

def step7_build_dashboard(
        data      : dict,
        params    : dict,
        signal    : dict,
        sentiment : dict,
        sim       : dict,
        stats     : dict,
) -> str:

    log("STEP 7 ▸ Building Bloomberg-style HTML dashboard …")

    S0      = sim["S0"]
    S_next  = sim["S_next"]
    pcts    = stats["percentiles"]
    chgs    = stats["changes"]
    probs   = stats["probs"]

    last_date = str(data["nifty"].index[-1].date())
    today     = datetime.date.today().strftime("%d %b %Y")

    # ── 100 sample paths for the fan chart ───────────────────────────────────
    n_paths  = 100
    rng2     = np.random.default_rng(seed=99)
    Z_fan    = rng2.standard_normal((n_paths, 5))
    dt       = 1/252
    mu_step  = params["adj_mu"]
    sig_step = params["sigma"]
    fan_paths= []
    for i in range(n_paths):
        path = [S0]
        for z in Z_fan[i]:
            path.append(path[-1] * np.exp(
                (mu_step - 0.5*sig_step**2)*dt + sig_step*np.sqrt(dt)*z))
        fan_paths.append(path)

    # ── Plotly traces as JSON ─────────────────────────────────────────────────
    def fig_to_json(fig) -> str:
        return pio.to_json(fig)

    # ── Chart 1: Sample paths ─────────────────────────────────────────────────
    fig1 = go.Figure()
    x_axis = list(range(6))
    for i, path in enumerate(fan_paths):
        opacity = 0.15 if i > 0 else 0.9
        width   = 0.6  if i > 0 else 1.5
        colour  = "#00e5b0" if path[-1] >= S0 else "#ff4560"
        fig1.add_trace(go.Scatter(
            x=x_axis, y=path,
            mode="lines",
            line=dict(color=colour, width=width),
            opacity=opacity,
            showlegend=False,
            hoverinfo="skip",
        ))
    fig1.add_hline(y=S0, line_dash="dash",
                   line_color="#f0c040", line_width=1.5,
                   annotation_text=f"  Current: {S0:,.2f}",
                   annotation_font_color="#f0c040")
    fig1.update_layout(
        title=dict(text="Monte Carlo Sample Paths (100 shown)",
                   font=dict(color="#e0e6f0", size=14)),
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f35",
        xaxis=dict(title="Days", color="#7a8fa6",
                   gridcolor="#162033", tickvals=[0,1,2,3,4,5]),
        yaxis=dict(title="NIFTY Level", color="#7a8fa6",
                   gridcolor="#162033",
                   tickformat=",.0f"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=340,
    )

    # ── Chart 2: Histogram of next-day simulated prices ───────────────────────
    hist_colours = ["#ff4560" if v < S0 else "#00e5b0" for v in S_next]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=S_next,
        nbinsx=120,
        marker_color="#00e5b0",
        opacity=0.8,
        name="Simulated Prices",
    ))
    for label, pct_val, col in [
        ("P5",  pcts["P5"],  "#ff4560"),
        ("P25", pcts["P25"], "#f0c040"),
        ("Med", pcts["Median"], "#00c8ff"),
        ("P75", pcts["P75"], "#f0c040"),
        ("P95", pcts["P95"], "#ff4560"),
    ]:
        fig2.add_vline(x=pct_val, line_color=col,
                       line_dash="dot", line_width=1.5,
                       annotation_text=f" {label}",
                       annotation_font_color=col,
                       annotation_font_size=11)
    fig2.add_vline(x=S0, line_color="#ffffff",
                   line_dash="dash", line_width=2,
                   annotation_text="  Current",
                   annotation_font_color="#ffffff")
    fig2.update_layout(
        title=dict(text="Distribution of Simulated Next-Day Prices",
                   font=dict(color="#e0e6f0", size=14)),
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f35",
        xaxis=dict(title="Price Level", color="#7a8fa6",
                   gridcolor="#162033", tickformat=",.0f"),
        yaxis=dict(title="Frequency", color="#7a8fa6",
                   gridcolor="#162033"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=340,
        bargap=0.02,
    )

    # ── Chart 3: Confidence band (fan using percentile envelope) ─────────────
    x_conf = [0, 1]
    fig3 = go.Figure()
    bands = [
        (1, 99, "rgba(255,69,96,0.12)",  "P1–P99"),
        (5, 95, "rgba(240,192,64,0.18)", "P5–P95"),
        (25, 75, "rgba(0,200,255,0.25)", "P25–P75"),
    ]
    for lo, hi, fill, name in bands:
        lo_v = float(np.percentile(S_next, lo))
        hi_v = float(np.percentile(S_next, hi))
        fig3.add_trace(go.Scatter(
            x=[0, 1, 1, 0],
            y=[S0, hi_v, lo_v, S0],
            fill="toself",
            fillcolor=fill,
            line=dict(width=0),
            name=name,
            mode="lines",
        ))
    for label, val, col in [
        ("P5",  pcts["P5"],  "#ff4560"),
        ("P25", pcts["P25"], "#f0c040"),
        ("Median", pcts["Median"], "#00c8ff"),
        ("Mean", pcts["Mean"], "#7b61ff"),
        ("P75", pcts["P75"], "#f0c040"),
        ("P95", pcts["P95"], "#ff4560"),
    ]:
        fig3.add_trace(go.Scatter(
            x=[0, 1], y=[S0, val],
            mode="lines+markers",
            line=dict(color=col, width=1.4, dash="dot"),
            marker=dict(size=7, color=col),
            name=label,
        ))
    fig3.update_layout(
        title=dict(text="Confidence Interval Bands (Today → Tomorrow)",
                   font=dict(color="#e0e6f0", size=14)),
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f35",
        xaxis=dict(title="", tickvals=[0,1],
                   ticktext=["Today", "Next Day"],
                   color="#7a8fa6", gridcolor="#162033"),
        yaxis=dict(title="NIFTY Level", color="#7a8fa6",
                   gridcolor="#162033", tickformat=",.0f"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=340,
        legend=dict(bgcolor="#0a1628", font=dict(color="#a0aec0", size=11)),
    )

    # ── Chart 4: KDE / Distribution with percentile markers ──────────────────
    kde_x = np.linspace(S_next.min(), S_next.max(), 500)
    if SCIPY_AVAILABLE and gaussian_kde is not None:
        _kde_obj = gaussian_kde(S_next)
        kde_y    = _kde_obj(kde_x)
        def kde(x):   # wrapper so kde(S0) returns array
            return _kde_obj(np.atleast_1d(x))
    else:
        # Pure-numpy histogram density (no scipy required)
        _counts, _edges = np.histogram(S_next, bins=200, density=True)
        _centres        = (_edges[:-1] + _edges[1:]) / 2
        kde_y           = np.interp(kde_x, _centres, _counts)
        def kde(x):
            return np.interp(np.atleast_1d(x), _centres, _counts)

    fig4 = go.Figure()
    # Colour the area: red below S0, green above
    mask_up   = kde_x >= S0
    mask_down = kde_x <  S0
    fig4.add_trace(go.Scatter(
        x=np.concatenate([[S0], kde_x[mask_up]]),
        y=np.concatenate([[kde(S0)[0]], kde_y[mask_up]]),
        fill="tozeroy",
        fillcolor="rgba(0,229,176,0.22)",
        line=dict(color="#00e5b0", width=1.5),
        name="Upside",
    ))
    fig4.add_trace(go.Scatter(
        x=np.concatenate([kde_x[mask_down], [S0]]),
        y=np.concatenate([kde_y[mask_down], [kde(S0)[0]]]),
        fill="tozeroy",
        fillcolor="rgba(255,69,96,0.22)",
        line=dict(color="#ff4560", width=1.5),
        name="Downside",
    ))
    pct_markers = {
        "P1": ("#ff4560", pcts["P1"]),
        "P5": ("#ff7c96", pcts["P5"]),
        "P25":("#f0c040", pcts["P25"]),
        "Median":("#00c8ff", pcts["Median"]),
        "Mean":("#7b61ff", pcts["Mean"]),
        "P75":("#f0c040", pcts["P75"]),
        "P95":("#ff7c96", pcts["P95"]),
        "P99":("#ff4560", pcts["P99"]),
    }
    for lbl, (col, val) in pct_markers.items():
        fig4.add_vline(x=val, line_color=col,
                       line_dash="dot", line_width=1.2,
                       annotation_text=f" {lbl}",
                       annotation_font_color=col,
                       annotation_font_size=10)
    fig4.add_vline(x=S0, line_color="#ffffff",
                   line_width=2, line_dash="dash",
                   annotation_text="  Current",
                   annotation_font_color="#ffffff")
    fig4.update_layout(
        title=dict(text="KDE Distribution with Percentile Markers",
                   font=dict(color="#e0e6f0", size=14)),
        template="plotly_dark",
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f35",
        xaxis=dict(title="Price Level", color="#7a8fa6",
                   gridcolor="#162033", tickformat=",.0f"),
        yaxis=dict(title="Density", color="#7a8fa6",
                   gridcolor="#162033"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=340,
        legend=dict(bgcolor="#0a1628", font=dict(color="#a0aec0", size=11)),
    )

    # ── Convert figures to JSON ───────────────────────────────────────────────
    j1 = fig_to_json(fig1)
    j2 = fig_to_json(fig2)
    j3 = fig_to_json(fig3)
    j4 = fig_to_json(fig4)

    # ── Global returns table rows ─────────────────────────────────────────────
    global_rows_html = ""
    for name, ret in signal["prev_returns"].items():
        colour = "#00e5b0" if ret >= 0 else "#ff4560"
        sign   = "▲" if ret >= 0 else "▼"
        global_rows_html += (
            f'<tr>'
            f'<td style="color:#c0cfe0;padding:6px 10px;">{name}</td>'
            f'<td style="color:{colour};padding:6px 10px;font-weight:600;text-align:right;">'
            f'{sign} {abs(ret)*100:.3f}%</td>'
            f'</tr>'
        )

    # ── Percentile table rows ─────────────────────────────────────────────────
    pct_rows_html = ""
    prices = list(pcts.values())
    p_min, p_max = min(prices), max(prices)
    for lbl in ["P1","P5","P25","Median","Mean","P75","P95","P99"]:
        price  = pcts[lbl]
        change = chgs[lbl]
        col    = _interp_colour(price, p_min, p_max)
        chg_col= "#00e5b0" if change >= 0 else "#ff4560"
        pct_rows_html += (
            f'<tr style="border-bottom:1px solid #1a2a40;">'
            f'<td style="padding:9px 14px;color:#a0b4c8;font-weight:600;">{lbl}</td>'
            f'<td style="padding:9px 14px;color:{col};font-weight:700;'
            f'text-align:right;font-size:15px;">{price:,.2f}</td>'
            f'<td style="padding:9px 14px;color:{chg_col};font-weight:600;'
            f'text-align:right;">{change:+.3f}%</td>'
            f'</tr>'
        )

    # ── Probability cards ─────────────────────────────────────────────────────
    prob_cards = [
        ("Price UP", probs["up"],       False, "🔼"),
        ("Price DOWN", probs["down"],   True,  "🔽"),
        ("Move > +0.5%", probs["above_p05"], False, "🚀"),
        ("Move < −0.5%", probs["below_m05"], True,  "📉"),
    ]
    prob_html = ""
    for title, val, invert, icon in prob_cards:
        colour = "#00e5b0" if (val >= 50) != invert else "#ff4560"
        prob_html += f"""
        <div class="prob-card">
          <div class="prob-icon">{icon}</div>
          <div class="prob-title">{title}</div>
          <div class="prob-value" style="color:{colour};">{val:.1f}%</div>
          {_pct_bar(val, invert)}
        </div>"""

    # ── Interpretation ────────────────────────────────────────────────────────
    mean_chg   = chgs["Mean"]
    gf         = signal["global_factor"]
    ns_score   = sentiment["score"]
    ann_vol    = params["ann_vol"]

    bias_str = ("bullish 📈" if mean_chg >  0.15
                else "bearish 📉" if mean_chg < -0.15
                else "neutral ⚖️")
    vol_str  = ("elevated" if ann_vol > 22
                else "moderate" if ann_vol > 14
                else "subdued")
    gf_str   = ("positive global tailwinds" if gf > 0.5
                else "negative global headwinds" if gf < -0.5
                else "mixed global cues")
    ns_str   = ("bullish news sentiment" if ns_score > 0.1
                else "bearish news sentiment" if ns_score < -0.1
                else "neutral news backdrop")

    interpretation = (
        f"The model carries a <strong>{bias_str}</strong> bias for tomorrow's NIFTY session, "
        f"with the mean simulated return of <strong>{mean_chg:+.3f}%</strong> reflecting an "
        f"adjusted drift driven by global and sentiment inputs. "
        f"Annualised volatility stands at <strong>{ann_vol:.2f}%</strong>, "
        f"indicating <strong>{vol_str}</strong> conditions with the 5th–95th percentile "
        f"price band spanning "
        f"<strong>{pcts['P5']:,.0f}–{pcts['P95']:,.0f}</strong>. "
        f"Global markets present <strong>{gf_str}</strong> "
        f"(normalised factor: {gf:+.3f}) while the news flow reflects "
        f"<strong>{ns_str}</strong> (score: {ns_score:+.3f}), "
        f"together contributing {((gf*0.3)+(ns_score*0.2))*252*100:+.2f}% "
        f"annualised drift adjustment above the historical baseline."
    )

    # ── Sentiment gauge ───────────────────────────────────────────────────────
    ns_pct   = (ns_score + 1) / 2 * 100
    ns_col   = "#00e5b0" if ns_score >= 0 else "#ff4560"
    gf_pct   = (gf + 3) / 6 * 100
    gf_col   = "#00e5b0" if gf >= 0 else "#ff4560"

    # ── Headline snippets ─────────────────────────────────────────────────────
    top_headlines = sentiment["headlines"][:6]
    headline_items = "".join(
        f'<li style="padding:5px 0;color:#8a9bb0;border-bottom:1px solid #1a2a40;'
        f'font-size:12px;">{h}</li>'
        for h in top_headlines
    )

    # ──────────────────────────────────────────────────────────────────────────
    #  HTML TEMPLATE
    # ──────────────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>NIFTY 50 — Monte Carlo Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  /* ── Reset & Base ── */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ font-size: 14px; }}
  body {{
    background: #050e1c;
    color: #c8d8e8;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.5;
    min-height: 100vh;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #071428 0%, #0d1f38 50%, #071a30 100%);
    border-bottom: 1px solid #1a3050;
    padding: 18px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .header-left {{ display: flex; align-items: center; gap: 16px; }}
  .terminal-badge {{
    background: #ff4560;
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 3px 8px;
    border-radius: 3px;
    text-transform: uppercase;
  }}
  .header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: #e8f0ff;
    letter-spacing: 0.5px;
  }}
  .header h1 span {{ color: #00e5b0; }}
  .header-right {{ text-align: right; }}
  .header-right .date {{ color: #5a7a9a; font-size: 12px; }}
  .header-right .price-big {{
    font-size: 28px;
    font-weight: 700;
    color: #00e5b0;
    font-variant-numeric: tabular-nums;
  }}

  /* ── Layout ── */
  .container {{ max-width: 1600px; margin: 0 auto; padding: 24px 28px; }}
  .section-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #3a7bd5;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1a3050, transparent);
  }}
  .section {{ margin-bottom: 32px; }}

  /* ── Cards ── */
  .card {{
    background: #0a1628;
    border: 1px solid #1a2f4a;
    border-radius: 8px;
    padding: 18px 22px;
  }}
  .card-title {{
    font-size: 11px;
    color: #4a6a8a;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px;
  }}
  .card-value {{
    font-size: 20px;
    font-weight: 700;
    color: #e0ecff;
    font-variant-numeric: tabular-nums;
  }}
  .card-sub {{ font-size: 12px; color: #5a7a9a; margin-top: 2px; }}

  /* ── Section A grid ── */
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 14px;
  }}

  /* ── Section B table ── */
  .pct-table {{
    width: 100%;
    border-collapse: collapse;
    background: #0a1628;
    border: 1px solid #1a2f4a;
    border-radius: 8px;
    overflow: hidden;
  }}
  .pct-table th {{
    background: #0f1e35;
    color: #4a7aaa;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid #1a3050;
  }}
  .pct-table th:nth-child(2),
  .pct-table th:nth-child(3) {{ text-align: right; }}
  .pct-table tr:hover {{ background: #0d1e33; }}

  /* ── Section C probability cards ── */
  .prob-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }}
  .prob-card {{
    background: #0a1628;
    border: 1px solid #1a2f4a;
    border-radius: 8px;
    padding: 22px 20px;
    text-align: center;
    transition: border-color .2s;
  }}
  .prob-card:hover {{ border-color: #2a4a6a; }}
  .prob-icon {{ font-size: 28px; margin-bottom: 8px; }}
  .prob-title {{
    font-size: 12px;
    color: #5a7a9a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }}
  .prob-value {{
    font-size: 36px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}

  /* ── Section D charts grid ── */
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  .chart-box {{
    background: #0a1628;
    border: 1px solid #1a2f4a;
    border-radius: 8px;
    overflow: hidden;
    padding: 6px;
  }}

  /* ── Global returns side panel ── */
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 16px;
    align-items: start;
  }}
  .global-table {{
    width: 100%;
    border-collapse: collapse;
  }}
  .global-table th {{
    background: #0f1e35;
    color: #4a7aaa;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid #1a3050;
  }}
  .global-table tr:hover {{ background: #0d1e33; }}

  /* ── Sentiment gauges ── */
  .gauge-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 12px;
  }}
  .gauge-item {{ background: #0a1628; border: 1px solid #1a2f4a; border-radius: 8px; padding: 14px 18px; }}
  .gauge-label {{ font-size: 11px; color: #4a6a8a; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
  .gauge-val {{ font-size: 18px; font-weight: 700; }}
  .gauge-bar {{ height: 8px; background: #1e2a3a; border-radius: 4px; margin-top: 6px; }}
  .gauge-fill {{ height: 100%; border-radius: 4px; transition: width .5s; }}

  /* ── Headlines ── */
  .headline-list {{ list-style: none; padding: 12px 0 0; }}

  /* ── Interpretation ── */
  .interp-box {{
    background: #071428;
    border: 1px solid #1a3a5a;
    border-left: 4px solid #3a7bd5;
    border-radius: 6px;
    padding: 18px 22px;
    font-size: 13px;
    line-height: 1.8;
    color: #a0bcd8;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    padding: 20px;
    font-size: 11px;
    color: #2a4a6a;
    border-top: 1px solid #0f1e30;
    margin-top: 24px;
  }}

  /* ── Responsive ── */
  @media (max-width: 900px) {{
    .prob-grid {{ grid-template-columns: repeat(2,1fr); }}
    .charts-grid {{ grid-template-columns: 1fr; }}
    .two-col {{ grid-template-columns: 1fr; }}
    .summary-grid {{ grid-template-columns: repeat(2,1fr); }}
  }}
</style>
</head>
<body>

<!-- ═══════════ HEADER ═══════════ -->
<div class="header">
  <div class="header-left">
    <span class="terminal-badge">LIVE SIM</span>
    <h1>NIFTY 50 <span>Monte Carlo</span> Prediction Dashboard</h1>
  </div>
  <div class="header-right">
    <div class="date">Last Data: {last_date} &nbsp;|&nbsp; Generated: {today}</div>
    <div class="price-big">₹ {S0:,.2f}</div>
  </div>
</div>

<div class="container">

<!-- ═══════════ SECTION A — MARKET SUMMARY ═══════════ -->
<div class="section">
  <div class="section-label">A — Market Summary</div>
  <div class="summary-grid">
    <div class="card">
      <div class="card-title">Current NIFTY Price</div>
      <div class="card-value" style="color:#00e5b0;">₹ {S0:,.2f}</div>
      <div class="card-sub">Last Close · {last_date}</div>
    </div>
    <div class="card">
      <div class="card-title">Annualised Volatility</div>
      <div class="card-value" style="color:#f0c040;">{params['ann_vol']:.2f}%</div>
      <div class="card-sub">σ × √252 · {LOOKBACK_YEARS}yr history</div>
    </div>
    <div class="card">
      <div class="card-title">Baseline Drift (μ)</div>
      <div class="card-value" style="color:{'#00e5b0' if params['ann_mu']>=0 else '#ff4560'};">{params['ann_mu']:+.2f}%</div>
      <div class="card-sub">Annualised mean daily return</div>
    </div>
    <div class="card">
      <div class="card-title">Adjusted Drift</div>
      <div class="card-value" style="color:{'#00e5b0' if params['ann_adjmu']>=0 else '#ff4560'};">{params['ann_adjmu']:+.2f}%</div>
      <div class="card-sub">After global + sentiment signals</div>
    </div>
    <div class="card">
      <div class="card-title">Global Market Factor</div>
      <div class="card-value" style="color:{gf_col};">{signal['global_factor']:+.4f}</div>
      <div class="card-sub">Normalised Z-score (±3 cap)</div>
      <div class="gauge-bar" style="margin-top:8px;">
        <div class="gauge-fill" style="width:{gf_pct:.1f}%;background:{gf_col};"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">News Sentiment Score</div>
      <div class="card-value" style="color:{ns_col};">{ns_score:+.4f}</div>
      <div class="card-sub">{sentiment['method']} · {len(sentiment['headlines'])} headlines</div>
      <div class="gauge-bar" style="margin-top:8px;">
        <div class="gauge-fill" style="width:{ns_pct:.1f}%;background:{ns_col};"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Simulation Paths</div>
      <div class="card-value" style="color:#7b61ff;">{N_SIMULATIONS:,}</div>
      <div class="card-sub">GBM · 1-step · seed 42</div>
    </div>
    <div class="card">
      <div class="card-title">Drift Adjustment</div>
      <div class="card-value" style="color:#00c8ff;">{((signal['global_factor']*0.3)+(sentiment['score']*0.2))*252*100:+.2f}%</div>
      <div class="card-sub">Global(×0.3) + Sentiment(×0.2)</div>
    </div>
  </div>
</div>

<!-- ═══════════ SECTION B — SIMULATION RESULTS ═══════════ -->
<div class="section">
  <div class="section-label">B — Simulation Results (10,000 Paths)</div>
  <div class="two-col">
    <div>
      <table class="pct-table">
        <thead>
          <tr>
            <th>Percentile</th>
            <th style="text-align:right;">Predicted Price</th>
            <th style="text-align:right;">Change %</th>
          </tr>
        </thead>
        <tbody>
          {pct_rows_html}
        </tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title" style="margin-bottom:10px;">Global Index — Previous Day Returns</div>
      <table class="global-table">
        <thead>
          <tr>
            <th>Index</th>
            <th style="text-align:right;">Return</th>
          </tr>
        </thead>
        <tbody>
          {global_rows_html}
        </tbody>
      </table>
      <div class="gauge-row">
        <div class="gauge-item">
          <div class="gauge-label">News Sentiment</div>
          <div class="gauge-val" style="color:{ns_col};">{ns_score:+.4f}</div>
          <div class="gauge-bar"><div class="gauge-fill" style="width:{ns_pct:.1f}%;background:{ns_col};"></div></div>
        </div>
        <div class="gauge-item">
          <div class="gauge-label">Top Headlines</div>
          <ul class="headline-list">{headline_items}</ul>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════ SECTION C — PROBABILITY GRID ═══════════ -->
<div class="section">
  <div class="section-label">C — Probability Grid</div>
  <div class="prob-grid">
    {prob_html}
  </div>
</div>

<!-- ═══════════ SECTION D — CHARTS ═══════════ -->
<div class="section">
  <div class="section-label">D — Charts (Plotly Dark)</div>
  <div class="charts-grid">
    <div class="chart-box">
      <div id="chart1"></div>
    </div>
    <div class="chart-box">
      <div id="chart2"></div>
    </div>
    <div class="chart-box">
      <div id="chart3"></div>
    </div>
    <div class="chart-box">
      <div id="chart4"></div>
    </div>
  </div>
</div>

<!-- ═══════════ INTERPRETATION ═══════════ -->
<div class="section">
  <div class="section-label">Quantitative Interpretation</div>
  <div class="interp-box">
    {interpretation}
  </div>
</div>

</div><!-- /container -->

<div class="footer">
  NIFTY 50 Monte Carlo Dashboard &nbsp;·&nbsp;
  Powered by yfinance · Plotly · GBM Simulation &nbsp;·&nbsp;
  <strong style="color:#3a7bd5;">For Educational &amp; Research Use Only — Not Investment Advice</strong>
</div>

<!-- ═══════════ PLOTLY RENDER ═══════════ -->
<script>
  const cfg = {{responsive:true, displayModeBar:false}};
  Plotly.newPlot('chart1', {j1}.replace('"config":', '"_config":'));
  Plotly.newPlot('chart2', {j2}.replace('"config":', '"_config":'));
  Plotly.newPlot('chart3', {j3}.replace('"config":', '"_config":'));
  Plotly.newPlot('chart4', {j4}.replace('"config":', '"_config":'));
</script>
<script>
(function(){{
  // Parse and render each figure properly
  const figs = [
    {j1},
    {j2},
    {j3},
    {j4}
  ];
  const divs = ['chart1','chart2','chart3','chart4'];
  figs.forEach((fig, i) => {{
    Plotly.react(divs[i], fig.data, fig.layout, {{responsive:true, displayModeBar:false}});
  }});
}})();
</script>

</body>
</html>"""

    log(f"  ✓ HTML built — length: {len(html):,} chars")
    return html


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    separator = "=" * 72
    print(separator)
    print("  NIFTY 50  ·  Monte Carlo Prediction Dashboard  ·  v2.0")
    print(separator)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    data      = step1_fetch_data()
    signal    = step2_global_signal(data)
    sentiment = step3_sentiment()
    params    = step4_adjust_drift(data, signal, sentiment)
    sim       = step5_simulate(data, params)
    stats     = step6_statistics(sim)
    html      = step7_build_dashboard(
                    data, params, signal, sentiment, sim, stats)

    # ── Write output ──────────────────────────────────────────────────────────
    with open(OUTPUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(separator)
    log(f"✅  Dashboard saved → {OUTPUT_HTML}")
    log(f"    Open in any modern browser (no server required).")
    print(separator)

    # ── Quick summary to console ──────────────────────────────────────────────
    S0   = sim["S0"]
    pcts = stats["percentiles"]
    prbs = stats["probs"]
    print(f"""
  ┌─── QUICK SUMMARY ───────────────────────────────────────────────┐
  │  Current Price  :  ₹ {S0:>10,.2f}                                │
  │  Mean Sim Price :  ₹ {pcts['Mean']:>10,.2f}   ({stats['changes']['Mean']:+.3f}%)          │
  │  Median         :  ₹ {pcts['Median']:>10,.2f}   ({stats['changes']['Median']:+.3f}%)          │
  │  P5 – P95 Range :  ₹ {pcts['P5']:,.0f}  –  ₹{pcts['P95']:,.0f}                │
  │  P(Up)          :  {prbs['up']:>5.1f}%                                    │
  │  P(Down)        :  {prbs['down']:>5.1f}%                                    │
  │  P(>+0.5%)      :  {prbs['above_p05']:>5.1f}%                                    │
  │  P(<-0.5%)      :  {prbs['below_m05']:>5.1f}%                                    │
  │  Ann. Volatility:  {params['ann_vol']:>5.2f}%                                    │
  │  Adj. Drift Ann :  {params['ann_adjmu']:>+5.2f}%                                   │
  └─────────────────────────────────────────────────────────────────┘
""")

if __name__ == "__main__":
    main()
