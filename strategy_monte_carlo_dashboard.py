"""
strategy_monte_carlo_dashboard.py
──────────────────────────────────
Quantitative Trading Strategy Simulator — Monte Carlo Edition
Run with:  streamlit run strategy_monte_carlo_dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Strategy Monte Carlo Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"
MUTED     = "#8b949e"
TEXT      = "#e6edf3"

st.markdown(f"""
<style>
  /* ── global ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {DARK_BG};
      color: {TEXT};
      font-family: 'Inter', 'Segoe UI', sans-serif;
  }}
  [data-testid="stSidebar"] {{
      background-color: {PANEL_BG};
      border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* ── metric cards ── */
  .metric-card {{
      background: {PANEL_BG};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 18px 22px;
      text-align: center;
  }}
  .metric-label {{
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {MUTED};
      margin-bottom: 6px;
  }}
  .metric-value {{
      font-size: 1.6rem;
      font-weight: 700;
      color: {TEXT};
  }}
  .metric-sub {{
      font-size: 0.78rem;
      color: {MUTED};
      margin-top: 4px;
  }}

  /* ── section headers ── */
  .section-header {{
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: {MUTED};
      border-bottom: 1px solid {BORDER};
      padding-bottom: 6px;
      margin: 28px 0 16px 0;
  }}

  /* ── risk badge ── */
  .risk-box {{
      background: {PANEL_BG};
      border-left: 4px solid {ACCENT};
      border-radius: 6px;
      padding: 16px 20px;
      font-size: 0.9rem;
      line-height: 1.7;
      color: {TEXT};
  }}

  /* ── table ── */
  .stat-table {{ width: 100%; border-collapse: collapse; }}
  .stat-table tr {{ border-bottom: 1px solid {BORDER}; }}
  .stat-table td {{
      padding: 9px 6px;
      font-size: 0.88rem;
      color: {TEXT};
  }}
  .stat-table td:last-child {{ text-align: right; font-weight: 600; }}

  div[data-testid="stHorizontalBlock"] > div {{ gap: 14px; }}
  .stMarkdown p {{ margin: 0; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _plotly_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=14), x=0.02),
        height=height,
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=MUTED, size=11),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=False),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=False),
        margin=dict(l=50, r=30, t=48, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(color=MUTED)),
    )


def metric_card(label, value, sub=""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {sub_html}
    </div>"""


def simulate_equity(n_trades, p, R, initial_capital, risk_pct, rng):
    equity = np.empty(n_trades + 1)
    equity[0] = initial_capital
    outcomes = rng.random(n_trades) < p          # True = win
    for i in range(n_trades):
        risk_amt = equity[i] * risk_pct
        equity[i + 1] = equity[i] + (R * risk_amt if outcomes[i] else -risk_amt)
        equity[i + 1] = max(equity[i + 1], 0.0)  # floor at zero
    return equity


def calc_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return dd


def run_monte_carlo(n_sims, n_trades, p, R, initial_capital, risk_pct, rng):
    finals   = np.empty(n_sims)
    max_dds  = np.empty(n_sims)
    paths    = []
    store_n  = min(n_sims, 100)

    for i in range(n_sims):
        eq = simulate_equity(n_trades, p, R, initial_capital, risk_pct, rng)
        finals[i]  = eq[-1]
        max_dds[i] = calc_drawdown(eq).min()
        if i < store_n:
            paths.append(eq)

    return finals, max_dds, np.array(paths)


# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"<div style='font-size:1.1rem;font-weight:700;color:{TEXT};margin-bottom:4px;'>⚙️ Strategy Parameters</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem;color:{MUTED};margin-bottom:20px;'>Adjust inputs to re-run simulation</div>", unsafe_allow_html=True)

    win_rate   = st.slider("Win Rate (%)", 1, 99, 50, 1)
    rr_ratio   = st.number_input("Risk-Reward Ratio (R)", min_value=0.1, max_value=20.0, value=1.5, step=0.1)
    init_cap   = st.number_input("Initial Capital ($)", min_value=1000, max_value=10_000_000, value=100_000, step=1000)
    risk_pct   = st.slider("Risk per Trade (%)", 0.1, 10.0, 1.0, 0.1) / 100
    n_trades   = st.slider("Number of Trades", 10, 2000, 200, 10)
    n_sims     = st.slider("Monte Carlo Repetitions", 100, 5000, 1000, 100)
    seed_input = st.number_input("Random Seed (0 = random)", min_value=0, max_value=99999, value=42)
    seed       = int(seed_input) if seed_input > 0 else None

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", use_container_width=True)

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────

st.markdown(f"""
<div style='margin-bottom:6px;'>
  <span style='font-size:1.5rem;font-weight:800;color:{TEXT};'>📊 Strategy Monte Carlo Simulator</span>
  <span style='font-size:0.8rem;color:{MUTED};margin-left:14px;'>Quantitative Performance & Risk Analysis</span>
</div>
<div style='font-size:0.78rem;color:{MUTED};margin-bottom:24px;'>
  Win Rate: <b style='color:{TEXT}'>{win_rate}%</b> &nbsp;|&nbsp;
  R:R Ratio: <b style='color:{TEXT}'>{rr_ratio}</b> &nbsp;|&nbsp;
  Trades: <b style='color:{TEXT}'>{n_trades}</b> &nbsp;|&nbsp;
  Capital: <b style='color:{TEXT}'>${init_cap:,}</b> &nbsp;|&nbsp;
  Risk/Trade: <b style='color:{TEXT}'>{risk_pct*100:.1f}%</b>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COMPUTE CORE METRICS
# ─────────────────────────────────────────────

p  = win_rate / 100
q  = 1 - p
EV = (p * rr_ratio) - (q * 1)          # in R units
EV_pct = EV * risk_pct * 100           # in % of equity per trade
breakeven_wr = 1 / (1 + rr_ratio) * 100
profit_factor = (p * rr_ratio) / (q * 1) if q > 0 else float("inf")
expected_profit = EV * risk_pct * init_cap

# ─────────────────────────────────────────────
# SECTION A — STRATEGY SUMMARY
# ─────────────────────────────────────────────

st.markdown(f"<div class='section-header'>A · Strategy Summary</div>", unsafe_allow_html=True)

cols = st.columns(6)
ev_color = GREEN if EV > 0 else RED
ev_str   = f'<span style="color:{ev_color}">{EV:+.4f}R</span>'
pf_str   = f'<span style="color:{GREEN if profit_factor>1 else RED}">{profit_factor:.2f}</span>'

metrics = [
    ("Win Rate",        f"{win_rate}%",                 f"Loss rate: {100-win_rate}%"),
    ("Risk-Reward",     f"{rr_ratio}R",                 f"Win = +{rr_ratio}R / Loss = −1R"),
    ("Expected Value",  f"{EV:+.4f}R",                  f"${expected_profit:,.2f} / trade"),
    ("Break-Even WR",   f"{breakeven_wr:.1f}%",          f"Current edge: {win_rate - breakeven_wr:+.1f}%"),
    ("Risk / Trade",    f"{risk_pct*100:.1f}%",          f"${init_cap*risk_pct:,.0f} at entry"),
    ("Profit Factor",   f"{profit_factor:.2f}x",         "Gross wins / Gross losses"),
]
for col, (lbl, val, sub) in zip(cols, metrics):
    col.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIMULATION (always runs on load / on button)
# ─────────────────────────────────────────────

rng = np.random.default_rng(seed)

# Single equity curve
single_eq = simulate_equity(n_trades, p, rr_ratio, init_cap, risk_pct, rng)
dd_series  = calc_drawdown(single_eq)
max_dd     = dd_series.min()
final_eq   = single_eq[-1]
total_ret  = (final_eq - init_cap) / init_cap * 100

# Monte Carlo
mc_finals, mc_dds, mc_paths = run_monte_carlo(
    n_sims, n_trades, p, rr_ratio, init_cap, risk_pct,
    np.random.default_rng(seed)
)

# MC stats
mean_eq  = mc_finals.mean()
med_eq   = np.median(mc_finals)
p5_eq    = np.percentile(mc_finals, 5)
p95_eq   = np.percentile(mc_finals, 95)
worst_dd = mc_dds.min()
avg_dd   = mc_dds.mean()
prob_prof = (mc_finals > init_cap).mean() * 100

trade_idx = np.arange(n_trades + 1)

# ─────────────────────────────────────────────
# SECTION B — EXPECTED VALUE METRICS
# ─────────────────────────────────────────────

st.markdown(f"<div class='section-header'>B · Expected Value Metrics</div>", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1])
with c1:
    table_rows = [
        ("Expected Value (R)",         f"{EV:+.5f} R"),
        ("Expected Profit / Trade",    f"${expected_profit:,.2f}"),
        ("Break-Even Win Rate",        f"{breakeven_wr:.2f}%"),
        ("Profit Factor",              f"{profit_factor:.3f}"),
        ("Edge over Break-Even",       f"{win_rate - breakeven_wr:+.2f}%"),
        ("Expected Trades to Double",  f"{int(0.693 / (EV * risk_pct)):,}" if EV > 0 else "∞"),
    ]
    rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in table_rows)
    st.markdown(f"<table class='stat-table'>{rows_html}</table>", unsafe_allow_html=True)

with c2:
    # Gauge-style EV visual
    fig_ev = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=EV,
        delta={"reference": 0, "increasing": {"color": GREEN}, "decreasing": {"color": RED}},
        number={"suffix": " R", "font": {"size": 28, "color": TEXT}},
        gauge={
            "axis": {"range": [-1, rr_ratio], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": GREEN if EV > 0 else RED},
            "bgcolor": DARK_BG,
            "bordercolor": BORDER,
            "steps": [
                {"range": [-1, 0], "color": "#2d1a1a"},
                {"range": [0, rr_ratio], "color": "#1a2d1a"},
            ],
            "threshold": {
                "line": {"color": ACCENT, "width": 3},
                "thickness": 0.8,
                "value": 0,
            },
        },
        title={"text": "Expected Value per Trade", "font": {"color": MUTED, "size": 12}},
    ))
    fig_ev.update_layout(
        paper_bgcolor=PANEL_BG,
        font={"color": TEXT},
        height=240,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    st.plotly_chart(fig_ev, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION C — EQUITY CURVE
# ─────────────────────────────────────────────

st.markdown(f"<div class='section-header'>C · Single-Run Equity Curve & Drawdown</div>", unsafe_allow_html=True)

fig_eq = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
)

# Equity
fig_eq.add_trace(go.Scatter(
    x=trade_idx, y=single_eq,
    name="Equity",
    line=dict(color=ACCENT, width=2),
    fill="tozeroy",
    fillcolor=f"rgba(88,166,255,0.07)",
), row=1, col=1)

# Baseline
fig_eq.add_hline(y=init_cap, line_dash="dot", line_color=MUTED, line_width=1, row=1, col=1)

# Drawdown
fig_eq.add_trace(go.Scatter(
    x=trade_idx, y=dd_series * 100,
    name="Drawdown %",
    line=dict(color=RED, width=1.5),
    fill="tozeroy",
    fillcolor=f"rgba(248,81,73,0.12)",
), row=2, col=1)

fig_eq.update_layout(
    **_plotly_layout("", 480),
    showlegend=True,
)
fig_eq.update_yaxes(title_text="Portfolio Value ($)", title_font=dict(color=MUTED), row=1, col=1)
fig_eq.update_yaxes(title_text="Drawdown (%)",        title_font=dict(color=MUTED), row=2, col=1)
fig_eq.update_xaxes(title_text="Trade #",             title_font=dict(color=MUTED), row=2, col=1)

st.plotly_chart(fig_eq, use_container_width=True)

# quick summary line
ret_color = GREEN if total_ret > 0 else RED
st.markdown(f"""
<div style='font-size:0.85rem;color:{MUTED};margin-top:-8px;margin-bottom:6px;'>
  Single-run result — Final equity:
  <b style='color:{TEXT}'>${final_eq:,.2f}</b> &nbsp;|&nbsp;
  Return: <b style='color:{ret_color}'>{total_ret:+.2f}%</b> &nbsp;|&nbsp;
  Max Drawdown: <b style='color:{RED}'>{max_dd*100:.2f}%</b>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION D — MONTE CARLO
# ─────────────────────────────────────────────

st.markdown(f"<div class='section-header'>D · Monte Carlo Analysis  ({n_sims:,} simulations)</div>", unsafe_allow_html=True)

# ── D1: 100 path spaghetti + confidence bands ──
path_eq_T = mc_paths.T           # shape (n_trades+1, ≤100)
pct5  = np.percentile(mc_paths, 5,  axis=0)
pct25 = np.percentile(mc_paths, 25, axis=0)
pct50 = np.percentile(mc_paths, 50, axis=0)
pct75 = np.percentile(mc_paths, 75, axis=0)
pct95 = np.percentile(mc_paths, 95, axis=0)

fig_paths = go.Figure()

# Spaghetti paths (faint)
for i, path in enumerate(mc_paths[:100]):
    fig_paths.add_trace(go.Scatter(
        x=trade_idx, y=path,
        mode="lines",
        line=dict(color=f"rgba(88,166,255,0.07)", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

# Confidence bands
fig_paths.add_trace(go.Scatter(
    x=np.concatenate([trade_idx, trade_idx[::-1]]),
    y=np.concatenate([pct95, pct5[::-1]]),
    fill="toself",
    fillcolor=f"rgba(88,166,255,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="5–95th pct",
))
fig_paths.add_trace(go.Scatter(
    x=np.concatenate([trade_idx, trade_idx[::-1]]),
    y=np.concatenate([pct75, pct25[::-1]]),
    fill="toself",
    fillcolor=f"rgba(88,166,255,0.20)",
    line=dict(color="rgba(0,0,0,0)"),
    name="25–75th pct",
))

# Median
fig_paths.add_trace(go.Scatter(
    x=trade_idx, y=pct50,
    mode="lines",
    name="Median",
    line=dict(color=ACCENT, width=2.5),
))

# Initial capital reference
fig_paths.add_hline(y=init_cap, line_dash="dot", line_color=MUTED, line_width=1)

fig_paths.update_layout(**_plotly_layout("Monte Carlo Equity Paths (100 shown)", 400))
fig_paths.update_xaxes(title_text="Trade #")
fig_paths.update_yaxes(title_text="Portfolio Value ($)")
st.plotly_chart(fig_paths, use_container_width=True)

col_d2, col_d3 = st.columns(2)

# ── D2: Final equity distribution ──
with col_d2:
    profitable = mc_finals[mc_finals > init_cap]
    losing     = mc_finals[mc_finals <= init_cap]

    fig_hist = go.Figure()
    if len(losing) > 0:
        fig_hist.add_trace(go.Histogram(
            x=losing, nbinsx=40,
            marker_color=f"rgba(248,81,73,0.7)",
            name="Below Initial",
        ))
    if len(profitable) > 0:
        fig_hist.add_trace(go.Histogram(
            x=profitable, nbinsx=40,
            marker_color=f"rgba(63,185,80,0.7)",
            name="Profitable",
        ))

    fig_hist.add_vline(x=init_cap,  line_dash="dot", line_color=MUTED,   annotation_text="Initial",  annotation_font_color=MUTED)
    fig_hist.add_vline(x=mean_eq,   line_dash="dash", line_color=YELLOW, annotation_text="Mean",     annotation_font_color=YELLOW)
    fig_hist.add_vline(x=med_eq,    line_dash="dash", line_color=ACCENT,  annotation_text="Median",   annotation_font_color=ACCENT)

    fig_hist.update_layout(**_plotly_layout("Final Equity Distribution", 360), barmode="overlay")
    fig_hist.update_xaxes(title_text="Final Equity ($)")
    fig_hist.update_yaxes(title_text="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

# ── D3: Drawdown distribution ──
with col_d3:
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Histogram(
        x=mc_dds * 100,
        nbinsx=40,
        marker_color=f"rgba(248,81,73,0.65)",
        name="Max Drawdown",
    ))
    fig_dd.add_vline(x=avg_dd * 100, line_dash="dash", line_color=YELLOW,
                     annotation_text=f"Avg {avg_dd*100:.1f}%", annotation_font_color=YELLOW)
    fig_dd.add_vline(x=worst_dd * 100, line_dash="dash", line_color=RED,
                     annotation_text=f"Worst {worst_dd*100:.1f}%", annotation_font_color=RED)

    fig_dd.update_layout(**_plotly_layout("Max Drawdown Distribution", 360))
    fig_dd.update_xaxes(title_text="Max Drawdown (%)")
    fig_dd.update_yaxes(title_text="Count")
    st.plotly_chart(fig_dd, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION E — MC STATISTICS TABLE
# ─────────────────────────────────────────────

st.markdown(f"<div class='section-header'>E · Monte Carlo Statistics</div>", unsafe_allow_html=True)

cols_e = st.columns(6)
mc_metrics = [
    ("Mean Final Equity",   f"${mean_eq:,.0f}",  f"{(mean_eq/init_cap-1)*100:+.1f}% return"),
    ("Median Final Equity", f"${med_eq:,.0f}",   f"{(med_eq/init_cap-1)*100:+.1f}% return"),
    ("5th Pct Equity",      f"${p5_eq:,.0f}",    f"Worst-case zone"),
    ("95th Pct Equity",     f"${p95_eq:,.0f}",   f"Best-case zone"),
    ("Worst Drawdown",      f"{worst_dd*100:.1f}%", f"Avg: {avg_dd*100:.1f}%"),
    ("Prob. of Profit",     f"{prob_prof:.1f}%",  f"Out of {n_sims:,} runs"),
]
for col, (lbl, val, sub) in zip(cols_e, mc_metrics):
    col.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

# Detailed stat table
st.markdown("<br>", unsafe_allow_html=True)
c_left, c_right = st.columns(2)
with c_left:
    rows2 = [
        ("Mean Final Equity",         f"${mean_eq:,.2f}"),
        ("Median Final Equity",       f"${med_eq:,.2f}"),
        ("5th Percentile Equity",     f"${p5_eq:,.2f}"),
        ("95th Percentile Equity",    f"${p95_eq:,.2f}"),
        ("Interquartile Range",       f"${np.percentile(mc_finals,75)-np.percentile(mc_finals,25):,.2f}"),
        ("Std Dev of Final Equity",   f"${mc_finals.std():,.2f}"),
    ]
    rows_html2 = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows2)
    st.markdown(f"<table class='stat-table'>{rows_html2}</table>", unsafe_allow_html=True)

with c_right:
    rows3 = [
        ("Worst Max Drawdown",        f"{worst_dd*100:.2f}%"),
        ("Average Max Drawdown",      f"{avg_dd*100:.2f}%"),
        ("Probability of Profit",     f"{prob_prof:.2f}%"),
        ("Probability of 2× Capital", f"{(mc_finals > 2*init_cap).mean()*100:.2f}%"),
        ("Probability of Ruin (<10%)",f"{(mc_finals < init_cap*0.1).mean()*100:.2f}%"),
        ("Kelly Criterion (full)",    f"{max(EV / rr_ratio, 0)*100:.2f}%"),
    ]
    rows_html3 = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows3)
    st.markdown(f"<table class='stat-table'>{rows_html3}</table>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION F — RISK INTERPRETATION
# ─────────────────────────────────────────────

st.markdown(f"<div class='section-header'>F · Quantitative Risk Interpretation</div>", unsafe_allow_html=True)

# Build interpretation text dynamically
if EV > 0:
    exp_line = (f"✅ <b>Positive expectancy:</b> This strategy has an expected value of "
                f"<b>{EV:+.4f}R</b> per trade (≈ ${expected_profit:,.2f}), meaning it generates "
                f"edge over a large sample. The profit factor of <b>{profit_factor:.2f}x</b> "
                f"confirms gross wins exceed gross losses.")
else:
    exp_line = (f"⛔ <b>Negative expectancy:</b> With an EV of "
                f"<b>{EV:+.4f}R</b>, this strategy loses money on average. "
                f"Consider raising the win rate above <b>{breakeven_wr:.1f}%</b> or improving the "
                f"risk-reward ratio.")

spread_pct = (p95_eq - p5_eq) / init_cap * 100
var_line = (f"📊 <b>Outcome variability:</b> Across {n_sims:,} Monte Carlo paths, the 5th–95th "
            f"percentile range spans <b>${p5_eq:,.0f} – ${p95_eq:,.0f}</b> "
            f"({spread_pct:.0f}% of capital). "
            f"The median outcome is <b>${med_eq:,.0f}</b>. "
            f"Higher risk-per-trade amplifies both gains and losses.")

if worst_dd < -0.30:
    dd_comment = f"⚠️ Severe tail-risk: worst simulated drawdown reached <b>{worst_dd*100:.1f}%</b>."
elif worst_dd < -0.15:
    dd_comment = f"🟡 Moderate drawdown risk: worst case <b>{worst_dd*100:.1f}%</b>, average <b>{avg_dd*100:.1f}%</b>."
else:
    dd_comment = f"🟢 Controlled drawdown risk: worst case <b>{worst_dd*100:.1f}%</b>, well within tolerance."

dd_line = (f"{dd_comment} At <b>{risk_pct*100:.1f}%</b> risk per trade, "
           f"consecutive losing streaks can erode capital significantly. "
           f"Consider reducing position size or adding a stop-loss rule if drawdown exceeds a threshold.")

prob_line = (f"🎯 <b>Probability of profit: {prob_prof:.1f}%</b> of all simulations ended above the initial capital. "
             f"Probability of doubling: <b>{(mc_finals > 2*init_cap).mean()*100:.1f}%</b>. "
             f"Probability of near-ruin (<10% capital remaining): "
             f"<b>{(mc_finals < init_cap*0.1).mean()*100:.1f}%</b>.")

kelly = max(EV / rr_ratio, 0) * 100
kelly_line = (f"💡 <b>Kelly Criterion:</b> Full Kelly suggests risking <b>{kelly:.1f}%</b> per trade. "
              f"Your current risk of <b>{risk_pct*100:.1f}%</b> is "
              f"{'below' if risk_pct*100 < kelly else 'above'} full Kelly — "
              f"{'conservative and robust' if risk_pct*100 < kelly else 'aggressive; consider halving for capital preservation'}.")

st.markdown(f"""
<div class="risk-box">
  {exp_line}<br><br>
  {var_line}<br><br>
  {dd_line}<br><br>
  {prob_line}<br><br>
  {kelly_line}
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='margin-top:28px;font-size:0.72rem;color:{MUTED};text-align:center;'>
  Built with Streamlit · Plotly · NumPy &nbsp;|&nbsp;
  Monte Carlo results are probabilistic — past simulation is not a guarantee of future results.
  Seed: {seed if seed else 'random'}
</div>
""", unsafe_allow_html=True)
