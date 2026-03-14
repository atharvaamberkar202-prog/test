import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
import requests

from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK=True
except:
    VADER_OK=False


st.set_page_config(layout="wide", page_title="NIFTY Quant Dashboard")

# -----------------------------
# SETTINGS
# -----------------------------

LOOKBACK_YEARS = st.sidebar.slider("Lookback Years",3,10,5)
SIMS = st.sidebar.selectbox("Monte Carlo Paths",[1000,5000,10000,20000],index=2)

NEWS_API = st.sidebar.text_input("NewsAPI key (optional)",type="password")

# -----------------------------
# DATA
# -----------------------------

@st.cache_data(ttl=3600)
def load_data():

    end = datetime.date.today()
    start = end - datetime.timedelta(days=365*LOOKBACK_YEARS)

    tickers = {
        "NIFTY":"^NSEI",
        "SP500":"^GSPC",
        "NASDAQ":"^NDX",
        "DOW":"^DJI",
        "FTSE":"^FTSE",
        "DAX":"^GDAXI",
        "NIKKEI":"^N225",
        "HANGSENG":"^HSI"
    }

    data={}

    for k,t in tickers.items():

        df=yf.download(t,start=start,end=end,progress=False)

        data[k]=df["Close"]

    return data

data=load_data()

nifty=data["NIFTY"]

# -----------------------------
# RETURNS
# -----------------------------

returns={}

for k,v in data.items():

    returns[k]=np.log(v/v.shift(1)).dropna()

nifty_ret=returns["NIFTY"]

# -----------------------------
# GARCH VOLATILITY FORECAST
# -----------------------------

am=arch_model(nifty_ret*100,vol='Garch',p=1,q=1,dist='t')

res=am.fit(disp="off")

forecast=res.forecast(horizon=1)

var_next=forecast.variance.values[-1,0]

sigma=np.sqrt(var_next)/100

# -----------------------------
# GLOBAL PCA FACTOR
# -----------------------------

global_df=pd.DataFrame({
k:v for k,v in returns.items() if k!="NIFTY"
}).dropna()

scaler=StandardScaler()

X=scaler.fit_transform(global_df)

pca=PCA(n_components=2)

factors=pca.fit_transform(X)

latest=global_df.iloc[-1].values.reshape(1,-1)

latest_std=scaler.transform(latest)

latest_factors=pca.transform(latest_std)

global_signal=latest_factors[0,0]*0.7 + latest_factors[0,1]*0.3

gf=np.tanh(global_signal)

# -----------------------------
# SENTIMENT
# -----------------------------

def get_sentiment():

    headlines=[]

    if NEWS_API:

        try:

            url=f"https://newsapi.org/v2/everything?q=stock market india nifty&language=en&pageSize=20&apiKey={NEWS_API}"

            r=requests.get(url)

            js=r.json()

            headlines=[x["title"] for x in js["articles"] if x.get("title")]

        except:
            pass

    if not headlines:

        headlines=[
        "Global markets rise after strong economic data",
        "Investors cautious ahead of central bank meeting",
        "Technology stocks rally across global markets",
        "Oil prices fall on demand concerns",
        "Strong earnings lift investor sentiment"
        ]

    scores=[]

    if VADER_OK:

        analyzer=SentimentIntensityAnalyzer()

        scores=[analyzer.polarity_scores(h)["compound"] for h in headlines]

    else:

        scores=[0]*len(headlines)

    sentiment=np.mean(scores)

    return sentiment,headlines

ns,headlines=get_sentiment()

# -----------------------------
# DRIFT MODEL
# -----------------------------

mu=nifty_ret.mean()*0.3

sent_alpha=0.01*np.tanh(ns*2)

global_alpha=0.02*gf

adj_mu=mu + (global_alpha + sent_alpha)/252

# -----------------------------
# MONTE CARLO
# -----------------------------

S0=float(nifty.iloc[-1])

rng=np.random.default_rng(42)

df_t=5

Z=rng.standard_t(df_t,SIMS)

Z=Z/np.sqrt(df_t/(df_t-2))

dt=1/252

S_next=S0*np.exp((adj_mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

# -----------------------------
# STATS
# -----------------------------

pcts={
"P5":np.percentile(S_next,5),
"P25":np.percentile(S_next,25),
"Median":np.median(S_next),
"P75":np.percentile(S_next,75),
"P95":np.percentile(S_next,95)
}

prob_up=(S_next>S0).mean()*100

# -----------------------------
# DASHBOARD
# -----------------------------

st.title("NIFTY Quant Prediction Dashboard")

col1,col2,col3,col4=st.columns(4)

col1.metric("NIFTY",f"{S0:,.2f}")
col2.metric("Forecast Volatility",f"{sigma*np.sqrt(252)*100:.2f}%")
col3.metric("Global Factor",f"{gf:.3f}")
col4.metric("Sentiment",f"{ns:.3f}")

st.metric("Probability Up Tomorrow",f"{prob_up:.2f}%")

# -----------------------------
# HISTOGRAM
# -----------------------------

fig=go.Figure()

fig.add_trace(go.Histogram(x=S_next,nbinsx=80))

fig.add_vline(x=S0,line_color="white")

fig.add_vline(x=pcts["Median"],line_color="green")

fig.update_layout(title="Next Day NIFTY Distribution")

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# CONFIDENCE TABLE
# -----------------------------

table=pd.DataFrame({
"Level":["P5","P25","Median","P75","P95"],
"Price":[pcts["P5"],pcts["P25"],pcts["Median"],pcts["P75"],pcts["P95"]]
})

st.dataframe(table,use_container_width=True)

# -----------------------------
# NEWS
# -----------------------------

st.subheader("Latest Headlines")

for h in headlines[:8]:

    st.write("•",h)
