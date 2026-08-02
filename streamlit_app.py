"""Single-file deployment build for Streamlit Community Cloud."""
from __future__ import annotations

import math
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="MarketLens India", page_icon=":material/insights:", layout="wide")
st.session_state.setdefault("watchlist", pd.DataFrame(columns=["Ticker", "Target", "Condition"]))
st.session_state.setdefault("portfolio", pd.DataFrame(columns=["Ticker", "Quantity", "Average price"]))


def yahoo(ticker: str, exchange: str) -> str:
    return f"{ticker.upper().strip()}{'.NS' if exchange == 'NSE' else '.BO'}"


@st.cache_data(ttl=600, show_spinner=False)
def load_stock(ticker: str, exchange: str, period: str):
    stock = yf.Ticker(yahoo(ticker, exchange))
    data = stock.history(period=period, auto_adjust=False).drop(columns=["Dividends", "Stock Splits"], errors="ignore")
    if data.empty: raise ValueError("No price data found. Check the ticker and exchange.")
    data.index = pd.to_datetime(data.index).tz_localize(None)
    close = data.Close
    data["EMA 20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA 50"] = close.ewm(span=50, adjust=False).mean()
    data["SMA 200"] = close.rolling(200).mean()
    delta = close.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    data["RSI"] = 100 - 100 / (1 + rs)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    data["MACD"], data["Signal"] = macd, macd.ewm(span=9, adjust=False).mean()
    info = stock.info or {}
    financials = stock.financials if stock.financials is not None else pd.DataFrame()
    news = stock.news or []
    return data, info, financials, news


def signal(data: pd.DataFrame, info: dict) -> tuple[int, str, list[str]]:
    row = data.iloc[-1]; score = 50; why = []
    rsi = row.RSI
    if pd.notna(rsi) and 45 <= rsi <= 65: score += 8; why.append("RSI is in a constructive 45–65 range (+8).")
    elif pd.notna(rsi) and rsi > 75: score -= 10; why.append("RSI indicates stretched momentum (−10).")
    if row.MACD > row.Signal: score += 8; why.append("MACD is above its signal line (+8).")
    else: score -= 8; why.append("MACD is below its signal line (−8).")
    if row["EMA 20"] > row["EMA 50"]: score += 8; why.append("20-day EMA is above 50-day EMA (+8).")
    else: score -= 8; why.append("20-day EMA is below 50-day EMA (−8).")
    if pd.notna(row["SMA 200"]):
        score += 6 if row.Close > row["SMA 200"] else -6
        why.append("Price is " + ("above" if row.Close > row["SMA 200"] else "below") + " the 200-day average.")
    if (info.get("returnOnEquity") or 0) > .15: score += 5; why.append("Reported ROE exceeds 15% (+5).")
    score = max(0, min(100, score)); label = "Strong Buy" if score >= 80 else "Buy" if score >= 65 else "Hold" if score >= 45 else "Sell" if score >= 30 else "Strong Sell"
    return score, label, why


st.title("MarketLens India")
st.caption("Transparent research for NSE/BSE equities · public data can be delayed or incomplete · not investment advice")
with st.sidebar:
    ticker = st.text_input("Ticker", "RELIANCE").upper().strip()
    exchange = st.segmented_control("Exchange", ["NSE", "BSE"], default="NSE")
    period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=1)
    run = st.button("Analyse company", type="primary", icon=":material/search:")

if run or "stock" not in st.session_state:
    try:
        with st.spinner("Loading market data…"):
            st.session_state.stock = (ticker, exchange, *load_stock(ticker, exchange, period))
    except Exception as exc: st.error(str(exc)); st.stop()

ticker, exchange, data, info, financials, news = st.session_state.stock
score, label, reasons = signal(data, info)
research, compare, screener, plan = st.tabs(["Research", "Compare", "Screener", "Portfolio & SIP"])

with research:
    with st.container(horizontal=True):
        st.metric("Last close", f"₹{data.Close.iloc[-1]:,.2f}", border=True)
        st.metric("Research label", label, f"{score}/100", border=True)
        st.metric("RSI (14)", f"{data.RSI.iloc[-1]:.1f}", border=True)
    figure = go.Figure(go.Candlestick(x=data.index, open=data.Open, high=data.High, low=data.Low, close=data.Close, name="Price"))
    for column in ["EMA 20", "EMA 50", "SMA 200"]: figure.add_trace(go.Scatter(x=data.index, y=data[column], name=column))
    figure.update_layout(height=550, xaxis_rangeslider_visible=False, title=f"{ticker}.{exchange} price history")
    st.plotly_chart(figure, width="stretch")
    st.subheader("Why this label")
    for reason in reasons: st.write("• " + reason)
    st.subheader("Available fundamentals")
    fundamentals = {"Market cap": info.get("marketCap"), "P/E": info.get("trailingPE"), "P/B": info.get("priceToBook"), "EPS": info.get("trailingEps"), "ROE": info.get("returnOnEquity"), "Debt/equity": info.get("debtToEquity"), "Dividend yield": info.get("dividendYield")}
    st.dataframe(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]), hide_index=True)
    with st.expander("Annual financial statements"): st.dataframe(financials)
    st.subheader("Recent news sentiment")
    analyser = SentimentIntensityAnalyzer()
    for item in news[:8]:
        content = item.get("content", item); title = content.get("title") or item.get("title")
        if title:
            value = analyser.polarity_scores(title)["compound"]; mood = "Positive" if value >= .2 else "Negative" if value <= -.2 else "Neutral"
            st.write(f"**{mood}** ({value:+.2f}) — {title}")
    st.download_button("Download price CSV", data.to_csv().encode(), f"{ticker}_{exchange}_prices.csv", "text/csv")

with compare:
    peers = st.multiselect("Compare with", [], accept_new_options=True, placeholder="Type tickers, for example TCS")
    if peers:
        series = {ticker: data.Close / data.Close.iloc[0] * 100}
        for peer in peers[:4]:
            try:
                peer_data, _, _, _ = load_stock(peer, exchange, period); series[peer.upper()] = peer_data.Close / peer_data.Close.iloc[0] * 100
            except Exception as exc: st.warning(f"Could not load {peer}: {exc}")
        st.line_chart(pd.DataFrame(series))
        st.caption("Relative performance is indexed to 100 at the start of the period.")
    else: st.info("Add one or more tickers to compare price performance.")

with screener:
    st.caption("On-demand scan of tickers you select. This is not a complete exchange-wide stock screener.")
    candidates = st.multiselect("Tickers to scan", ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT", "TATAMOTORS", "SUNPHARMA"], default=["RELIANCE", "TCS", "INFY"])
    min_rsi, max_rsi = st.slider("RSI range", 0, 100, (35, 70))
    if st.button("Run screener", type="primary"):
        rows=[]
        with st.spinner("Scanning…"):
            for candidate in candidates:
                try:
                    candidate_data, _, _, _ = load_stock(candidate, exchange, "6mo"); row=candidate_data.iloc[-1]
                    rows.append({"Ticker":candidate, "Last close":row.Close, "1-day %":(row.Close/candidate_data.Close.iloc[-2]-1)*100, "RSI":row.RSI, "Above EMA 20":row.Close>row["EMA 20"]})
                except Exception: pass
        scan = pd.DataFrame(rows)
        if not scan.empty:
            st.dataframe(scan[scan.RSI.between(min_rsi,max_rsi)].sort_values("1-day %", ascending=False), hide_index=True)
            st.plotly_chart(px.bar(scan, x="Ticker", y="1-day %", color="1-day %", color_continuous_scale="RdYlGn"), width="stretch")

with plan:
    left, right = st.columns(2)
    with left:
        st.subheader("Local watchlist")
        st.session_state.watchlist = st.data_editor(st.session_state.watchlist, num_rows="dynamic", hide_index=True, key="watchlist_editor")
        st.caption("Temporary for this browser session on free hosting.")
    with right:
        st.subheader("SIP calculator")
        monthly = st.number_input("Monthly investment (₹)", min_value=100.0, value=5000.0, step=500.0)
        annual = st.number_input("Illustrative return (%)", min_value=0.0, max_value=40.0, value=12.0, step=.5)
        years = st.slider("Years", 1, 40, 10)
        months = years * 12; rate = annual / 1200
        value = monthly * months if rate == 0 else monthly * (((1+rate)**months-1)/rate)*(1+rate)
        st.metric("Total invested", f"₹{monthly*months:,.0f}")
        st.metric("Illustrative value", f"₹{value:,.0f}")
        st.caption("Hypothetical constant-return calculation; actual returns can be lower or negative.")

st.caption("Yahoo Finance is a convenient public-data source, not an official exchange feed. Verify figures with exchange filings before investing.")
