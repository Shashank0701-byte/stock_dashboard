import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

ticker_symbol = st.text_input("Enter Stock Ticker", "AAPL")

try:
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(period="1d", start="2015-01-01", end=date.today())
    
    if df.empty:
        st.error("No data found for the given ticker symbol. Please check the ticker.")
    else:
        st.subheader(f"Price Chart for: {ticker_symbol}")
        
        # Create the candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        
        fig.update_layout(xaxis_rangeslider_visible=False, 
                          title=f"{ticker_symbol} Candlestick Chart",
                          yaxis_title="Price (USD)")
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")