import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

ticker_symbol = st.text_input("Enter Stock Ticker", "AAPL")

today = date.today()
start_date = st.date_input('Start date', today - timedelta(days=365*5))
end_date = st.date_input('End date', today)

try:
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(start=start_date, end=end_date)
    
    if df.empty:
        st.error("No data found for the given ticker symbol and date range.")
    else:
        # Calculate Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        st.subheader(f"Price Chart for: {ticker_symbol}")
        
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'],
                                             name='Candlestick')])
        
        # Add SMA traces to the figure
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='purple')))
        
        fig.update_layout(xaxis_rangeslider_visible=False, 
                          title=f"{ticker_symbol} Candlestick Chart with Moving Averages",
                          yaxis_title="Price (USD)")
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")