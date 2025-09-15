import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Sidebar for User Inputs ---
st.sidebar.header("User Input")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
today = date.today()
start_date = st.sidebar.date_input('Start date', today - timedelta(days=365*5))
end_date = st.sidebar.date_input('End date', today)

# Indicator toggles
show_sma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)

# --- Main App Logic ---
try:
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(start=start_date, end=end_date)
    
    if df.empty:
        st.error("No data found for the given ticker symbol and date range.")
    else:
        # --- Price Chart ---
        st.subheader(f"Price Chart for: {ticker_symbol}")
        fig_price = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')])
        
        if show_sma:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA', line=dict(color='orange')))
            fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='purple')))
        
        fig_price.update_layout(xaxis_rangeslider_visible=False, title=f"{ticker_symbol} Candlestick Chart", yaxis_title="Price (USD)")
        st.plotly_chart(fig_price, use_container_width=True)
        
        # --- RSI Chart ---
        if show_rsi:
            df['RSI'] = calculate_rsi(df)
            st.subheader("Relative Strength Index (RSI)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(title="RSI Chart", yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")