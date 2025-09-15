import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date

# Set the page title and a header
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

# User input for the stock ticker
ticker_symbol = st.text_input("Enter Stock Ticker", "AAPL")

# Fetch data
try:
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(period="1d", start="2015-01-01", end=date.today())
    
    if df.empty:
        st.error("No data found for the given ticker symbol. Please check the ticker.")
    else:
        st.subheader(f"Displaying data for: {ticker_symbol}")
        st.dataframe(df)

except Exception as e:
    st.error(f"An error occurred: {e}")