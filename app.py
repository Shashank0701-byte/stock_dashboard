import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
import requests # New import

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

# Download the VADER lexicon for sentiment analysis on first run
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    st.info("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Helper Functions (Cached for performance) ---
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Fetches historical stock data."""
    return yf.Ticker(ticker).history(start=start_date, end=end_date)

@st.cache_data
def get_ticker_info(ticker_symbol):
    """Fetches company information."""
    return yf.Ticker(ticker_symbol).info

# --- START OF REWRITTEN NEWS FUNCTION ---
@st.cache_data
def get_news(ticker_symbol):
    """Fetches recent news from NewsAPI.org."""
    try:
        API_KEY = st.secrets["NEWS_API_KEY"]
        url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&apiKey={API_KEY}&sortBy=publishedAt&language=en"
        
        response = requests.get(url)
        response.raise_for_status() # Raises an exception for bad status codes
        
        articles = response.json().get("articles", [])
        # Format the data to match what our app expects
        formatted_news = []
        for article in articles:
            formatted_news.append({
                'title': article.get('title'),
                'publisher': article.get('source', {}).get('name'),
                'link': article.get('url')
            })
        return formatted_news
    except Exception as e:
        st.error(f"Could not fetch news. Error: {e}. Please check your API key in secrets.toml.")
        return []
# --- END OF REWRITTEN NEWS FUNCTION ---

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Sidebar for User Inputs ---
st.sidebar.header("User Input")
ticker_list_str = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL, GOOG, MSFT")
ticker_list = [ticker.strip().upper() for ticker in ticker_list_str.split(',')]
primary_ticker = ticker_list[0] if ticker_list else None

today = date.today()
start_date = st.sidebar.date_input('Start date', today - timedelta(days=365*5))
end_date = st.sidebar.date_input('End date', today)

# Feature Toggles
st.sidebar.subheader("Display Options")
show_sma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_comparison = st.sidebar.checkbox("Show Comparative Analysis", value=True)
show_news = st.sidebar.checkbox("Show Recent News", value=True)

# --- Main App Logic ---
if primary_ticker:
    try:
        df = get_stock_data(primary_ticker, start_date, end_date)
        info = get_ticker_info(primary_ticker)
        
        if df.empty:
            st.error("No data found for the given ticker symbol and date range.")
        else:
            # Display Company Profile and Metrics
            st.subheader(f"Company Profile: {info.get('longName', primary_ticker)}")
            st.markdown(f"**Sector**: {info.get('sector', 'N/A')} | **Industry**: {info.get('industry', 'N/A')}")
            with st.expander("Business Summary"):
                st.write(info.get('longBusinessSummary', 'No summary available.'))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
            with col2:
                st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
            with col3:
                st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
            
            # Price and Volume Chart
            st.subheader(f"Price Chart for: {primary_ticker}")
            fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig_price.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'), row=1, col=1)
            
            if show_sma:
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA', line=dict(color='orange')), row=1, col=1)
                fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='purple')), row=1, col=1)

            if show_volume:
                fig_price.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
            
            fig_price.update_layout(xaxis_rangeslider_visible=False, title=f"{primary_ticker} Price and Volume", yaxis_title="Price (USD)")
            fig_price.update_yaxes(title_text="Volume", row=2, col=1)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # RSI Chart
            if show_rsi:
                df['RSI'] = calculate_rsi(df)
                st.subheader("Relative Strength Index (RSI)")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(title="RSI Chart", yaxis_title="RSI")
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Comparative Analysis
            if len(ticker_list) > 1 and show_comparison:
                st.subheader("Comparative Performance")
                norm_df = pd.DataFrame()
                for ticker in ticker_list:
                    data = get_stock_data(ticker, start_date, end_date)
                    if not data.empty:
                        norm_df[ticker] = (data['Close'] / data['Close'].iloc[0]) - 1
                
                if not norm_df.empty:
                    fig_comp = go.Figure()
                    for ticker in norm_df.columns:
                        fig_comp.add_trace(go.Scatter(x=norm_df.index, y=norm_df[ticker], mode='lines', name=ticker))
                    
                    fig_comp.update_layout(title="Normalized Stock Performance",
                                           yaxis_title="Percentage Change",
                                           yaxis_tickformat=".2%")
                    st.plotly_chart(fig_comp, use_container_width=True)

            # News Sentiment Analysis
            if show_news:
                st.subheader(f"Recent News for {primary_ticker}")
                news = get_news(primary_ticker)
                if not news:
                    st.write("No recent news found.")
                else:
                    sia = SentimentIntensityAnalyzer()
                    for article in news[:10]: 
                        article_title = article.get('title', 'No Title Available')
                        article_link = article.get('link', '#')
                        article_publisher = article.get('publisher', 'N/A')
                        
                        sentiment = sia.polarity_scores(article_title)
                        color = "gray" 
                        if sentiment['compound'] >= 0.05:
                            color = "green"
                        elif sentiment['compound'] <= -0.05:
                            color = "red"
                        
                        st.markdown(f"""
                        <div style="border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                            <p style="margin: 0;"><strong><a href="{article_link}" target="_blank" style="text-decoration: none; color: inherit;">{article_title}</a></strong></p>
                            <small>Publisher: {article_publisher} | Compound Score: {sentiment['compound']:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter at least one stock ticker in the sidebar to begin.")