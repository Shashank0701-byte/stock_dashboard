import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
import requests
import xgboost as xgb
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_autorefresh import st_autorefresh
import sqlite3

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# --- USER AUTHENTICATION ---
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Authentication configuration file ('config.yaml') not found.")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

st.title("Stock Analysis Dashboard")
authenticator.login()

if st.session_state["authentication_status"]:
    # --- AUTO-REFRESH ---
    st_autorefresh(interval=60 * 1000, key="datarefresh")

    # --- LOGOUT BUTTON IN SIDEBAR ---
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')
    authenticator.logout()

    # --- DATABASE HELPER FUNCTIONS ---
    def load_watchlist(username):
        conn = sqlite3.connect('watchlist.db')
        cursor = conn.cursor()
        cursor.execute("SELECT tickers FROM watchlists WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "AAPL,GOOG,MSFT"

    def save_watchlist(username, tickers):
        conn = sqlite3.connect('watchlist.db')
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO watchlists (username, tickers) VALUES (?, ?)", (username, tickers))
        conn.commit()
        conn.close()

    # --- START OF THE MAIN APPLICATION LOGIC ---
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        st.info("Downloading VADER lexicon for sentiment analysis...")
        nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # --- Helper Functions (Cached for performance) ---
    @st.cache_data
    def get_stock_data(ticker, start_date, end_date):
        return yf.Ticker(ticker).history(start=start_date, end=end_date)

    @st.cache_data
    def get_ticker_info(ticker_symbol):
        return yf.Ticker(ticker_symbol).info

    @st.cache_data
    def get_financials(ticker_symbol):
        return yf.Ticker(ticker_symbol).financials

    @st.cache_data
    def get_balance_sheet(ticker_symbol):
        return yf.Ticker(ticker_symbol).balance_sheet

    @st.cache_data
    def get_news(ticker_symbol):
        try:
            API_KEY = st.secrets["NEWS_API_KEY"]
            url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&apiKey={API_KEY}&sortBy=publishedAt&language=en"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            formatted_news = []
            for article in articles:
                formatted_news.append({'title': article.get('title'), 'publisher': article.get('source', {}).get('name'), 'link': article.get('url')})
            return formatted_news
        except Exception as e:
            st.error(f"Could not fetch news. Error: {e}. Please check your API key.")
            return []

    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # --- SIMPLIFIED SIDEBAR ---
    st.sidebar.header("Search & Save")
    username = st.session_state["username"]
    
    saved_watchlist = load_watchlist(username)
    ticker_list_str = st.sidebar.text_input("Enter Stock Tickers", value=saved_watchlist)
    
    if st.sidebar.button("Save to Watchlist"):
        save_watchlist(username, ticker_list_str)
        st.toast("Watchlist updated!", icon="💾")

    # The sidebar expander to VIEW the watchlist
    with st.sidebar.expander("View My Watchlist"):
        st.write(saved_watchlist.replace(",", ", "))

    ticker_list = [ticker.strip().upper() for ticker in ticker_list_str.split(',') if ticker.strip()]
    primary_ticker = ticker_list[0] if ticker_list else None

    today = date.today()
    start_date = st.sidebar.date_input('Start date', today - timedelta(days=365*5))
    end_date = st.sidebar.date_input('End date', today)

    st.sidebar.subheader("Display Options")
    show_sma = st.sidebar.checkbox("Show Moving Averages", value=True)
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    show_comparison = st.sidebar.checkbox("Show Comparative Analysis", value=True)
    show_news = st.sidebar.checkbox("Show Recent News", value=True)

    st.sidebar.subheader("Technical Indicators")
    show_rsi = st.sidebar.checkbox("Show RSI", value=True)
    show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    show_macd = st.sidebar.checkbox("Show MACD", value=True)

    # --- MAIN DASHBOARD LOGIC ---
    if primary_ticker:
        df = get_stock_data(primary_ticker, start_date, end_date)
        info = get_ticker_info(primary_ticker)
        
        if df.empty:
            st.error(f"No data found for {primary_ticker} in the selected date range.")
        else:
            st.header(f"Analysis for {info.get('longName', primary_ticker)}")
            st.markdown(f"**Sector**: {info.get('sector', 'N/A')} | **Industry**: {info.get('industry', 'N/A')}")
            with st.expander("Business Summary"):
                st.write(info.get('longBusinessSummary', 'No summary available.'))
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
            with c2:
                st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
            with c3:
                st.metric("Dividend Yield", f"${info.get('dividendYield', 0)*100:.2f}%")
            
            tab_chart, tab_financials, tab_news, tab_forecast = st.tabs(["📈 Chart Analysis", "💰 Financials", "📰 News", "🔮 Forecast"])

            with tab_chart:
                st.subheader(f"Price Chart for: {primary_ticker}")
                fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig_price.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'), row=1, col=1)
                if show_sma or show_bb:
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                if show_sma:
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA', line=dict(color='orange')), row=1, col=1)
                    fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='purple')), row=1, col=1)
                if show_bb:
                    df['BB_20_std'] = df['Close'].rolling(window=20).std()
                    df['BB_Upper'] = df['SMA_20'] + (df['BB_20_std'] * 2)
                    df['BB_Lower'] = df['SMA_20'] - (df['BB_20_std'] * 2)
                    fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='Upper Band', line=dict(color='gray', width=1)))
                    fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='Lower Band', line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
                if show_volume:
                    fig_price.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
                fig_price.update_layout(xaxis_rangeslider_visible=False, title=f"{primary_ticker} Price and Volume", yaxis_title="Price (USD)")
                fig_price.update_yaxes(title_text="Volume", row=2, col=1)
                st.plotly_chart(fig_price, use_container_width=True)
                if show_macd:
                    st.subheader("MACD (Moving Average Convergence Divergence)")
                    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
                    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = ema_12 - ema_26
                    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD Line', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line', line=dict(color='orange')))
                    fig_macd.add_trace(go.Bar(x=df.index, y=(df['MACD'] - df['Signal_Line']), name='Histogram', marker_color='gray'))
                    fig_macd.update_layout(title="MACD Chart", yaxis_title="Value")
                    st.plotly_chart(fig_macd, use_container_width=True)
                if show_rsi:
                    df['RSI'] = calculate_rsi(df)
                    st.subheader("Relative Strength Index (RSI)")
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                    fig_rsi.update_layout(title="RSI Chart", yaxis_title="RSI")
                    st.plotly_chart(fig_rsi, use_container_width=True)
                if len(ticker_list) > 1 and show_comparison:
                    st.subheader("Comparative Performance")
                    norm_df = pd.DataFrame()
                    for ticker_col in ticker_list:
                        data = get_stock_data(ticker_col, start_date, end_date)
                        if not data.empty:
                            norm_df[ticker_col] = (data['Close'] / data['Close'].iloc[0]) - 1
                    if not norm_df.empty:
                        fig_comp = go.Figure()
                        for ticker_col in norm_df.columns:
                            fig_comp.add_trace(go.Scatter(x=norm_df.index, y=norm_df[ticker_col], mode='lines', name=ticker_col))
                        fig_comp.update_layout(title="Normalized Stock Performance", yaxis_title="Percentage Change", yaxis_tickformat=".2%")
                        st.plotly_chart(fig_comp, use_container_width=True)

            with tab_financials:
                st.subheader(f"Financials for {primary_ticker}")
                financials = get_financials(primary_ticker)
                balance_sheet = get_balance_sheet(primary_ticker)
                if financials.empty or balance_sheet.empty:
                    st.warning("Could not retrieve financial data for this stock.")
                else:
                    st.write("**Income Statement Highlights (Annual)**")
                    try:
                        income_metrics = {"Total Revenue": financials.loc['Total Revenue'].iloc[0], "Gross Profit": financials.loc['Gross Profit'].iloc[0], "Net Income": financials.loc['Net Income'].iloc[0]}
                        c1f, c2f, c3f = st.columns(3)
                        c1f.metric("Total Revenue", f"${income_metrics['Total Revenue']/1e6:,.0f}M")
                        c2f.metric("Gross Profit", f"${income_metrics['Gross Profit']/1e6:,.0f}M")
                        c3f.metric("Net Income", f"${income_metrics['Net Income']/1e6:,.0f}M")
                    except KeyError:
                        st.warning("Could not display some income statement metrics.")
                    st.write("**Balance Sheet Highlights (Annual)**")
                    try:
                        balance_metrics = {"Total Assets": balance_sheet.loc['Total Assets'].iloc[0], "Total Liabilities": balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0], "Total Equity": balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0]}
                        c1b, c2b, c3b = st.columns(3)
                        c1b.metric("Total Assets", f"${balance_metrics['Total Assets']/1e6:,.0f}M")
                        c2b.metric("Total Liabilities", f"${balance_metrics['Total Liabilities']/1e6:,.0f}M")
                        c3b.metric("Total Equity", f"${balance_metrics['Total Equity']/1e6:,.0f}M")
                    except KeyError:
                        st.warning("Could not display some balance sheet metrics.")
                    with st.expander("View Full Annual Income Statement"):
                        st.dataframe(financials)
                    with st.expander("View Full Annual Balance Sheet"):
                        st.dataframe(balance_sheet)

            with tab_news:
                if show_news:
                    st.subheader(f"Recent News for {primary_ticker}")
                    news = get_news(primary_ticker)
                    if not news:
                        st.write("No recent news found.")
                    else:
                        sia = SentimentIntensityAnalyzer()
                        for article in news[:15]:
                            article_title = article.get('title', 'No Title Available')
                            article_link = article.get('link', '#')
                            article_publisher = article.get('publisher', 'N/A')
                            sentiment = sia.polarity_scores(article_title)
                            color = "gray" 
                            if sentiment['compound'] >= 0.05: color = "green"
                            elif sentiment['compound'] <= -0.05: color = "red"
                            st.markdown(f"""<div style="border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                                <p style="margin: 0;"><strong><a href="{article_link}" target="_blank" style="text-decoration: none; color: inherit;">{article_title}</a></strong></p>
                                <small>Publisher: {article_publisher} | Compound Score: {sentiment['compound']:.2f}</small>
                                </div>""", unsafe_allow_html=True)
            
            with tab_forecast:
                st.subheader(f"Custom Price Forecast for {primary_ticker}")
                try:
                    st.write("Step 1: Creating features from the data...")
                    df_features = df.copy()
                    df_features['lag_1'] = df_features['Close'].shift(1)
                    df_features['lag_7'] = df_features['Close'].shift(7)
                    df_features['rolling_mean_7'] = df_features['Close'].rolling(window=7).mean()
                    df_features['rolling_std_7'] = df_features['Close'].rolling(window=7).std()
                    df_features['day_of_week'] = df_features.index.dayofweek
                    df_features['month'] = df_features.index.month
                    features = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'day_of_week', 'month']
                    target = 'Close'
                    df_features.dropna(inplace=True)
                    X = df_features[features]
                    y = df_features[target]

                    st.write("Step 2: Splitting data into training and testing sets...")
                    split_point = int(len(X) * 0.8)
                    X_train, X_test = X[:split_point], X[split_point:]
                    y_train, y_test = y[:split_point], y[split_point:]
                    st.write(f"Training on {len(X_train)} data points, testing on {len(X_test)} data points.")

                    st.write("Step 3: Training the XGBoost model... (This may take a moment)")
                    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, objective='reg:squarederror')
                    model.fit(X_train, y_train)

                    st.write("Step 4: Making predictions and visualizing the results...")
                    predictions = model.predict(X_test)
                    results_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': predictions})
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Historical Training Data'))
                    fig_forecast.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual Price'], mode='lines', name='Actual Price (Test)'))
                    fig_forecast.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted Price'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
                    fig_forecast.update_layout(title=f"XGBoost Forecast vs. Actual Price for {primary_ticker}",
                                               xaxis_title="Date", yaxis_title="Price (USD)")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    with st.expander("View Model Feature Importance"):
                        feature_importance = model.feature_importances_
                        importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance}).sort_values('importance', ascending=False)
                        st.bar_chart(importance_df.set_index('feature'))
                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")
    else:
        st.info("Enter a stock ticker in the sidebar to begin analysis.")

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')