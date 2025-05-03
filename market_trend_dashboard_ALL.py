import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Function to get list of tickers (Here, a sample list; Replace or extend as needed)
@st.cache_data(ttl=24*3600)
def get_stock_tickers():
    # Sample ticker list from NSE and NASDAQ for demo; Expand or load from file or API for real app
    return [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'
    ]

st.title("📈 Market Trend Predictor (Live Market Demo)")

# Load ticker list
tickers = get_stock_tickers()

# Sidebar for stock selection - dropdown now
ticker = st.sidebar.selectbox("Select Stock Ticker", options=tickers, index=tickers.index('RELIANCE.NS') if 'RELIANCE.NS' in tickers else 0)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Fetch stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Check if data is empty
if data.empty:
    st.error("No data found. Please check the ticker symbol or try another.")
    st.stop()

# Display the structure of the DataFrame for debugging
st.write("Data fetched from Yahoo Finance:")
st.write(data)

# Check if 'Close' column exists
if 'Close' not in data.columns:
    st.error("The 'Close' column is missing from the data. Please check the ticker symbol.")
    st.stop()

# Ensure 'Close' column is numeric and handle any non-numeric values
try:
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
except Exception as e:
    st.error(f"Error converting 'Close' column to numeric: {e}")
    st.stop()

# Drop rows with NaN values in 'Close'
data = data.dropna(subset=['Close'])

# Check for minimum data points
if len(data) < 14:
    st.error("Not enough data points for SMA calculation. Please select a larger date range.")
    st.stop()

# Add technical indicators
data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# Trend column (1 if next day up, 0 if down)
data['Trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# Features and target
X = data[['SMA', 'RSI']]
y = data['Trend']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2f}")
st.text(classification_report(y_test, y_pred))

# Latest prediction
latest = X.tail(1)
pred = model.predict(latest)[0]
trend = "🔼 Uptrend Expected" if pred == 1 else "🔽 Downtrend Expected"

# Display charts and prediction
st.subheader(f"Close Price Chart for {ticker}")
st.line_chart(data['Close'])

st.subheader("Latest Prediction")
st.success(trend)

st.subheader("Technical Indicators")
st.line_chart(data[['SMA', 'RSI']])

st.markdown(
    '''
---
*Note: The ticker list is a sample. For a comprehensive live stock list, replace the `get_stock_tickers` function with a method to load tickers from a reliable source such as an updated CSV file or an API.*
'''
)
