import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("📈 Market Trend Predictor (Basic Demo)")

# Sidebar for stock selection
ticker = st.sidebar.text_input("Enter Stock Ticker", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Load data
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("No data found. Please check the ticker symbol.")
    st.stop()

# Technical indicators
data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# Trend column (1 if next day is up, 0 if down)
data['Trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# Features and target
X = data[['SMA', 'RSI']]
y = data['Trend']

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

# Prediction on latest data
latest = X.tail(1)
pred = model.predict(latest)[0]
trend = "🔼 Uptrend Expected" if pred == 1 else "🔽 Downtrend Expected"

# Display results
st.subheader("Close Price Chart")
st.line_chart(data['Close'])

st.subheader("Latest Prediction")
st.success(trend)

st.subheader("Technical Indicators")
st.line_chart(data[['SMA', 'RSI']])
# Fill NaN values with the previous available value (or use another method)
data['Close'].fillna(method='ffill', inplace=True)

# Apply the SMA and other indicators
data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
