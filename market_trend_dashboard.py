import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Title
st.title("📈 Market Trend Predictor (Basic Demo)")

# Sidebar for stock selection
ticker = st.sidebar.text_input("Enter Stock Ticker", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Load data
data = yf.download(ticker, start=start_date, end=end_date)

# Debug: Show columns to understand structure
st.write("Data columns:", data.columns)

# Handle multi-index columns if present
if isinstance(data.columns, pd.MultiIndex):
    # Flatten multi-index columns by joining with underscore
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

# After flattening, try to locate 'Close' or 'Close_<ticker>' column
close_col_candidates = [col for col in data.columns if 'Close' in col]
if not close_col_candidates:
    st.error("No 'Close' column found in the data, please check the ticker symbol and data availability.")
    st.stop()

# Use the first candidate 'Close' column
close_col = close_col_candidates[0]

# Ensure 'Close' column is numeric and handle any non-numeric values
try:
    data[close_col] = pd.to_numeric(data[close_col], errors='coerce')
except Exception as e:
    st.error(f"Error converting '{close_col}' column to numeric: {e}")
    st.stop()

# Drop rows with NaN values in 'Close'
data = data.dropna(subset=[close_col])

# Check if there are enough data points for SMA (window=14)
if len(data) < 14:
    st.error("Not enough data points for SMA calculation. Please select a larger date range.")
    st.stop()

# Add technical indicators on close_col
data['SMA'] = ta.trend.sma_indicator(data[close_col], window=14)
data['RSI'] = ta.momentum.rsi(data[close_col], window=14)

# Trend column (1 if next day is up, 0 if down)
data['Trend'] = (data[close_col].shift(-1) > data[close_col]).astype(int)
data = data.dropna()

# Features and target for the model
X = data[['SMA', 'RSI']]
y = data['Trend']

# Train a RandomForest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy:.2f}")
st.text(classification_report(y_test, y_pred))

# Prediction on latest data
latest = X.tail(1)
pred = model.predict(latest)[0]
trend = "🔼 Uptrend Expected" if pred == 1 else "🔽 Downtrend Expected"

# Display results
st.subheader("Close Price Chart")
st.line_chart(data[close_col])

st.subheader("Latest Prediction")
st.success(trend)

st.subheader("Technical Indicators")
st.line_chart(data[['SMA', 'RSI']])
