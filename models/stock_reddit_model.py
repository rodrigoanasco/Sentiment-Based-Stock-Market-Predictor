import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

'''
This file contains the code used to analyze Reddit datasets through machine learning models.
Reddit sentiment is combined with historical stock data
'''

# Load data from CSV files
try:
    day = pd.read_json("get_stock_Data/stock_one_day.json")
    reddit = pd.read_csv("reddit_article/combined_reddit.csv")

except FileNotFoundError:
    print("file not found")
    exit(1)

# Make sure the formats match
reddit['Date'] = pd.to_datetime(reddit['Date'])
day['Date'] = pd.to_datetime(day['Date'])  

# Feature Engineering
day['daily_return'] = day["Close"].pct_change()
day['Price_Range'] = day['High'] - day['Low']
day['SMA_3'] = day['Close'].rolling(window=3).mean()
day['SMA_5'] = day['Close'].rolling(window=5).mean()
day['EMA_3'] = day['Close'].ewm(span=3, adjust=False).mean()
day['Volume_Change'] = day['Volume'].pct_change()
day['Momentum_3'] = day['Close'] - day['Close'].shift(3)

# RSI (7-day)
delta = day['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta <= 0, 0)
avg_gain = gain.rolling(window=7).mean()
avg_loss = loss.rolling(window=7).mean()
rs = avg_gain / avg_loss
day['RSI_7'] = 100 - (100 / (1 + rs))

day['Lag_Close_1'] = day['Close'].shift(1)
day['Volatility_5'] = day['daily_return'].rolling(window=5).std()

# Target variable
day['Target'] = (day['Close'].shift(-1) > day['Close']).astype(int)

# Merge with Reddit
day_reddit = pd.merge(day, reddit, on='Date', how='inner')

# Features
features = [
    'daily_return', 'Price_Range', 'SMA_3', 'SMA_5', 'EMA_3',
    'Volume_Change', 'Momentum_3', 'RSI_7', 'Lag_Close_1', 'Volatility_5', 'Score'
]

X = day_reddit[features]
y = day_reddit['Target']

# Handle missing values
X = X.ffill().bfill()

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
split_index = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Random forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42
)

# xgb model
xgb_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)

# Voting ensemble 
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    voting='soft'
)

# Train
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# Report
print("STOCK + REDDIT DATA")
print(classification_report(y_test, y_pred))