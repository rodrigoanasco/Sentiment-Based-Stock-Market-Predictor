import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# load the data

try:
    news = pd.read_csv("news_processing/DAILY_AVG_SCORE.csv")
    day = pd.read_json("get_stock_Data/stock_one_day.json")
    #gov = pd.read_csv("FRED/TJ/consumer_sentiment_daily.csv")
except FileNotFoundError:
    print("file not found")
    exit(1)

day['Date'] = pd.to_datetime(day['Date'])
news['Date'] = pd.to_datetime(news['Date'])
# gov.reset_index(inplace=True)
# gov.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
# gov['Date'] = pd.to_datetime(gov['Date'])
#change the names sentiment1 sentiment2 sentiment3

# good features and highest yield found
day['daily_return'] = day["Close"].pct_change()
day['Price_Range'] = day['High'] - day['Low']
day['SMA_3'] = day['Close'].rolling(window=3).mean()
day['SMA_5'] = day['Close'].rolling(window=5).mean()
day['EMA_3'] = day['Close'].ewm(span=3, adjust=False).mean()
day['Volume_Change'] = day['Volume'].pct_change()
day['Momentum_3'] = day['Close'] - day['Close'].shift(3)

#https://blog.quantinsti.com/rsi-indicator/
# getting RSI-7
delta = day['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta <= 0, 0)
avg_gain = gain.rolling(window=7).mean()
avg_loss = loss.rolling(window=7).mean()
rs = avg_gain / avg_loss
day['RSI_7'] = 100 - (100 / (1 + rs))

# what is typeing to be guessing
day['Target'] = (day['Close'].shift(-1) > day['Close']).astype(int)
day['Volatility_5'] = day['daily_return'].rolling(window=5).std()
day = pd.merge(day,news,on='Date', how='inner')


features = [
    'daily_return', 'Price_Range', 'SMA_3', 'SMA_5', 
    'EMA_3', 'Volume_Change', 'Momentum_3', 'RSI_7',"Score", 'Volatility_5',
]
X = day[features]
y = day['Target']
X = X.ffill().bfill()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split (80/20)
split_index = int(len(X_scaled) * 0.8) # for continous data
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]



rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


xgb_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print("STOCK DATA + NEWS ARTICLES")
print(classification_report(y_test, y_pred))


# plot data






