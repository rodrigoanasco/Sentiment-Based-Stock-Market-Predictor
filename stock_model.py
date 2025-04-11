import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


# load the data
try:
    day = pd.read_json("stock_one_day.json")
except FileNotFoundError:
    print("file not found")
    exit(1)

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


features = [
    'daily_return', 'Price_Range', 'SMA_3', 'SMA_5', 
    'EMA_3', 'Volume_Change', 'Momentum_3', 'RSI_7',
]

X = day[features]
y = day['Target']


# Train-test split (80/20)
split_index = int(len(X) * 0.8) # for continous data
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]



rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
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
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)


ensemble_pred = ((y_pred_rf.astype(int) + y_pred_xgb.astype(int)) >= 1).astype(int)
print(classification_report(y_test, ensemble_pred))


# figure out the important features
importance_rf = pd.Series(rf_model.feature_importances_, index=features)
importance_xgb = pd.Series(xgb_model.feature_importances_, index=features)

# Combine feature importances (average them)
importance_combined = (importance_rf + importance_xgb) / 2
importance_combined = importance_combined.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importance_combined.plot(kind='bar')
plt.title('Important Features')
plt.tight_layout()
plt.savefig('ensemble_feature_importance.png')
print("Feature importance plot saved as 'ensemble_feature_importance.png'")




