import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Debug info function
def print_df_info(df, name):
    print(f"\n--- {name} Info ---")
    print(f"Shape: {df.shape}")
    print(f"NaN counts:\n{df.isna().sum()}")
    if not df.empty:
        print(f"First few rows:\n{df.head(3)}")
    else:
        print("DataFrame is EMPTY!")

# Load the data with debugging
try:
    day = pd.read_json("stock_one_day.json")
    print_df_info(day, "Original Data")
except FileNotFoundError:
    print("File not found")
    exit(1)
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit(1)

# Check data structure
print("\nDataFrame columns:", day.columns.tolist())
print("\nDataFrame index:", day.index)

# Create features with checks along the way
# Let's check for NaN values after each transformation

# First transformation
day['daily_return'] = day["Close"].pct_change()
print_df_info(day[['Close', 'daily_return']], "After daily_return")

# Handle problematic rows immediately
day['daily_return'] = day['daily_return'].fillna(0)

day['Price_Range'] = day['High'] - day['Low']
day['SMA_3'] = day['Close'].rolling(window=3).mean()
print_df_info(day[['Close', 'SMA_3']], "After SMA_3")

day['SMA_5'] = day['Close'].rolling(window=5).mean()
day['EMA_3'] = day['Close'].ewm(span=3, adjust=False).mean()
day['Volume_Change'] = day['Volume'].pct_change()
day['Volume_Change'] = day['Volume_Change'].fillna(0)  # Fill NaN in first row

day['Momentum_3'] = day['Close'] - day['Close'].shift(3)
print_df_info(day[['Close', 'Momentum_3']], "After Momentum_3")

# RSI-7 calculation with careful handling
delta = day['Close'].diff()
delta = delta.fillna(0)  # Handle first NaN
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta <= 0, 0)
avg_gain = gain.rolling(window=7).mean()
avg_loss = loss.rolling(window=7).mean()
# Add small epsilon to avoid division by zero
avg_loss_safe = avg_loss.copy()
avg_loss_safe[avg_loss_safe == 0] = 1e-10
rs = avg_gain / avg_loss_safe
day['RSI_7'] = 100 - (100 / (1 + rs))
print_df_info(day[['Close', 'RSI_7']], "After RSI_7")

# Target variable
day['Target'] = (day['Close'].shift(-1) > day['Close']).astype(int)
print_df_info(day[['Close', 'Target']], "After Target")

# Before dropna, show how many rows will be dropped
nan_count = day.isna().any(axis=1).sum()
print(f"\nRows with any NaN values: {nan_count} out of {len(day)}")

# Let's keep more data by handling NaNs differently
features = [
    'daily_return', 'Price_Range', 'SMA_3', 'SMA_5',
    'EMA_3', 'Volume_Change', 'Momentum_3', 'RSI_7',
]

# Instead of dropping NaN values, let's handle them appropriately
# 1. For SMA and EMA, use forward fill
for col in ['SMA_3', 'SMA_5', 'EMA_3']:
    day[col] = day[col].fillna(method='ffill')

# 2. For Momentum, RSI, use 0 for initial values
for col in ['Momentum_3', 'RSI_7']:
    day[col] = day[col].fillna(0)

# Now check if we have NaN values in our features or target
X = day[features]
y = day['Target']

print_df_info(X, "Features after handling NaNs")
print_df_info(pd.DataFrame(y), "Target after handling NaNs")

# Handle any remaining NaNs
X = X.fillna(0)
y = y.fillna(0)  # For the last target value that might be NaN

# Verify no NaNs remain
assert not X.isna().any().any(), "There are still NaN values in features"
assert not y.isna().any(), "There are still NaN values in target"

# Now check if we have enough data
if len(X) < 10:  # Arbitrary minimum threshold
    print("WARNING: Very small dataset, results may be unreliable")

# Train-test split (80/20)
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"\nTraining set: {X_train.shape} samples")
print(f"Test set: {X_test.shape} samples")

# Only proceed if we have data
if X_train.shape[0] > 0 and X_test.shape[0] > 0:
    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # Ensemble prediction
    ensemble_pred = ((y_pred_rf.astype(int) + y_pred_xgb.astype(int)) >= 1).astype(int)
    
    # Performance metrics
    print("\nRandom Forest accuracy:", accuracy_score(y_test, y_pred_rf))
    print("XGBoost accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("Ensemble accuracy:", accuracy_score(y_test, ensemble_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred))

    # Plot feature importance
    importance_rf = pd.Series(rf_model.feature_importances_, index=features)
    importance_xgb = pd.Series(xgb_model.feature_importances_, index=features)
    importance_combined = (importance_rf + importance_xgb) / 2
    importance_combined = importance_combined.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    importance_combined.plot(kind='bar')
    plt.title('Important Features')
    plt.tight_layout()
    plt.savefig('ensemble_feature_importance.png')
    print("Feature importance plot saved as 'ensemble_feature_importance.png'")
    
    # Print top features
    print("\nTop 5 Features by Importance:")
    for feature, importance in importance_combined.head(5).items():
        print(f"{feature}: {importance:.4f}")
else:
    print("ERROR: Not enough data for training after handling NaNs")
    print("Possible issues:")
    print("1. Your JSON file may be empty or corrupted")
    print("2. All rows might have been removed during feature calculation")
    print("3. The data might not span enough days for the features requiring lookback periods")