import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('aapl_imputed_complete.csv', index_col=0, parse_dates=True)

# 1. Forward fill initial missing values (beginning of dataset)
df = df.ffill()

# 2. Scale consumer sentiment to [-1, 1] range
scaler = MinMaxScaler(feature_range=(-1, 1))
df['consumer_sentiment'] = scaler.fit_transform(df[['consumer_sentiment']])

# 3. Create complete daily index
daily_index = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
df_daily = pd.DataFrame(index=daily_index).join(df, how='left')

# 4. Enhanced interpolation with edge handling
def prepare_daily_data(df):
    # Stock Price - Time interpolation with cubic spline
    df['AAPL_Price'] = df['AAPL_Price'].interpolate(method='time', limit_direction='both')
    
    # Economic Indicators - Hybrid approach
    econ_cols = ['consumer_sentiment', 'dollar_index', 'retail_sales']
    for col in econ_cols:
        # Linear interpolation first
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        # Then 7-day rolling average
        df[col] = df[col].rolling(7, min_periods=1, center=True).mean()
    
    # Corporate Profits - Quadratic with forward fill
    if 'corp_profits' in df.columns:
        df['corp_profits'] = df['corp_profits'].interpolate(method='quadratic').ffill()
    
    return df

# 5. Process data
df_daily = prepare_daily_data(df_daily)

# 6. Add volatility weights
df_daily['sentiment_weight'] = 1 / df_daily['AAPL_Price'].pct_change().rolling(21).std()

# 7. Verify first 3 months of 2020
print("First 10 days of 2020:")
print(df_daily.loc['2020-01-01':'2020-01-10', 
                  ['AAPL_Price', 'consumer_sentiment', 'dollar_index']])

# 8. Final checks
print("\nMissing values after processing:")
print(df_daily.isnull().sum())

# 9. Save
df_daily.to_csv('aapl_daily_complete_scaled.csv')