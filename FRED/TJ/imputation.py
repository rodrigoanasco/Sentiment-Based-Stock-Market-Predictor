import pandas as pd
import numpy as np

def safe_impute(df):
    """Robust imputation with fallbacks"""
    try:
        # Try quadratic interpolation if SciPy available
        from scipy import interpolate
        df['corp_profits'] = df['corp_profits'].interpolate(method='quadratic')
    except ImportError:
        # Fallback to polynomial interpolation if SciPy missing
        df['corp_profits'] = df['corp_profits'].interpolate(method='polynomial', order=2)
    
    return df

def impute_data(df):
    # Ensure monthly frequency
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
    df = df.reindex(full_index)
    
    # Stock Price - Time-based linear
    df['AAPL_Price'] = df['AAPL_Price'].interpolate(method='time')
    
    # Economic Indicators - Forward fill
    econ_cols = ['consumer_sentiment', 'dollar_index', 'retail_sales']
    df[econ_cols] = df[econ_cols].ffill()
    
    # Corporate Profits - Special handling
    if 'corp_profits' in df.columns:
        df = safe_impute(df)
    
    # Add missing flags
    for col in df.columns:
        if df[col].isnull().any():
            df[f'{col}_was_missing'] = df[col].isnull().astype(int)
    
    return df

# Load and process
df = pd.read_csv('aapl_sentiment_correlation.csv', index_col=0, parse_dates=True)
df_imputed = impute_data(df)
df_imputed.to_csv('aapl_imputed_complete.csv')

print("Missing values after imputation:")
print(df_imputed.isnull().sum())