from fredapi import Fred
import pandas as pd
import yfinance as yf

# Initialize FRED with your API key
fred = Fred(api_key='c8caefee70ea6ac131dcf6c25a634339')  # Replace with your key

# Economic indicators that act as sentiment proxies
SENTIMENT_PROXIES = {
    'consumer_sentiment': 'UMCSENT',       # Direct sentiment measure
    'dollar_index': 'DTWEXBGS',            # FX markets' view
    'corp_profits': 'CP',                  # Corporate profits (tech sector)
    'retail_sales': 'MRTSSM453USN'         # Electronics retail demand
}

# Date range
START_DATE = '2020-01-01'
END_DATE = '2024-12-30'

def get_sentiment_proxies():
    """Fetch economic indicators that correlate with market sentiment"""
    data = pd.DataFrame()
    
    for name, series_id in SENTIMENT_PROXIES.items():
        try:
            s = fred.get_series(series_id, START_DATE, END_DATE)
            s.index = s.index.tz_localize(None)  # Remove timezone
            data[name] = s
            print(f"✓ {name} ({series_id}): {s.index[0].date()} to {s.index[-1].date()}")
        except Exception as e:
            print(f"✗ Failed {name} ({series_id}): {str(e)}")
    
    return data.ffill().dropna()

def get_aapl_stock():
    """Get AAPL stock prices with timezone handling"""
    try:
        aapl = yf.Ticker("AAPL")
        hist = aapl.history(start=START_DATE, end=END_DATE)
        hist.index = hist.index.tz_localize(None)
        return hist['Close']
    except Exception as e:
        print(f"Failed to get AAPL data: {str(e)}")
        return pd.Series()

# Fetch and combine data
sentiment_data = get_sentiment_proxies()
aapl_prices = get_aapl_stock()

combined = pd.concat([aapl_prices, sentiment_data], axis=1).dropna()
combined.columns = ['AAPL_Price'] + list(SENTIMENT_PROXIES.keys())
combined.to_csv('aapl_sentiment_correlation.csv')

print("\nFirst 5 rows:")
print(combined.head())