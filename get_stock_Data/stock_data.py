# install the library
# pip install yfinance 

import yfinance as yf
from datetime import datetime

apple = yf.Ticker("AAPL")
stock =  apple.info
stock_data = apple.history(period="1d")
splits = apple.splits





data = apple.history(start ="2020-01-01", end = "2024-12-31",interval="1d",auto_adjust=True)
stock_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']]
splits_data = splits.to_frame(name='Splits')
stock_data = stock_data.join(splits_data, how='left')
stock_data['Date'] =stock_data.index.date
stock_data.to_json("get_stock_Data/stock_one_day" + ".json",orient="records", date_format="iso")






