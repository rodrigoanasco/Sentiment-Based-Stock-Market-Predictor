import pandas as pd

# 1. Load the CSV file with date parsing
df = pd.read_csv('aapl_daily_complete_scaled.csv', index_col=0, parse_dates=True)

# 2. Extract only the consumer_sentiment col
sentiment_data = df[['consumer_sentiment']].copy()

# 3. Verify the extraction
print("First 5 sentiment values:")
print(sentiment_data.head())

print("\nLast 5 sentiment values:")
print(sentiment_data.tail())

# 4. Save to new CSV file
sentiment_data.to_csv('consumer_sentiment_daily.csv')
print("\nSaved consumer sentiment data to 'consumer_sentiment_daily.csv'")

# 5. Not necessary just for testing
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
sentiment_data.plot(title='Daily Consumer Sentiment (-1 to 1)')
plt.savefig('consumer_sentiment_plot.png')
plt.close()