import pandas as pd
import matplotlib.pyplot as plt

# Load full sentiment data
all_sentiment_df = pd.read_csv("DAILY_AVG_SCORE.csv")
all_sentiment_df['Date'] = pd.to_datetime(all_sentiment_df['Date'])
all_sentiment_df.sort_values(by='Date', inplace=True)

# Plot: Full 2020–2024 Sentiment 
plt.figure(figsize=(14, 6))
plt.plot(all_sentiment_df['Date'], all_sentiment_df['Score'], label='Daily Sentiment Score', color='mediumseagreen', alpha=0.7)
plt.ylim(-1, 1)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Daily Financial News Sentiment (2020–2024)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("daily_sentiment_bipolar.png", dpi=300)
plt.show()

#  Prepare Sentiment Data: Only 2024 
sentiment_df = all_sentiment_df[all_sentiment_df['Date'].dt.year == 2024]

# Load stock market data
stock_columns = ['Date', 'Score', 'Sentiment', 'Open', 'Close', 'Volume', 'Percent_Diff']
stock_df = pd.read_csv("stock_market_price_2024.csv", names=stock_columns, header=0)
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df.sort_values(by='Date', inplace=True)

# Merge sentiment and stock data
merged_df = pd.merge(sentiment_df, stock_df, on='Date', how='inner')

# Plot: 2024 Sentiment vs Apple Price
fig, ax1 = plt.subplots(figsize=(14, 6))

# Left Y-axis: Sentiment
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment Score', color='tab:blue')
ax1.plot(merged_df['Date'], merged_df['Score_x'], color='tab:blue', label='Sentiment Score')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(-1, 1)
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

# Right Y-axis: Apple Close Price
ax2 = ax1.twinx()
ax2.set_ylabel('Apple Stock Price (Close)', color='tab:red')
ax2.plot(merged_df['Date'], merged_df['Close'], color='tab:red', linestyle='--', label='Close Price')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Finalize
plt.title("Sentiment vs Apple Stock Price (2024)", fontsize=16)
fig.tight_layout()
plt.grid(True)
plt.legend(loc='upper left')
plt.savefig("sentiment_vs_apple_price_2024.png", dpi=300)
plt.show()


correlation = merged_df['Score_x'].corr(merged_df['Close'])
print(f"Correlation between Sentiment and Apple Close Price (2024): {correlation:.4f}")
correlation = merged_df['Score_x'].corr(merged_df['Percent_Diff'])
print(f"Correlation between Sentiment and % Change (Open→Close) (2024): {correlation:.4f}")
