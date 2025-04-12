import pandas as pd
import matplotlib.pyplot as plt

# Load the completed file
df = pd.read_csv("DAILY_AVG_SCORE.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Sort for safety
df.sort_values(by='Date', inplace=True)

# Create a 7-day rolling mean
df['RollingScore'] = df['Score'].rolling(window=7, min_periods=1).mean()

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['RollingScore'], label='7-Day Rolling Sentiment Score', color='dodgerblue')

plt.title("Financial News Sentiment Over Time (2020–2024)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("sentiment_over_time.png", dpi=300)


