import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the prepared data, with impuattion no missing montlhy val
data = pd.read_csv('aapl_imputed_complete.csv', index_col=0, parse_dates=True)

# 1. Calculateing correlations 
correlations = data.corr()['AAPL_Price'].sort_values(ascending=False)
print("Correlation with AAPL Price:")
print(correlations)

# 2. Generate correlation matrix plot
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('AAPL Price vs Sentiment Proxies Correlation')
plt.tight_layout()
plt.savefig('aapl_sentiment_correlation.png')
plt.show()

# 3. Time-series plot of key relationships
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Top positive correlation
top_positive = correlations.index[1]  # indicator with the highes correlation val with aapl price 
data[['AAPL_Price', top_positive]].plot(ax=axes[0], secondary_y=top_positive)
axes[0].set_title(f'AAPL Price vs {top_positive}')

# Top negative correlation
top_negative = correlations.index[-1] # indicator with the lowest correlation val with aapl price 
data[['AAPL_Price', top_negative]].plot(ax=axes[1], secondary_y=top_negative)
axes[1].set_title(f'AAPL Price vs {top_negative}')

plt.tight_layout()
plt.savefig('aapl_sentiment_timeseries.png')
plt.show()