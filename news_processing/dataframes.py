import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Import CSV Files 
text_blob_2020 = pd.read_csv("../news_processing/finace_article/financial_2020_2020-01-01.csv")
finbert_2020 = pd.read_csv("../news_processing/using_finbert/financial_2020-01-01.csv")

text_blob_2021 = pd.read_csv("../news_processing/finace_article/financial_2020_2021-01-01.csv")
finbert_2021 = pd.read_csv("../news_processing/using_finbert/financial_2021-01-01.csv")

text_blob_2022 = pd.read_csv("../news_processing/finace_article/financial_2022_2022-01-01.csv")
finbert_2022 = pd.read_csv("../news_processing/using_finbert/financial_2022-01-01.csv")

text_blob_2023 = pd.read_csv("../news_processing/finace_article/financial_2023_2023-01-01.csv")
finbert_2023 = pd.read_csv("../news_processing/using_finbert/financial_2023-01-01.csv")

text_blob_2024 = pd.read_csv("../news_processing/finace_article/financial_2024_2024-01-01.csv")
finbert_2024 = pd.read_csv("../news_processing/using_finbert/financial_2024-01-01.csv")

#  Adjust TextBlob Sentiment Labels 
for df in [text_blob_2020, text_blob_2021, text_blob_2022, text_blob_2023, text_blob_2024]:
    df.loc[df['Score'] < 0.05, 'Sentiment'] = 'Negative'
    df.loc[df['Sentiment'] == 'Negative', 'Score'] = df['Score'].apply(lambda x: -abs(x) if pd.notna(x) else x)

#  Adjust FinBERT Negative Scores
for df in [finbert_2020, finbert_2021, finbert_2022, finbert_2023, finbert_2024]:
    df.loc[df['Sentiment'] == 'negative', 'Score'] = df['Score'].apply(lambda x: -abs(x) if pd.notna(x) else x)

# Train Regression to Align TextBlob to FinBERT
tb20 = text_blob_2020[text_blob_2020['Sentiment'].str.lower().isin(['positive', 'negative', 'neutral'])].copy()
fb20 = finbert_2020[finbert_2020['Sentiment'].str.lower().isin(['positive', 'negative', 'neutral'])].copy()

tb20['Source'] = 'TextBlob'
fb20['Source'] = 'FinBERT'

combined = pd.concat([tb20, fb20], ignore_index=True)
combined.sort_values(by=["Date", "Title"], inplace=True)
combined['RowID'] = combined.groupby(['Date', 'Title']).cumcount()

pivoted = combined.pivot_table(
    index=['Date', 'Title'],
    columns='Source',
    values='Score',
    aggfunc='first'
).reset_index()

pivoted.dropna(inplace=True)

X = pivoted['TextBlob'].values.reshape(-1, 1)
y = pivoted['FinBERT'].values
reg = LinearRegression().fit(X, y)

# Scale TextBlob Scores with Preserved Sign and Clip to [-1, 1]
all_textblob_scaled = []

for year, df in {
    2020: text_blob_2020,
    2021: text_blob_2021,
    2022: text_blob_2022,
    2023: text_blob_2023,
    2024: text_blob_2024,
}.items():
    df = df.copy()

    signs = np.sign(df['Score'].values)
    scaled = reg.predict(np.abs(df['Score'].values).reshape(-1, 1))
    df['Score'] = (scaled * signs).clip(-1, 1)

    df.loc[df['Title'].isna(), 'Score'] = 0.0
    df.loc[df['Title'].isna(), 'Sentiment'] = 'Neutral'

    all_textblob_scaled.append(df[['Date', 'Title', 'Score', 'Sentiment']])

textblob_all_years = pd.concat(all_textblob_scaled, ignore_index=True)
textblob_all_years['Source'] = 'TextBlob_scaled'

# Prepare FinBERT All Years Combined
finbert_all_years = pd.concat([
    finbert_2020, finbert_2021, finbert_2022, finbert_2023, finbert_2024
], ignore_index=True)[['Date', 'Title', 'Score', 'Sentiment']]
finbert_all_years['Source'] = 'FinBERT'

# Merge TextBlob and FinBERT
merged = pd.concat([textblob_all_years, finbert_all_years], ignore_index=True)
merged.sort_values(by=["Date", "Title"], inplace=True)

# Ensure all dates covered
all_dates = pd.date_range(start='2020-01-01', end='2024-12-31')
full_index = pd.DataFrame({'Date': all_dates})
merged['Date'] = pd.to_datetime(merged['Date'])
complete = full_index.merge(merged, on='Date', how='left')

# Convert sentiment to signed score
complete['Sentiment'] = complete['Sentiment'].str.lower()
complete.loc[complete['Sentiment'] == 'negative', 'Score'] *= -1

# Save full dataset
complete[['Date', 'Title', 'Score', 'Sentiment']].to_csv("processed_data/FINAL_merged_with_all_dates.csv", index=False)

# Save score-only column 
score_only = complete[['Score']]
score_only.to_csv("processed_data/NEWS_SCORE_COLUMN_only.csv", index=False)

# Average score per day
daily_avg = complete.groupby('Date', as_index=False)['Score'].mean()
daily_avg.to_csv("processed_data/DAILY_AVG_SCORE.csv", index=False)
