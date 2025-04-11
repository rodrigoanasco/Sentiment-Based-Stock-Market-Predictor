import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Importing csv files
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

# If sentiment is less than 0.05, turn sentiment to negative in text_blob
text_blob_2020.loc[text_blob_2020['Score'] < 0.05, 'Sentiment'] = 'Negative'
text_blob_2021.loc[text_blob_2021['Score'] < 0.05, 'Sentiment'] = 'Negative'
text_blob_2022.loc[text_blob_2022['Score'] < 0.05, 'Sentiment'] = 'Negative'
text_blob_2023.loc[text_blob_2023['Score'] < 0.05, 'Sentiment'] = 'Negative'
text_blob_2024.loc[text_blob_2024['Score'] < 0.05, 'Sentiment'] = 'Negative'

# Concatenating and sorting
# 2020
df_2020 = pd.concat([text_blob_2020, finbert_2020], ignore_index=True)
df_2020.sort_values(by=["Date", "Title"], inplace=True)

# 2021
df_2021 = pd.concat([text_blob_2021, finbert_2021], ignore_index=True)
df_2021.sort_values(by=["Date", "Title"], inplace=True)

# 2022
df_2022 = pd.concat([text_blob_2022, finbert_2022], ignore_index=True)
df_2022.sort_values(by=["Date", "Title"], inplace=True)

# 2023
df_2023 = pd.concat([text_blob_2023, finbert_2023], ignore_index=True)
df_2023.sort_values(by=["Date", "Title"], inplace=True)

# 2024
df_2024 = pd.concat([text_blob_2024, finbert_2024], ignore_index=True)
df_2024.sort_values(by=["Date", "Title"], inplace=True)


# Training Linear Regression on 2020 TextBlob vs FinBERT scores
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


# Scale TextBlob scores for all years
all_textblob_scaled = []

for year, df in {
    2020: text_blob_2020,
    2021: text_blob_2021,
    2022: text_blob_2022,
    2023: text_blob_2023,
    2024: text_blob_2024,
}.items():
    df = df.copy()
    df['Score'] = reg.predict(df['Score'].values.reshape(-1, 1)).clip(0, 1)
    df.loc[df['Title'].isna(), 'Score'] = 0.0
    df.loc[df['Title'].isna(), 'Sentiment'] = 'Neutral'
    all_textblob_scaled.append(df[['Date', 'Title', 'Score', 'Sentiment']])

# Add Source labels
textblob_all_years = pd.concat(all_textblob_scaled, ignore_index=True)
textblob_all_years['Source'] = 'TextBlob_scaled'

finbert_all_years = pd.concat([
    finbert_2020, finbert_2021, finbert_2022, finbert_2023, finbert_2024
], ignore_index=True)[['Date', 'Title', 'Score', 'Sentiment']]
finbert_all_years['Source'] = 'FinBERT'

# Merge all data
merged = pd.concat([textblob_all_years, finbert_all_years], ignore_index=True)
merged.sort_values(by=["Date", "Title"], inplace=True)

# Group and resolve duplicates
def resolve_group(group):
    if group['Sentiment'].nunique() == 1:
        return group.iloc[0]  # All sentiments the same, keep any
    else:
        return group.loc[group['Score'].idxmax()]  # Keep the one with highest score

final = merged.groupby(['Date', 'Title'], group_keys=False).apply(resolve_group)
final = final.reset_index(drop=True)

# Ensure full daily coverage between Jan 1, 2020 and Dec 31, 2024
all_dates = pd.date_range(start='2020-01-01', end='2024-12-31')

# Create a DataFrame with just dates
full_index = pd.DataFrame({'Date': all_dates})

# Merge on 'Date' to ensure every date appears
final['Date'] = pd.to_datetime(final['Date'])  # just in case
complete = full_index.merge(final, on='Date', how='left')


# Save the cleaned file
complete[['Date', 'Title', 'Score', 'Sentiment']].to_csv("merged_deduplicated_with_all_dates.csv", index=False)
