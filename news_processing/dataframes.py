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

#2023
df_2023 = pd.concat([text_blob_2023, finbert_2023], ignore_index=True)
df_2023.sort_values(by=["Date", "Title"], inplace=True)

#2024
df_2024 = pd.concat([text_blob_2024, finbert_2024], ignore_index=True)
df_2024.sort_values(by=["Date", "Title"], inplace=True)


''' Linear Regression set up '''
# Filter valid sentiment labels only
text_blob_2020 = text_blob_2020[text_blob_2020['Sentiment'].str.lower().isin(['positive', 'negative', 'neutral'])]
finbert_2020 = finbert_2020[finbert_2020['Sentiment'].str.lower().isin(['positive', 'negative', 'neutral'])]

# Add a Source label
text_blob_2020['Source'] = 'TextBlob'
finbert_2020['Source'] = 'FinBERT'

# Combine and sort
combined = pd.concat([text_blob_2020, finbert_2020], ignore_index=True)
combined.sort_values(by=["Date", "Title"], inplace=True)

# Create RowID to help match rows
combined['RowID'] = combined.groupby(['Date', 'Title']).cumcount()

# Pivot to get one row per (Date, Title) with TextBlob and FinBERT scores side by side
paired = combined.pivot_table(
    index=['Date', 'Title'],
    columns='Source',
    values='Score',
    aggfunc='first'
).reset_index()

# Training(ish)
paired.dropna(inplace=True)

X = paired['TextBlob'].values.reshape(-1, 1)
y = paired['FinBERT'].values

reg = LinearRegression().fit(X, y)

print(f"Learned mapping: FinBERT = {reg.coef_[0]:.4f} x TextBlob + {reg.intercept_:.4f}")

# Apply the learned mapping to the TextBlob entries
text_blob_2020['Score_scaled'] = reg.predict(text_blob_2020['Score'].values.reshape(-1, 1)).clip(0, 1)

# Exporting side to side in .csv text_blob without scaling and text_blob with scaling
text_blob_2020.loc[text_blob_2020['Title'].isna(), 'Score_scaled'] = 0.0
text_blob_2020.loc[text_blob_2020['Title'].isna(), 'Sentiment'] = 'Neutral'
text_blob_2020[['Date', 'Title', 'Score', 'Score_scaled', 'Sentiment', 'language']].to_csv("textblob_2020_scaled.csv", index=False)


# Grouping by name


print(df_2020)
print(df_2021)
print(df_2022)
print(df_2023)
print(df_2024)



