import pandas as pd


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

print(df_2020)
print(df_2021)
print(df_2022)
print(df_2023)
print(df_2024)