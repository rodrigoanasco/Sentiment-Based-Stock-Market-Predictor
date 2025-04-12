# 353-Project
Sentiment Analysis on Stock Market

(IMPORTANT NOTE: Due to Git push/pull errors, contributors do not show up properly. Obtain evidence of contribution by clicking on commit history.)

## Reddit Analysis
### Setup Instructions
Install all required packages:
```
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels yfinance transformers praw langdetect langcodes 
```
1. Get Data

Set up Reddit API (provided ones are personal and only work for 2 weeks after submission)

Adjust parameters if needed (changing time period of query parameters)

Then run the following command:
```
python reddit_get_data.py
```

Output:
- CSV file(s) between the specified years containing raw data

2. Cleaning Data

A CSV file is needed here, feel free to use the ones provided.

Run the following command (replace data.csv with actual file name, ie. reddit_2024_2024-01-01.csv):

```
python reddit_clean_data.py data.csv
```

Output:
- CSV file that contains organized and cleaned data

3. Analysis

A CSV file is needed here, feel free to use the ones provided.

You can input multiple CSV files if you wish to combine their data points.

Run the following command (replace data.csv with actual file name, ie. cleaned_reddit_2024.csv):

```
python reddit_analysis.py data.csv
```

Output:
- Pearson Correlation values on command line
- PNG images of the diagrams

4. Combine

Optional step - used for combining CSVs together to create a single larger one

Run the following code (replace data1.csv and data2.csv with actual file names):

```
python reddit_combine.py data1.csv data2.csv
```

Output:
- CSV file that contains one dataset appended to another (combined)

## News Processing
### Setup Instructions
Install all required packages:
```
pip install gdeltdoc pandas tqdm transformers beautifulsoup4 requests scikit-learn textblob lxml matplotlib
```
1. TextBlob-Based Pipeline
```
main.py
```
Fetches articles from GDELT and performs sentiment analysis using TextBlob.

Run:
```
python main.py
```
Output:

- financial_2024_2024-01-01.csv

To adjust the date range, edit:

main(date(2024, 1, 1), date(2024, 12, 31))

parameters.py – Utility Functions

This module supports the TextBlob pipeline with:

- Article scraping via requests + BeautifulSoup
- Credible domain filtering
- Archive fallback (Wayback Machine)
- Parallel sentiment analysis with concurrent.futures
- Sentiment classification using TextBlob

2. FinBERT-Based Pipeline
```
using_finbert/main.py
```
Runs the same process as main.py, but uses FinBERT from Hugging Face for sentiment analysis.

What It Does:

- Fetches news using GDELT
- Applies FinBERT to classify as positive, neutral, or negative
- Saves results per year (e.g., financial_2020-01-01.csv, ..., financial_2024-01-01.csv)

```
parameters_with_finbert.py
```
Contains:

- FinBERT pipeline setup
- Transformers-based tokenization and scoring
- Web scraping logic similar to the TextBlob version

Score Harmonization
```
dataframes.py
```
Aligns TextBlob scores to FinBERT’s sentiment scale via linear regression.

What It Does:

1. Loads sentiment CSVs for both TextBlob and FinBERT (2020–2024)
2. Makes negative sentiment scores negative (if not already)
3. Trains regression model to align TextBlob scores to FinBERT
4. Scales and clips scores to [-1, 1]
5. Outputs:

- FINAL_merged_with_all_dates.csv
- NEWS_SCORE_COLUMN_only.csv
- DAILY_AVG_SCORE.csv

### Sentiment Visualization

```
processing_data.py
```
Visualizes sentiment trends over time using a 7-day rolling average.

What It Does:

1. Loads DAILY_AVG_SCORE.csv
2. Applies 7-day smoothing to scores
3. Plots sentiment trend from 2020–2024
4. Saves output to:

sentiment_over_time.png
