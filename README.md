# 353-Project
Determining in which way the sentiment affect stock market

## News Processing


### Finace Article (TextBlob)
### `main.py`
Dependencies

To install all required libraries, run:
```
pip install gdeltdoc pandas tqdm transformers beautifulsoup4 requests scikit-learn textblob
```

### 1. Run with TextBlob

The following command runs the default pipeline from January 1 to December 31, 2024:

```
python main.py
```

This generates a CSV file named:

```
financial_2024_2024-01-01.csv
```

To adjust the date range, modify the last line in `main.py`:

```python
main(date(2024, 1, 1), date(2024, 12, 31))
```

### `parameters.py`
## What It Does

This file contains utility functions used for:

- Checking if article URLs come from **credible sources**
- Verifying if a website is **online**
- **Archiving fallback** with the Wayback Machine if needed
- Scraping article content using `BeautifulSoup`
- Running **TextBlob** sentiment analysis on content
- **Parallel URL processing** for performance



## Dependencies

Install required packages if not already done:

```
pip install gdeltdoc pandas beautifulsoup4 textblob requests lxml
```


### Using Finbert (Finbert)
## Dependencies:
Install all necessary packages with:

```
pip install gdeltdoc pandas tqdm transformers beautifulsoup4 requests scikit-learn lxml
```

FinBERT uses Hugging Face's Transformers library, so the `transformers` package is essential.

## What it does:

### `main(start, end)`

Processes articles over a date range, stores results into a CSV.

### `process_date(current_date)`

Fetches articles for one day, runs FinBERT sentiment, and formats output.

## dataframes.py
## Dependencies

Install the required libraries if not already installed:

```
pip install pandas scikit-learn numpy
```

## What It Does

1. **Loads sentiment data** from CSV files for both TextBlob and FinBERT across years 2020 to 2024.
2. **Adjusts sentiment polarity**:
   - Ensures negative sentiment scores are negative values.
   - Uses regression to map TextBlob scores onto the FinBERT scale.
3. **Scales TextBlob scores** using a Linear Regression fit between TextBlob and FinBERT scores from 2020.
4. **Combines and merges all data** into a single DataFrame with full date coverage.
5. **Generates and saves** the following outputs:
   - `FINAL_merged_with_all_dates.csv`: Main dataset with all articles, scores, and sentiments.
   - `NEWS_SCORE_COLUMN_only.csv`: Sentiment score only (for model input or graphing).
   - `DAILY_AVG_SCORE.csv`: Daily average sentiment score.



## processing_data.py
## Dependencies

Install required libraries using:

```
pip install pandas matplotlib
```

## What It Does

1. Loads `DAILY_AVG_SCORE.csv`, which should include average sentiment scores per day from 2020 to 2024.
2. Calculates a **7-day rolling mean** to smooth fluctuations in sentiment.
3. Generates a line plot showing sentiment over time.
4. Saves the figure as a high-resolution PNG:
