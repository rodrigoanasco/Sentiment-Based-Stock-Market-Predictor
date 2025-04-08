import pandas as pd
import numpy as np
import argparse
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

'''
This file contains the code used to reorganize, clean, and combine the data from reddit with the apple stock data.
The data is once again exported and saved as a CSV for usage in the upcoming analysis (reddit_analyze_data).
'''

# Load FinBERT
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)
    
# ---- Sentiment Classification (FinBERT) ----
'''
Function that calls finbert (trained model) to evaulate the sentiment of each combined post of the day
-> More accurate than other models like textblob
'''
def get_sentiment(text):
    try:
        result = finbert_pipeline(text[:512])[0]  
        label = result["label"]
        score = result["score"]

        # Case: positive
        if label == "positive":
            return score, "Positive"
        
        # Case: negative
        elif label == "negative":
            return -score, "Negative"
        
        # Case: neutral
        else:
            return 0.0, "Neutral"
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        return 0.0, "Neutral"

# ---- Parser Function ----
'''
Function that receives user input and parses it
Needed to access year parameter
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Process Reddit sentiment data.')
    parser.add_argument('input_csv', help='Input CSV file path')
    return parser.parse_args()

# ---- Apple Data Retriever Function ----
'''
Function that calls yfinance to get day to day stock data
Flattens, extracts, and recomputes the data
'''
def fetch_apple_stock_data(year):
    print(f"📈 Fetching AAPL stock data for {year}")
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    stock_data = yf.download("AAPL", start=start_date, end=end_date, auto_adjust=False)

    # Flatten MultiIndex columns when needed
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = [' '.join(col).strip() for col in stock_data.columns.values]

    stock_data.columns = [col.replace(" AAPL", "") for col in stock_data.columns]

    # Ensure the necessary columns exist
    required_columns = ['Open', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in stock_data.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in stock data: {missing_cols}")

    # Extract and compute new column
    stock_data = stock_data[required_columns]
    stock_data['Percent_Diff'] = ((stock_data['Close'] - stock_data['Open']) / stock_data['Open']) * 100

    # Reset and set to datetime for proper merging later
    stock_data = stock_data.reset_index()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    return stock_data

# ---- Main Processing Function ----
'''
Contains the code to process the data
Reads the user-inputted csv file, and combines with stock data
Fills in any empty dates/entries for better analysis 
'''
def process_data(input_csv):

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Strip spaces from column names to avoid issues
    df.columns = df.columns.str.strip()

    # Check if 'Date' column exists
    if 'Date' not in df.columns:
        print("Error: 'Date' column not found in the CSV file.")
        return

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Remove rows where 'Date' could not be converted (invalid dates)
    df = df.dropna(subset=['Date'])

    # Combine full post text (Title + selftext) by date
    daily_texts = df.groupby('Date')['Text'].apply(lambda texts: ' '.join(str(t) for t in texts)).reset_index()

    # Run FinBERT on each day's combined text
    scores, sentiments = [], []
    for _, row in daily_texts.iterrows():
        score, sentiment = get_sentiment(row['Text'])
        scores.append(score)
        sentiments.append(sentiment)

    # Add results to DataFrame
    daily_texts['Score'] = scores
    daily_texts['Sentiment'] = sentiments
    df_grouped = daily_texts[['Date', 'Score', 'Sentiment']]

    # Get the year from the data
    year = df_grouped['Date'].dt.year.min()

    # Generate full date range from Jan 1 to Dec 31 of that year
    all_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')

    # Reindex with all dates, filling with NaN for missing dates (ensures proper merge)
    df_grouped = df_grouped.set_index('Date').reindex(all_dates, fill_value=np.nan).reset_index()
    df_grouped = df_grouped.rename(columns={'index': 'Date'})

    # Fetch Apple stock data and merge on 'date'
    stock_df = fetch_apple_stock_data(year)
    merged_df = pd.merge(df_grouped, stock_df, on='Date', how='left')

    # Replace empty values in all columns after merging with NaN
    merged_df = merged_df.fillna('NaN')

    # Save the final merged dataset 
    output_csv = f"cleaned_reddit_{year}.csv"
    merged_df.drop(columns=['Language', 'Text'], inplace=True, errors='ignore')     # Drop Language and Text columns
    merged_df.to_csv(output_csv, index=False)
    print(f"Final merged data saved as: {output_csv}")

if __name__ == '__main__':
    # Parse arguments from the command line
    args = parse_args()

    # Process the data
    process_data(args.input_csv)