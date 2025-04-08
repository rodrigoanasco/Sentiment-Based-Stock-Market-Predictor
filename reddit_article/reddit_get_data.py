import praw
import pandas as pd
from datetime import datetime
import time
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from langcodes import Language
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

'''
This file contains the code used to gather, clean (partially), and organize the data from reddit.
The data is exported and saved as a CSV for further cleaning in the next file (reddit_clean_data).
'''

# Load FinBERT model and tokenizer once
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)

# ---- Reddit API Setup ----
'''
Reddit -> Preferences -> Apps -> Create Application -> Script & Redirect uri = http://localhost:8080
After creating, fill in data below to connect to reddit API (id is right below name of application)
'''
reddit = praw.Reddit(
    client_id="Eg87PPxOl37V9_jcvci25g",
    client_secret="lhJeadQ-V6KMZzxvUe2exy2p1S9_tw",
    user_agent="script:apple_stock_scraper:v1.0 (by u/YOUR_USERNAME)"
)

# Error checking for proper connection of reddit
try:
    print(f"Authenticated as: {reddit.user.me()}")
    print(f"Accessing r/personalfinance...")
    posts = reddit.subreddit("personalfinance").hot(limit=5)
    for post in posts:
        print(f"Title: {post.title}")
except Exception as e:
    print("Authentication failed or issue with access:", e)

# ---- Config ----
'''
Key words to be used during the search
Add more to broaden the search, remove some to narrow the search
'''
query = (
    '("apple stock" OR AAPL OR "Apple Inc" OR "Tim Cook" OR iPhone OR MacBook '
    'OR iOS OR "tech stocks" OR "technology sector" OR "Silicon Valley" '
    'OR FAANG OR "Apple earnings" OR "Apple share price" OR "Apple revenue" '
    'OR "Apple forecast" OR "Apple innovation")'
)

'''
Subreddits that will be searched and returned
Add more to broaden the search, remove some to narrow the search
'''
subreddits = [
    "wallstreetbets", "gme", "personalfinance", "stocks", "pennystocks",
    "stockmarket", "investing", "options", "robinhoodpennystocks", "robinhood",
    "forex", "financialindependence", "finance", "securityanalysis",
    "daytrading", "swingtrading", "valueinvesting", "growthstocks",
    "apple", "mac", "iphone", "economy", "technology", "business", 
    "futurology", "webull", "tdameritrade", "fidelity", "thinkorswim",
    "bogleheads", "algotrading", "quantfinance"
]

# ---- Sentiment Classification (FinBERT) ----
'''
Function that calls finbert (trained model) to evaulate the sentiment of each post
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

DetectorFactory.seed = 0  # for consistent results

# ---- Language Detection ----
'''
Function to detect the language of the post
Utilizes the langdetect library
'''
def detect_language(text):
    try:
        lang_code = detect(text)

        # Get full language name instead of shorthand
        lang = Language.get(lang_code).display_name()
        return lang
    except LangDetectException:
        return "Unknown"

# ---- Main Scraper Function ----
'''
Main data gathering function, and calls the reddit API through praw
Searches and sorts the data by date (year)
Appends in the desired format for resulting CSV file
'''
def search_subreddit(subreddit, year, results):
    print(f"Searching r/{subreddit} for posts about Apple-related topics")
    try:
        submissions = reddit.subreddit(subreddit).search(
            query, sort="new", syntax="lucene", limit=1000
        )

        # Loop through finding data
        for submission in submissions:
            post_date = datetime.fromtimestamp(submission.created_utc)

            # If result is in the year we need
            if post_date.year == year:
                text = submission.title + " " + (submission.selftext or "")
                sentiment_score, sentiment_label = get_sentiment(text)
                language = detect_language(text)

                # Append the results for saving
                results.append({
                    "Date": post_date.strftime("%Y-%m-%d"),
                    "Title": submission.title,
                    "Text": text,                   # includes title for recomputation at next step
                    "Score": sentiment_score,
                    "Sentiment": sentiment_label,   
                    "Language": language
                })
        time.sleep(1)

    except Exception as e:
        print(f"Error with r/{subreddit}: {e}")

# ---- Per-Year Scraping ----
''' 
Function that calls the main searching one, with a results list
'''
def scrape_year(year):
    results = []
    for subreddit in subreddits:
        search_subreddit(subreddit, year, results)
    return results

# ---- Save to CSV ----
'''
Function to save the data into a CSV
'''
def save_to_csv(posts, year):
    if not posts:
        print(f"No posts found for {year}")
        return
    df = pd.DataFrame(posts)
    filename = f"reddit_{year}_{year}-01-01.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} posts to {filename}")

# ---- Run for Multiple Years ----
if __name__ == "__main__":
    for year in range(2020, 2025):
        print(f"\nScraping Year: {year}")
        posts = scrape_year(year)
        save_to_csv(posts, year)

# ---- Run for a Single Year ----
# if __name__ == "__main__":
#     year = 2022  # Change to any year (2020–2024)
#     print(f"\nScraping Year: {year}")
#     posts = scrape_year(year)
#     save_to_csv(posts, year)