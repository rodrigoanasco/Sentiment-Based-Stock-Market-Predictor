import praw
import pandas as pd
from datetime import datetime
import time
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from langcodes import Language
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Load FinBERT model and tokenizer once
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)

# ---- Reddit API Setup ----
reddit = praw.Reddit(
    client_id="Eg87PPxOl37V9_jcvci25g",
    client_secret="lhJeadQ-V6KMZzxvUe2exy2p1S9_tw",
    user_agent="script:apple_stock_scraper:v1.0 (by u/YOUR_USERNAME)"
)

try:
    print(f"Authenticated as: {reddit.user.me()}")
    print(f"Accessing r/personalfinance...")
    posts = reddit.subreddit("personalfinance").hot(limit=5)
    for post in posts:
        print(f"Title: {post.title}")
except Exception as e:
    print("❌ Authentication failed or issue with access:", e)

# ---- Config ----
query = (
    '("apple stock" OR AAPL OR "Apple Inc" OR "Tim Cook" OR iPhone OR MacBook '
    'OR iOS OR "tech stocks" OR "technology sector" OR "Silicon Valley" '
    'OR FAANG OR "Apple earnings" OR "Apple share price" OR "Apple revenue" '
    'OR "Apple forecast" OR "Apple innovation")'
)

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
def get_sentiment(text):
    try:
        result = finbert_pipeline(text[:512])[0]  
        label = result["label"]
        score = result["score"]
        if label == "positive":
            return score, "Positive"
        elif label == "negative":
            return -score, "Negative"
        else:
            return 0.0, "Neutral"
    except Exception as e:
        print(f"⚠️ Sentiment analysis failed: {e}")
        return 0.0, "Neutral"

DetectorFactory.seed = 0  # for consistent results

# ---- Language Detection ----
def detect_language(text):
    try:
        lang_code = detect(text)
        lang = Language.get(lang_code).display_name()
        return lang
    except LangDetectException:
        return "Unknown"

# ---- Main Scraper Function ----
def search_subreddit(subreddit, year, results):
    print(f"🔎 Searching r/{subreddit} for posts about Apple-related topics")
    try:
        submissions = reddit.subreddit(subreddit).search(
            query, sort="new", syntax="lucene", limit=1000
        )
        for submission in submissions:
            post_date = datetime.fromtimestamp(submission.created_utc)
            if post_date.year == year:
                text = submission.title + " " + (submission.selftext or "")
                sentiment_score, sentiment_label = get_sentiment(text)
                language = detect_language(text)
                results.append({
                    "Date": post_date.strftime("%Y-%m-%d"),
                    "Title": submission.title,
                    "Score": sentiment_score,
                    "Sentiment": sentiment_label,
                    "Language": language
                })
        time.sleep(1)

    except Exception as e:
        print(f"❌ Error with r/{subreddit}: {e}")

# ---- Per-Year Scraping ----
def scrape_year(year):
    results = []
    for subreddit in subreddits:
        search_subreddit(subreddit, year, results)
    return results

# ---- Save to CSV ----
def save_to_csv(posts, year):
    if not posts:
        print(f"⚠️ No posts found for {year}")
        return
    df = pd.DataFrame(posts)
    filename = f"reddit_{year}_{year}-01-01.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} posts to {filename}")

# ---- Run for Multiple Years ----
if __name__ == "__main__":
    for year in range(2020, 2025):
        print(f"\n📅 Scraping Year: {year}")
        posts = scrape_year(year)
        save_to_csv(posts, year)

# # ---- Run for a Single Year ----
# if __name__ == "__main__":
#     year = 2022  # Change to any year (2020–2024)
#     print(f"\n📅 Scraping Year: {year}")
#     posts = scrape_year(year)
#     save_to_csv(posts, year)