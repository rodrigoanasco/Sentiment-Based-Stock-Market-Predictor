# pip install gdeltdoc
import gdeltdoc as gd
import pandas as pd
import requests
from datetime import date, timedelta
from bs4 import BeautifulSoup
from textblob import TextBlob
from urllib.parse import urlparse
import signal
from functools import lru_cache
import concurrent.futures
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Convert to set for faster lookups
CREDIBLE_SITES = set([
    "bloomberg.com",
    "reuters.com",
    "ft.com",
    "forbes.com",
    "fortune.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "wsj.com",  # The Wall Street Journal
    "economist.com",
    "cnbc.com",
    "cnn.com",  # CNN Business is a section
    "nytimes.com",  # The New York Times - Business section
    "washingtonpost.com",
    "businessinsider.com",
    "investopedia.com",
    "seekingalpha.com",
    "fool.com",  # The Motley Fool
    "npr.org",
    "bbc.com",
    "theguardian.com",
    "usatoday.com",
    "latimes.com",  # Los Angeles Times
    "chicagotribune.com",
    "theatlantic.com",
    "usnews.com",
    "msn.com",
    "thestreet.com",
    "morningstar.com",
    "nasdaq.com",
    "businesstech.co.za"
])

# Create a persistent session with retry logic
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=50)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Set default headers to mimic a browser
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
})

@lru_cache(maxsize=1000)
def is_credible(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace('www.', '')
    return domain in CREDIBLE_SITES

@lru_cache(maxsize=100)
def is_website_current(url):
    try:
        # Use HEAD request with shorter timeout
        response = session.head(url, timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False

@lru_cache(maxsize=500)
def get_archived_url(original_url):
    try:
        api = "http://archive.org/wayback/available"
        response = session.get(api, params={"url": original_url}, timeout=5)
        data = response.json()
        return data.get("archived_snapshots", {}).get("closest", {}).get("url")
    except:
        return None

def get_sentiment_from_url(url):
    if not url:
        return "URL not available", None
        
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        
        # Use lxml parser for faster parsing
        soup = BeautifulSoup(response.text, "lxml")
        
        # Optimize text extraction with list comprehension
        paragraphs = soup.find_all('p')
        text = ' '.join(para.get_text() for para in paragraphs)
        
        if len(text) < 50:  # not enough data
            return "Insufficient content", None
            
        # Perform sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "Positive", polarity
        elif polarity < -0.1:
            return "Negative", polarity
        else:
            return "Neutral", polarity
    except Exception as e:
        return f"Error: {str(e)[:100]}", None

def pipeline_sentiment(url):
    if not is_credible(url):
        return "Not credible", None
        
    if is_website_current(url):
        return get_sentiment_from_url(url)
    else:
        archived_url = get_archived_url(url)
        return get_sentiment_from_url(archived_url)

def parallel_requests(urls, max_workers=10):
    # Filter out None or empty URLs first
    valid_urls = [url for url in urls if url]
    
    # Return empty dict if no valid URLs
    if not valid_urls:
        return {}
    
    # Process URLs in parallel
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(pipeline_sentiment, url): url for url in valid_urls}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                sentiment, polarity = future.result()
                results[url] = (sentiment, polarity)
            except Exception as e:
                results[url] = (f"Error: {str(e)[:100]}", None)
    
    return results

# Additional function to process a batch of articles at once
def process_articles_batch(articles_df, max_results=3):
    if articles_df.empty:
        return []
    
    # Extract all URLs at once
    urls = articles_df['url'].tolist()
    
    # Process all URLs in parallel
    url_results = parallel_requests(urls)
    
    # Collect results
    results = []
    count = 0
    
    for index, row in articles_df.iterrows():
        sentiment, score = url_results.get(row["url"], (None, None))
        
        if score is not None and count < max_results:
            results.append({
                "Date": row.get("date", None),
                "Title": row['title'],
                "Score": score,
                "Sentiment": sentiment,
                "language": row.get("language", None)
            })
            count += 1
            
        if count >= max_results:
            break
            
    return results
