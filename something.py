import gdeltdoc as gd
import pandas as pd
import requests
from datetime import date, timedelta
from bs4 import BeautifulSoup
from textblob import TextBlob
from urllib.parse import urlparse
import concurrent.futures
import time
import random

# List of credible financial news sources
credible_sites = [
    "bloomberg.com", "reuters.com", "ft.com", "forbes.com", "fortune.com",
    "marketwatch.com", "finance.yahoo.com", "wsj.com", "economist.com",
    "cnbc.com", "cnn.com", "nytimes.com", "washingtonpost.com",
    "businessinsider.com", "investopedia.com", "seekingalpha.com",
    "fool.com", "npr.org", "bbc.com", "theguardian.com",
    "usatoday.com", "latimes.com", "chicagotribune.com", "theatlantic.com",
    "usnews.com", "msn.com", "thestreet.com", "morningstar.com",
    "nasdaq.com", "businesstech.co.za"
]

# Convert to set for faster lookups
credible_domains = set(credible_sites)

def is_credible(url):
    """Check if URL belongs to a credible domain"""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace('www.', '')
        return domain in credible_domains
    except:
        return False

def get_sentiment_from_url(url):
    """Extract text from URL and analyze sentiment"""
    if not url:
        return None, None
        
    try:
        # Rotating user agents to avoid detection
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        
        # Use a very short timeout to avoid hanging
        response = requests.get(url, headers=headers, timeout=3)
        
        if response.status_code != 200:
            return None, None
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Get text from both paragraphs and article tags for better coverage
        text_elements = soup.find_all(['p', 'article', 'div.article-body', 'div.story-body'])
        text = ' '.join([elem.get_text() for elem in text_elements])
        
        # Limit to first 2000 characters for speed
        text = text[:2000]
        
        if len(text) < 200:  # Skip if too little text was found
            return None, None
        
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
        return None, None

def process_batch(date_str, urls_titles, language_list):
    """Process a batch of articles for a given date"""
    results = []
    
    for url, title, language in zip(urls_titles[0], urls_titles[1], language_list):
        if not is_credible(url):
            continue
            
        # Skip archive.org completely - too slow
        sentiment, score = None, None
        
        try:
            sentiment, score = get_sentiment_from_url(url)
        except:
            pass
            
        if sentiment and score is not None:
            results.append({
                "Date": date_str,
                "Title": title,
                "Score": score,
                "Sentiment": sentiment,
                "language": language
            })
            
        # Stop after finding 3 valid articles
        if len(results) >= 3:
            break
            
    return results

def main():
    # Date range to process
    start_date = date(2020, 1, 1)
    end_date = date(2020, 1, 3)
    
    # Initialize GDELT client once
    gdelt = gd.GdeltDoc()
    
    # Pre-initialize results DataFrame
    information = pd.DataFrame(columns=["Date", "Title", "Score", "Sentiment", "language"])
    
    # Define themes and keywords
    theme = ["ECON_FINANCE", "ECON_STOCKMARKET", "ECON_INFLATION", "BUSINESS", "TECHNOLOGY"]
    keywords = ["Apple stock", "iphone sales", 'apple sales', "inflation", 
               "interest rates", "CPI", "Federal Reserve"]
    
    # Process date range in smaller chunks to avoid timeouts
    current_start = start_date
    
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=5), end_date)
        print(f"Processing date range: {current_start} to {current_end}")
        
        try:
            # Get a batch of articles for the date range
            filters = gd.Filters(
                keyword=keywords,
                theme=theme,
                start_date=current_start.strftime("%Y-%m-%d"),
                end_date=current_end.strftime("%Y-%m-%d"),
                country="US",
                num_records=250  # Get more articles to increase chances of finding valid ones
            )
            
            articles = gdelt.article_search(filters)
            
            if not articles.empty:
                # Create a map of date -> articles
                date_articles = {}
                
                for _, row in articles.iterrows():
                    article_date = row['seendate'].date()
                    date_str = str(article_date)
                    
                    if date_str not in date_articles:
                        date_articles[date_str] = [[], [], []]  # urls, titles, languages
                        
                    date_articles[date_str][0].append(row['url'])
                    date_articles[date_str][1].append(row['title'])
                    date_articles[date_str][2].append(row['language'])
                
                # Process each date's articles
                for date_str, urls_titles_langs in date_articles.items():
                    print(f"Processing {date_str}...")
                    
                    # Process articles with reduced timeout
                    results = process_batch(date_str, urls_titles_langs[:2], urls_titles_langs[2])
                    
                    # Add results to DataFrame
                    for result in results[:3]:  # Take max 3 articles
                        information.loc[len(information)] = [
                            result["Date"], result["Title"], result["Score"], 
                            result["Sentiment"], result["language"]
                        ]
                    
                    # Fill with neutral entries if needed
                    while len(results) < 3:
                        information.loc[len(information)] = [date_str, None, 0, "Neutral", None]
                        results.append({})  # Just to increment the counter
            
            # Save intermediate results to avoid losing progress
            information.to_csv('financial_progress.csv', index=False)
            
        except Exception as e:
            print(f"Error processing batch {current_start} to {current_end}: {e}")
        
        # Move to next chunk of dates
        current_start = current_end + timedelta(days=1)
        
        # Rate limiting to avoid API blocks
        time.sleep(2)
    
    # Final save
    information.to_csv('financial_optimized.csv', index=False)
    print("Completed processing!")

if __name__ == "__main__":
    main()