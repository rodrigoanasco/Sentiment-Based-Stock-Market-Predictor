# pip install gdeltdoc
import  gdeltdoc as gd
import pandas as pd
import requests
from datetime import date, timedelta
from bs4 import BeautifulSoup
from textblob import TextBlob
from urllib.parse import urlparse
import signal

credible_sites = [
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
]

def is_credible(url):
    # List of common reputable domains
    credible_domains = credible_sites

    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace('www.', '')

    return domain in credible_domains

# check if current link is availble 
def is_website_current(url):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            return True
        else:
            return False  
    except requests.RequestException as e:
        # If there was an error (like connection issues), return False
        print(f"Error occurred while trying to access the website: {e}")
        return False


# try to get archived data
def get_archived_url(original_url):
    api = "http://archive.org/wayback/available"
    response = requests.get(api, params={"url": original_url})
    data = response.json()
    try:
        snapshot = data["archived_snapshots"]["closest"]["url"]
        return snapshot
    except KeyError:
        return None

def get_sentiment_from_url(url):
  
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all visible text
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])

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
        return f"Error: {e}", None
    

def pipeline_sentiment(url):
    if not is_credible(url):
        return None,None
    if is_website_current(url):
        print("used current url")
        return get_sentiment_from_url(url)
    else:
       archived_url = get_archived_url(url)
       return get_sentiment_from_url(archived_url)



def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

# Set up signal handler
signal.signal(signal.SIGALRM, timeout_handler)

def with_timeout(func, *args, timeout=30, **kwargs):
    # Set the alarm signal before calling the function
    signal.alarm(timeout)
    
    try:
        return func(*args, **kwargs)
    except TimeoutError as e:
        print(f"Timeout occurred: {e}")
        return None,None
    finally:
        # Disable the alarm once the function is done
        signal.alarm(0)