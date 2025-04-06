# pip install gdeltdoc
import gdeltdoc as gd
import pandas as pd
from datetime import date, timedelta
from paramters import pipeline_sentiment, parallel_requests  # Fixed typo in "parameters"
import concurrent.futures
import time
from tqdm import tqdm

# Function to process a single date
def process_date(current_date):
    theme = ["ECON_FINANCE", "ECON_STOCKMARKET", "ECON_INFLATION", "BUSINESS", "TECHNOLOGY"]
    results = pd.DataFrame(columns=["Date", "Title", "Score", "Sentiment", "language"])
    
    current_end = current_date + timedelta(days=1)
    print(f"------------------- {current_date} ----------------------------------------")
    count = 0
    
    filters = gd.Filters(
        keyword=["Apple stock", "iphone sales", 'apple sales', "inflation", "interest rates", "CPI", "Federal Reserve"],
        theme=theme,
        start_date=current_date.strftime("%Y-%m-%d"),
        end_date=current_end.strftime("%Y-%m-%d"),
        country="US",
        num_records=15)
    
    initial = gd.GdeltDoc()
    try:
        articles = initial.article_search(filters)
        if not articles.empty:
            # Get all URLs at once
            urls = articles['url'].tolist()
            # Process URLs in parallel
            url_results = parallel_requests(urls, max_workers=5)
            
            for index, row in articles.iterrows():
                sentiment, score = url_results.get(row["url"], (None, None))
                if score is not None and count < 3:
                    results.loc[len(results)] = [
                        str(current_date),
                        row['title'],
                        score,
                        sentiment,
                        row["language"]
                    ]
                    count += 1
                
                if count == 3:
                    break
        else:
            print("No articles found for the given filter criteria.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # Fill in with neutral entries if needed
    while count < 3:
        results.loc[len(results)] = [str(current_date), None, 0, "Neutral", None]
        count += 1
    
    print("--------------------------------------------------------------------------------")
    return results

def main():
    start_time = time.time()
    
    # Date range to process
    start_date = date(2020, 1, 1)
    end_date = date(2020, 12, 31)
    
    # Create list of all dates to process
    date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    # Initialize master DataFrame
    all_results = pd.DataFrame(columns=["Date", "Title", "Score", "Sentiment", "language"])
    
    # Process dates in chunks to avoid overwhelming resources
    chunk_size = 5
    for i in range(0, len(date_list), chunk_size):
        chunk_dates = date_list[i:i+chunk_size]
        
        # Process chunk in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(chunk_dates))) as executor:
            # Map dates to futures
            future_to_date = {executor.submit(process_date, d): d for d in chunk_dates}
            
            # Process results as they complete with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_date),
                            total=len(chunk_dates),
                            desc=f"Processing days {i+1}-{min(i+chunk_size, len(date_list))}"):
                date_results = future.result()
                all_results = pd.concat([all_results, date_results], ignore_index=True)
    
    # Save final results
    all_results.to_csv('financial.csv', index=False)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Processed {len(date_list)} days with {len(all_results)} total entries")

if __name__ == "__main__":
    main()
