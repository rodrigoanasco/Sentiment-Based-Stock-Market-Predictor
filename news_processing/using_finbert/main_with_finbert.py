# main.py
# pip install gdeltdoc
import gdeltdoc as gd
import pandas as pd
from datetime import date, timedelta
import concurrent.futures
import time
from tqdm import tqdm

# Import the functions from our parameters.py
from using_finbert.parameters_with_finbert import pipeline_sentiment, parallel_requests

def process_date(current_date):
    """Fetch articles for a single day, run sentiment analysis, 
       and return a DataFrame with up to 3 results for that day."""
    theme = ["ECON_FINANCE", "ECON_STOCKMARKET", "ECON_INFLATION", "BUSINESS", "TECHNOLOGY"]
    results = pd.DataFrame(columns=["Date", "Title", "Score", "Sentiment", "language"])
    
    current_end = current_date + timedelta(days=1)
    print(f"------------------- {current_date} ----------------------------------------")
    count = 0
    
    # Customize these keywords and filters as needed
    filters = gd.Filters(
        keyword=["Apple stock", "iphone sales", "apple sales", "inflation", 
                 "interest rates", "CPI", "Federal Reserve"],
        theme=theme,
        start_date=current_date.strftime("%Y-%m-%d"),
        end_date=current_end.strftime("%Y-%m-%d"),
        country="US",
        num_records=15
    )
    
    gdelt_client = gd.GdeltDoc()
    try:
        articles = gdelt_client.article_search(filters)
        if not articles.empty:
            urls = articles['url'].tolist()
            url_results = parallel_requests(urls, max_workers=5)
            
            for index, row in articles.iterrows():
                sentiment, score = url_results.get(row["url"], (None, None))
                
                # If we got a real sentiment polarity and haven't reached 3 articles yet
                if score is not None and count < 5:
                    results.loc[len(results)] = [
                        str(current_date),
                        row['title'],
                        score,
                        sentiment,
                        row["language"]
                    ]
                    count += 1
                
                if count == 5:
                    break
        else:
            print("No articles found for the given filter criteria.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # If fewer than 3 results, fill in with neutral placeholders
    while count < 5:
        results.loc[len(results)] = [str(current_date), None, 0, "Neutral", None]
        count += 1
    
    print("--------------------------------------------------------------------------------")
    return results

def main(start, end):
    """Run the entire pipeline from start date to end date, 
       saving results to a CSV."""
    start_time = time.time()
    
    # Create a list of all dates in the range
    date_list = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    
    # Initialize master DataFrame
    all_results = pd.DataFrame(columns=["Date", "Title", "Score", "Sentiment", "language"])
    
    # We process dates in chunks to avoid overwhelming resources
    chunk_size = 5
    for i in range(0, len(date_list), chunk_size):
        chunk_dates = date_list[i:i+chunk_size]
        
        # Process each chunk of dates in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(chunk_dates))) as executor:
            future_to_date = {executor.submit(process_date, d): d for d in chunk_dates}
            
            # Use tqdm to track progress of each chunk
            for future in tqdm(concurrent.futures.as_completed(future_to_date),
                               total=len(chunk_dates),
                               desc=f"Processing days {i+1}-{min(i+chunk_size, len(date_list))}"):
                date_results = future.result()
                all_results = pd.concat([all_results, date_results], ignore_index=True)
    
    # Save final results
    csv_filename = f'financial_{start}.csv'
    all_results.to_csv(csv_filename, index=False)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Processed {len(date_list)} days with {len(all_results)} total entries")
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    # Example: 2022 full year
    main(date(2020, 1, 1), date(2020, 12, 31))
    main(date(2021, 1, 1), date(2021, 12, 31))
    main(date(2023, 1, 1), date(2023, 12, 31))
    main(date(2024, 1, 1), date(2024, 12, 31))

