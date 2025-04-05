# pip install gdeltdoc
import  gdeltdoc as gd
import pandas as pd
from datetime import date, timedelta
from paramters import pipeline_sentiment,with_timeout

information = pd.DataFrame(columns=["Date","Title","Score","Sentiment","language"])
# setup
theme = ["ECON_FINANCE", "ECON_STOCKMARKET", "ECON_INFLATION", "BUSINESS", "TECHNOLOGY"]
start_date = date(2020, 1, 1)
end_date = date(2020, 1, 20)

current_start = start_date
current_end = start_date + timedelta(days=1)
while current_start <= end_date:
    print("-------------------" + str(current_start) + "----------------------------------------")
    count = 0
    filters = gd.Filters(
    keyword=["Apple stock","iphone sales",'apple sales',"inflation","interest rates", "CPI","Federal Reserve"],
    theme = theme,
    start_date=current_start.strftime("%Y-%m-%d"),
    end_date=current_end.strftime("%Y-%m-%d"),
    country = "US",
    num_records=31)
    initial = gd.GdeltDoc()
    try:
        articles = initial.article_search(filters)
        if not articles.empty:
            
            for index, row in articles.iterrows():
                sentiment, score = with_timeout(pipeline_sentiment,(row["url"]))

                if score is not None and count < 3:
                    information.loc[len(information)] = [str(current_start),row['title'],score,sentiment,row["language"]]
                    count += 1
                    
                if (count == 3):
                    break
            
           
            
      


        else:
            print("No articles found for the given filter criteria.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    print("--------------------------------------------------------------------------------")


    
    while(count < 3):
        information.loc[len(information)] = [str(current_start),None,0,"Neutral",None]
    current_start += timedelta(days=1)
    current_end += timedelta(days=1)


information.to_csv('financial.csv', index=False)