import pandas as pd

# Replace with your actual CSV file and column name
csv_file = 'FRED/TJ/consumer_sentiment_daily.csv'
date_column = 'date'  # Change this to match the column name in your CSV

# Load the CSV and parse the date column
df = pd.read_csv(csv_file, parse_dates=[date_column])

# Sort dates
df = df.sort_values(by=date_column)

# Create a full date range from min to max date
full_range = pd.date_range(start=df[date_column].min(), end=df[date_column].max())

# Find missing dates
missing_dates = full_range.difference(df[date_column])

# Output
print("Missing Dates:")
for date in missing_dates:
    print(date.strftime('%Y-%m-%d'))
