import pandas as pd
import argparse

'''
This file contains the code used to combine CSV files and replace NaN values for model training.
For more information, check out the relevent model training files
'''

# ---- Parser Function ----
'''
Function that receives user input and parses it
Use flag --output for custom name
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Combine and clean Reddit sentiment CSV files.')
    parser.add_argument('input_csvs', nargs='+', help='Paths to input CSV files')
    parser.add_argument('--output', default='combined_reddit.csv', help='Path to output CSV file')
    return parser.parse_args()

# ---- Combiner Function ----
'''
Function that does the combining and value replacement
'''
def combine_and_clean(file_paths):

    df = pd.concat([pd.read_csv(fp) for fp in file_paths], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Replace NaN in Score with 0
    score_nan_mask = df['Score'].isna()
    df.loc[score_nan_mask, 'Score'] = 0

    # Replace corresponding Sentiment NaN with 'Neutral'
    df.loc[score_nan_mask & df['Sentiment'].isna(), 'Sentiment'] = 'Neutral'

    return df

# ---- Main Function ----
'''
Main that calls the other functions
'''
def main():
    args = parse_args()
    combined_df = combine_and_clean(args.input_csvs)
    combined_df.to_csv(args.output, index=False)
    print(f"\nCombined and cleaned CSV saved to: {args.output}")

if __name__ == '__main__':
    main()
