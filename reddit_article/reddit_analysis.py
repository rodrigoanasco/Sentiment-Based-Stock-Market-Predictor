import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

'''
This file contains the code used to analyze and understand the data that was cleaned up previously.
This data is turned into numerical values and/or diagrams that we can extract information from.
'''

# ---- Parser Function ----
'''
Function that receives user input and parses it
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Reddit sentiment vs Apple stock.')
    parser.add_argument('input_csvs', nargs='+', help='One or more cleaned CSV file paths')
    return parser.parse_args()

# ---- Load and Combine Data ----
'''
Function that loads the user inputted CSV file(s) 
Results in a dataframe with all the files combined
'''
def load_data(file_paths):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df['Percent_Diff'] = pd.to_numeric(df['Percent_Diff'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        dfs.append(df)
    combined_df = pd.concat(dfs).dropna(subset=['Date'])
    combined_df = combined_df.sort_values(by='Date').reset_index(drop=True)
    return combined_df

# ---- 1. Correlation Analysis ----
'''
Same day analysis...
Function to check correlation using Pearson Correlation on same day
'''
def correlation_analysis(df):
    print("\n--- Correlation Analysis ---")
    print(f"Pearson Correlation between Score and Percent_Diff: {df['Score'].corr(df['Percent_Diff']):.10f}")
    print(f"Pearson Correlation between Score and Volume: {df['Score'].corr(df['Volume']):.10f}")

# ---- 2. Prediction Analysis ----
'''
Maybe sentiment can be used to predict future changes in stock?
Function to check correlation using Pearson Correlation on preceding data
'''
def prediction_analysis(df):
    df['Score_1'] = df['Score'].shift(1)
    df['Score_2'] = df['Score'].shift(2)
    print("\n--- Prediction Analysis ---")
    print(f"Pearson Correlation between Score shifted once and Percent_Diff: {df['Score_1'].corr(df['Percent_Diff']):.10f}")
    print(f"Pearson Correlation between Score shifted once and Volume: {df['Score_1'].corr(df['Volume']):.10f}")
    print(f"Pearson Correlation between Score shifted twice and Percent_Diff: {df['Score_2'].corr(df['Percent_Diff']):.10f}")
    print(f"Pearson Correlation between Score shifted twice and Volume: {df['Score_2'].corr(df['Volume']):.10f}")

# ---- 3. Visualizations ----
'''
Functions that generate (hopefully) useful diagrams
Visualize our data to find something
'''
def visualizations(df):

    # Line: Score vs Price
    plt.figure(figsize=(14, 6))
    ax = df.set_index('Date')[['Score', 'Close']].plot(secondary_y='Close', figsize=(14, 6), grid=True)
    ax.set_ylabel("Sentiment Score")
    ax.right_ax.set_ylabel("Apple Stock Price (Close)")
    plt.title("Sentiment Score vs Apple Stock Price (Close)")
    plt.tight_layout()
    plt.savefig("score_vs_stock_price.png")
    # plt.show()

    # Scatter: Score vs Volume
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Score', y='Volume', data=df)
    plt.title("Score vs Trading Volume")
    plt.tight_layout()
    plt.savefig("score_vs_volume.png")
    # plt.show()

    # Line: Score vs Volume
    plt.figure(figsize=(14, 6))
    ax = df.set_index('Date')[['Score', 'Volume']].plot(secondary_y='Volume', figsize=(14, 6), grid=True)
    ax.set_ylabel("Sentiment Score")
    ax.right_ax.set_ylabel("Trading Volume")
    plt.title("Sentiment Score vs Trading Volume")
    plt.tight_layout()
    plt.savefig("score_vs_volume_2.png")
    # plt.show()

    print("\nPlots saved as PNG files.")

# ---- Main ----
'''
Runs the functions above and generates output
'''
def main():
    args = parse_args()
    df = load_data(args.input_csvs)

    correlation_analysis(df)
    prediction_analysis(df)
    visualizations(df)

if __name__ == '__main__':
    main()
