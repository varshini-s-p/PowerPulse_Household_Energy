# main.py
import matplotlib
matplotlib.use('Agg')  


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
DATA_PATH = os.path.join("data", "household_power_consumption.txt")

# Load data
def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(
        path, 
        sep=';', 
        low_memory=False, 
        na_values='?', 
        parse_dates=[[0, 1]],  # Combine 'Date' and 'Time' columns
        infer_datetime_format=True
    )
    df.rename(columns={'Date_Time': 'Datetime'}, inplace=True)
    return df

# Initial checks
def data_overview(df):
    print("\nDataset Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSummary Statistics:\n", df.describe())
    print("\nUnique Value Counts:\n", df.nunique())

# Run EDA
def exploratory_analysis(df):
    # Plot Global Active Power over time (sample to speed up rendering)
    df_sample = df[['Datetime', 'Global_active_power']].dropna().sample(10000)
    plt.figure(figsize=(10, 4))
    plt.plot(df_sample['Datetime'], df_sample['Global_active_power'], linestyle='-', marker='', color='blue')
    plt.title("Sample of Global Active Power over Time")
    plt.xlabel("Datetime")
    plt.ylabel("Global Active Power (kilowatts)")
    plt.tight_layout()
    plt.savefig("outputs/global_active_power_over_time.png")
    plt.close()

    # Correlation Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("outputs/correlation_matrix.png")
    plt.close()

# Main execution
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    data_overview(df)
    exploratory_analysis(df)
    print("Data loading and EDA complete.")

from src.preprocessing import load_data, preprocess_data

# Step 1: Load
df = load_data("data/household_power_consumption.txt")

# Step 2: Preprocess
df_cleaned = preprocess_data(df)

# Save a sample of cleaned data
df_cleaned.to_csv("outputs/cleaned_power_data.csv")
