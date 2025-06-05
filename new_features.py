import os
import pandas as pd
import numpy as np

# Set up dynamic paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, "outputs", "cleaned_power_data.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "processed_data.csv")

def engineer_features(df):
    # Fill missing values
    df = df.ffill()

    # Add new time-based features
    df["hour"] = df["Datetime"].dt.hour
    df["day"] = df["Datetime"].dt.day
    df["month"] = df["Datetime"].dt.month
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Rolling average features (example: 3-hour window)
    df["rolling_avg_3hr"] = df["Global_active_power"].rolling(window=3).mean()

    # Drop any NA rows created by rolling mean
    df = df.dropna()

    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, parse_dates=["Datetime"])
    df = engineer_features(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Processed data saved to:", OUTPUT_PATH)
