import pandas as pd

import pandas as pd

def load_data(file_path):
    # Step 1: Read raw data
    df = pd.read_csv(
        file_path,
        sep=';',
        na_values='?',
        low_memory=False
    )

    # Step 2: Combine Date and Time into Datetime AFTER reading
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.dropna(subset=['Datetime'], inplace=True)
    df.set_index('Datetime', inplace=True)

    return df


def preprocess_data(df):
    df = df.astype(float, errors='ignore')
    df.dropna(inplace=True)

    # Time features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday

    return df
