import pandas as pd
import joblib
import os

# === CONFIGURATION ===
input_csv_path = 'data/processed_data.csv'
output_dir = 'outputs/models'
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(input_csv_path)

# === DEBUG: Show columns ===
print(" Available columns:", df.columns.tolist())

# === CHECK IF 'Global_active_power' EXISTS ===
if 'Global_active_power' in df.columns:
    X = df.drop('Global_active_power', axis=1)
else:
    X = df

# === SAVE FEATURE COLUMNS ===
feature_columns = list(X.columns)
joblib.dump(feature_columns, os.path.join(output_dir, 'feature_columns.pkl'))

print("Saved updated feature columns:")
print(feature_columns)
