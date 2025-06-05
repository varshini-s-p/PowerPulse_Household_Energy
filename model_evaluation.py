import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the processed dataset
df = pd.read_csv("data/processed_data.csv", parse_dates=["Datetime"])

# Define features (X) and target (y)
X = df.drop(columns=["Datetime", "Global_active_power"])  # Replace 'target_column' with your actual target name
y = df["Global_active_power"]                             # Replace with actual column name

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model = joblib.load("model_training.pkl")  # Replace with your actual model filename

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

