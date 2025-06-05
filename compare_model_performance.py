import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load actual and predicted data
df = pd.read_csv('outputs/predicted_results.csv')

# Extract true and predicted values
y_true = df['Global_active_power']
y_pred = df['Predicted_Global_active_power']

# Compute metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Display results
print("Evaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# Load predictions and actuals
df = pd.read_csv("outputs/predictions/all_model_predictions.csv")

# Define actual and predictions
y_true = df['Global_active_power']
models = {
    'Random Forest': df['RF_Predicted'],
    'Linear Regression': df['LR_Predicted'],
    'Neural Network': df['NN_Predicted']
}

# Calculate metrics for each model
results = []

for model_name, y_pred in models.items():
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    })

# Create comparison table
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("outputs/model_comparison_metrics.csv", index=False)
print(metrics_df)

