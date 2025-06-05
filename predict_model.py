import pandas as pd
import joblib

# Load data
new_data = pd.read_csv('data/processed_data.csv')

# Common feature columns
feature_columns = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                   'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                   'hour', 'day', 'month', 'weekday']
X_new = new_data[feature_columns]

# Predict with Random Forest
rf_model = joblib.load('outputs/models/RandomForest.pkl')
rf_predictions = rf_model.predict(X_new)
print("Random Forest Predictions:", rf_predictions)

# Predict with XGBoost
gb_model = joblib.load('outputs/models/GradientBoosting.pkl')
gb_predictions = gb_model.predict(X_new)
print("Gradient Boosting Predictions:", gb_predictions)

# Predict with Linear Regression
lr_model = joblib.load('outputs/models/LinearRegression.pkl')
lr_predictions = lr_model.predict(X_new)
print("Linear Regression Predictions:", lr_predictions)

# Predict with Linear Regression
nn_model = joblib.load('outputs/models/NeuralNetwork.pkl')
nn_predictions = nn_model.predict(X_new)
print("Neural Network Predictions:", nn_predictions)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming you used rf_predictions for the prediction result
y_true = new_data['Global_active_power']
y_pred = rf_predictions 

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(" Evaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

import os

# Create output directory if it doesn't exist
os.makedirs('outputs/predictions', exist_ok=True)

# Create a copy of the input data
output_df = new_data.copy()

# Add predictions to the dataframe
output_df['RF_Predicted'] = rf_predictions
output_df['LR_Predicted'] = lr_predictions
output_df['GB_Predicted'] = gb_predictions 
output_df['NN_Predicted'] = nn_predictions # Add more as needed

# Save to CSV
output_df[['Datetime', 'Global_active_power', 'RF_Predicted', 'LR_Predicted', 'NN_Predicted']].to_csv(
    'outputs/predictions/all_model_predictions.csv', index=False
)

print(" All model predictions exported to: outputs/predictions/all_model_predictions.csv")


