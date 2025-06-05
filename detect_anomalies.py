import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Using non-GUI backend for plotting
import matplotlib.pyplot as plt


# Load the predictions
df = pd.read_csv('outputs/predictions/all_model_predictions.csv')

# Use Global_active_power as the actual value
y_test = df['Global_active_power']
rf_predictions = df['RF_Predicted']

# Calculate residuals (absolute errors)
residuals = np.abs(y_test - rf_predictions)

# Define anomaly threshold (e.g., 95th percentile)
threshold = np.percentile(residuals, 95)
print(f"Anomaly threshold (95th percentile): {threshold}")

# Detect anomalies
df['Residuals'] = residuals
df['Anomaly'] = df['Residuals'] > threshold

# Save anomalies to a separate CSV
df[df['Anomaly']].to_csv('outputs/anomalies_detected.csv', index=False)
print("Anomalies saved to outputs/anomalies_detected.csv")

# Optional: Plot
plt.figure(figsize=(14, 6))
plt.plot(df['Datetime'], y_test, label='Actual', alpha=0.7)
plt.plot(df['Datetime'], rf_predictions, label='RF Predicted', alpha=0.7)
plt.scatter(df[df['Anomaly']]['Datetime'], df[df['Anomaly']]['Global_active_power'],
            color='red', label='Anomalies', marker='x')
plt.title("Anomaly Detection in Energy Consumption")
plt.xlabel("Datetime")
plt.ylabel("Global Active Power")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/anomaly_plot.png')
plt.show()
