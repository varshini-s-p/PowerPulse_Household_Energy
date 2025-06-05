import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from src.preprocessing import load_data, preprocess_data
import matplotlib
matplotlib.use('Agg')


# Paths
DATA_PATH = 'data/household_power_consumption.txt'
MODEL_DIR = 'outputs/models'
METRIC_DIR = 'outputs/metrics'
PLOTS_DIR = 'outputs/plots'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(" Loading and preprocessing data...")
df = load_data(DATA_PATH)
df = preprocess_data(df)

# Define features and target
y = df['Global_active_power']
X = df.drop(columns=['Global_active_power'])

# Use only numeric features
X = X.select_dtypes(include=[np.number])

# Split dataset
print(" Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
    "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=300, random_state=42)
}

# Train and evaluate
print("\n Starting model training...\n" + "-"*50)

for name, model in models.items():
    print(f" Training: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f" Evaluation for {name}:")
    print(f"    - RMSE: {rmse:.4f}")
    print(f"    - MAE : {mae:.4f}")
    print(f"    - R²  : {r2:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, model_path, compress=3)
    print(f" Model saved to: {model_path}")

    # Save metrics
    metrics_path = os.path.join(METRIC_DIR, f"{name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {name}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")
    print(f" Metrics saved to: {metrics_path}")

    # Plot predictions vs actual
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name} - Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plot_path = os.path.join(PLOTS_DIR, f"{name}_pred_vs_actual.png")
    plt.savefig(plot_path)
    plt.close()
    print(f" Prediction plot saved to: {plot_path}")

    # Feature importance (if applicable)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        importance_plot_path = os.path.join(PLOTS_DIR, f"{name}_feature_importance.png")
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f"{name} - Feature Importance")
        plt.tight_layout()
        plt.savefig(importance_plot_path)
        plt.close()
        print(f" Feature importance plot saved to: {importance_plot_path}")

    print("-" * 50)

print(" All models trained, evaluated, and saved successfully!")
