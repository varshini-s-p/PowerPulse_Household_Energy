import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load predicted data
df = pd.read_csv("outputs/predictions/all_model_predictions.csv")
df.columns = df.columns.str.strip().str.lower()

# Drop non-model columns
df = df.drop(columns=['datetime'])
df = df.drop(columns=['global_active_power'], errors='ignore')  # Remove it if exists

# Step 1: Summary Stats
summary_stats = df.describe().T
summary_stats["std_dev"] = summary_stats["std"]
summary_stats["range"] = summary_stats["max"] - summary_stats["min"]

# Step 2: Distribution Histograms
plt.figure(figsize=(15, 5))
for i, col in enumerate(df.columns):
    plt.subplot(1, len(df.columns), i + 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'{col} distribution')
plt.tight_layout()
plt.savefig("outputs/model_distribution.png")

# Step 3: Correlation Analysis
corr_matrix = df.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Model Prediction Correlation")
plt.savefig("outputs/model_correlation.png")

# Step 4: Scoring Models
std_score = 1 / summary_stats["std_dev"]
range_score = 1 / summary_stats["range"]
mean_corr = corr_matrix.mean()
corr_score = mean_corr

summary_stats["score"] = (
    0.4 * std_score +
    0.3 * range_score +
    0.3 * corr_score
)

# Step 5: Ranking
summary_stats["rank"] = summary_stats["score"].rank(ascending=False)
summary_stats = summary_stats.sort_values(by="rank")

# Output ranked table
summary_stats[["std_dev", "range", "score", "rank"]].to_csv("outputs/model_ranking.csv", index=True)

# Print ranking summary
print(" Model Ranking Based on Prediction Quality (No Actuals):\n")
print(summary_stats[["std_dev", "range", "score", "rank"]])


