import pickle
import pandas as pd

def load_cleaned_data(filepath):
    # Load cleaned dataset (already preprocessed and saved after training)
    df = pd.read_csv(filepath)

    # Drop target column to get features for prediction
    X = df.drop(columns=['Global_active_power'])

    return X

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, X):
    predictions = model.predict(X)
    return predictions

if __name__ == "__main__":
    # Load the cleaned data
    data_path = "outputs/cleaned_power_data.csv"
    model_path = "outputs/models/GradientBoosting.pkl"

    print("Loading data...")
    X = load_cleaned_data(data_path)

    print("Loading model...")
    model = load_model(model_path)

    print("Making predictions on last 5 records...")
    predictions = predict(model, X.tail(5))

    print("Predicted Global Active Power values:")
    for i, pred in enumerate(predictions, start=1):
        print(f"{i}: {pred:.4f}")
