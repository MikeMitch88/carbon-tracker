# model_training.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    return pd.read_csv("country_emissions.csv")

def prepare_data(df):
    # Create features and target
    X = df[['Country', 'Year']]
    y = df['Per_Capita_CO2_kg']
    
    return X, y

def train_model(X, y):
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('country', OneHotEncoder(handle_unknown='ignore'), ['Country'])
        ],
        remainder='passthrough'
    )
    
    # Model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train model
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R²: {train_score:.4f}")
    print(f"Testing R²: {test_score:.4f}")

def main():
    # Load data
    df = load_data()
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Evaluate model
    evaluate_model(model, X, y)
    
    # Save model
    joblib.dump(model, "country_emissions_model.pkl")
    print("Model saved as country_emissions_model.pkl")

if __name__ == "__main__":
    main()