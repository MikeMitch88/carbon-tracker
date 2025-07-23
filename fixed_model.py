# fix_model.py
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load your data
emissions_df = pd.read_csv("country_emissions.csv")

# Prepare features
X = emissions_df[['Country', 'Year']].copy()
y = emissions_df['Per_Capita_CO2_kg'].copy()

# Create preprocessor with OBJECTS (not strings)
preprocessor = ColumnTransformer(
    transformers=[
        ('country', OneHotEncoder(handle_unknown='ignore'), ['Country']),
        ('year', StandardScaler(), ['Year'])
    ]
)

# Create pipeline with OBJECTS
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit and save
model.fit(X, y)
joblib.dump(model, "country_emissions_model_fixed.pkl")
print("Fixed model saved!")