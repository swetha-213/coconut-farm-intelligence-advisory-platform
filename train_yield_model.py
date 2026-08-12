# train_yield_model.py - Coconut Yield Model Training (Fixed Path Version)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import os

# ==================== FIXED PATHS ====================

PROJECT_ROOT = r"D:\coconut_project"

# Data & Model paths
YIELD_CSV = os.path.join(PROJECT_ROOT, 'data', 'yield_data', 'Tamilnadu agriculture yield data.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

print("[DEBUG] PROJECT_ROOT:", PROJECT_ROOT)
print("[DEBUG] YIELD_CSV:", YIELD_CSV)
print("[DEBUG] MODEL_DIR:", MODEL_DIR)

# Folders create
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if CSV exists
if not os.path.exists(YIELD_CSV):
    print(f"[ERROR] CSV file not found: {YIELD_CSV}")
    print("Please check if file name is exactly 'Tamilnadu agriculture yield data.csv'")
    print("Available files in yield_data:", os.listdir(os.path.join(PROJECT_ROOT, 'data', 'yield_data')))
    exit(1)

# Load & clean dataset
df = pd.read_csv(YIELD_CSV)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

print("Dataset loaded. Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Select features & target
features = ['crop_year', 'season', 'crop', 'district_name', 'area']
target = 'production'

df = df[features + [target]].dropna()
df = df[df['area'] > 0]
df = df[df['production'] > 0]

# Preprocessing pipeline
categorical_features = ['season', 'crop', 'district_name']
numerical_features = ['crop_year', 'area']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42))
])

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training yield model...")
model_pipeline.fit(X_train, y_train)

# Evaluate
train_score = model_pipeline.score(X_train, y_train)
test_score = model_pipeline.score(X_test, y_test)
print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")

# Save full pipeline
model_save_path = os.path.join(MODEL_DIR, 'yield_model_pipeline.pkl')
joblib.dump(model_pipeline, model_save_path)
print(f"Yield model pipeline saved to: {model_save_path}")