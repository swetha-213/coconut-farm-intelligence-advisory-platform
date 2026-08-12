# main.py - Coconut Yield, Disease & Price Prediction Project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import os
import warnings
from prophet import Prophet

warnings.filterwarnings('ignore')

# Folder Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PRICE_CSV = os.path.join(DATA_DIR, 'price_data', 'clean_prices.csv')
YIELD_CSV = os.path.join(DATA_DIR, 'yield_data', 'Tamilnadu agriculture yield data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PRICE_MODEL_DIR = os.path.join(MODEL_DIR, 'price_models')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRICE_MODEL_DIR, exist_ok=True)

# Debug: Show models folder content at startup
print("[MAIN] MODEL_DIR:", MODEL_DIR)
if os.path.exists(MODEL_DIR):
    print("[MAIN] Files in models:", os.listdir(MODEL_DIR))
else:
    print("[MAIN] MODEL_DIR not found!")

# =============================================
# 1. Yield Prediction Model (XGBoost Pipeline)
# =============================================

def train_yield_model():
    try:
        df = pd.read_csv(YIELD_CSV)
        print("Yield CSV loaded. Shape:", df.shape)
        print("Columns:", df.columns.tolist())
    except Exception as e:
        print("Yield CSV error:", str(e))
        return

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace(r'[^a-z0-9_]', '', regex=True)

    target_col = 'production'
    area_col = 'area'

    if target_col not in df.columns or area_col not in df.columns:
        print("Required columns missing. Available:", df.columns.tolist())
        return

    # Clean data
    df = df[df[area_col] > 0]
    df = df.dropna(subset=[target_col, area_col])

    # Features & target
    features = ['crop_year', 'season', 'crop', 'district_name', 'area']
    df = df[features + [target_col]].dropna()

    X = df[features]
    y = df[target_col]

    # Train-test split (for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Full pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    categorical_features = ['season', 'crop', 'district_name']
    numerical_features = ['crop_year', 'area']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42))
    ])

    print("Training yield model...")
    model_pipeline.fit(X_train, y_train)

    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    print(f"Train R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")

    # Save pipeline
    joblib.dump(model_pipeline, os.path.join(MODEL_DIR, 'yield_model_pipeline.pkl'))
    print("Yield model pipeline trained & saved!")

def predict_yield(input_dict):
    model_path = os.path.join(MODEL_DIR, 'yield_model_pipeline.pkl')
    
    # Debug prints
    print(f"[YIELD] Attempting to load model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[YIELD ERROR] Model file not found: {model_path}")
        if os.path.exists(MODEL_DIR):
            print(f"[YIELD] Files in models folder: {os.listdir(MODEL_DIR)}")
        return None

    try:
        model_pipeline = joblib.load(model_path)
        print("[YIELD] Model loaded successfully!")
    except Exception as e:
        print(f"[YIELD ERROR] Load failed: {str(e)}")
        return None

    try:
        df_input = pd.DataFrame([input_dict])
        df_input.columns = df_input.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        print("[YIELD] Input to model:", df_input.to_dict(orient='records'))
        
        predicted = model_pipeline.predict(df_input)[0]
        print(f"[YIELD] Predicted value: {predicted}")
        
        return round(predicted, 2)
    except Exception as e:
        print(f"[YIELD ERROR] Prediction failed: {str(e)}")
        return None

def yield_recommendations(prediction):
    recs = []
    if prediction < 5000:
        recs.append("Very low yield — Check soil, fertilizer, irrigation immediately")
    elif prediction < 12000:
        recs.append("Below average — Consider increasing fertilizer or pest control")
    elif prediction < 25000:
        recs.append("Average yield — Continue current practices")
    else:
        recs.append("Excellent yield — Great job!")
    return recs

# =============================================
# 2. Disease Prediction (Rule-based)
# =============================================

def predict_disease_from_symptoms(symptoms):
    if not symptoms or symptoms.strip() == '':
        return "No symptoms entered", "Please describe the symptoms or upload a clear leaf/plant photo"
    
    symptoms = symptoms.lower()
    
    if 'yellow' in symptoms or 'yellow leaves' in symptoms:
        return "Nutrient Deficiency", "Apply balanced NPK fertilizer (e.g., 10:10:10) and check soil pH. Add micronutrients like zinc/magnesium if needed."
    
    elif 'spot' in symptoms or 'spots' in symptoms or 'black spot' in symptoms:
        return "Leaf Spot Disease", "Spray copper-based fungicide (e.g., Bordeaux mixture). Improve air circulation by pruning and avoid overhead watering."
    
    elif 'bud rot' in symptoms or 'bud rotting' in symptoms or 'rotting' in symptoms:
        return "Bud Rot", "Remove and destroy affected buds/parts immediately. Apply Bordeaux mixture or systemic fungicide to healthy parts."
    
    elif 'wilt' in symptoms or 'wilting' in symptoms:
        return "Possible Wilt / Root Rot", "Improve soil drainage, reduce watering, apply Trichoderma or fungicide. Check for root damage."
    
    else:
        return "Unknown / No clear disease detected", "Upload a clear photo of affected leaves/bud for better diagnosis or consult local agriculture officer."

# =============================================
# 3. Price Forecasting (District-wise Prophet)
# =============================================

def train_price_model():
    try:
        df = pd.read_csv(PRICE_CSV)
        print("Price CSV loaded. Shape:", df.shape)
        print("Raw columns:", df.columns.tolist())
        
        if len(df.columns) == 1:
            print("CSV header issue - fixing...")
            df = pd.read_csv(PRICE_CSV, sep=',', engine='python', on_bad_lines='skip')
            if len(df.columns) == 1:
                df = df.iloc[:,0].str.split(',', expand=True)
                df.columns = ['date', 'region', 'price_INR']
        
        df = df.rename(columns={
            'date': 'ds',
            'region': 'district',
            'price_INR': 'y'
        })
        
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df[['ds', 'y', 'district']].dropna().sort_values(['district', 'ds'])
        
        districts = df['district'].unique()
        print(f"Training for {len(districts)} districts: {list(districts)}")
        
        for district in districts:
            dist_df = df[df['district'] == district][['ds', 'y']]
            if len(dist_df) < 30:
                print(f"Skipping {district} - too few rows ({len(dist_df)})")
                continue
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative'
            )
            
            model.fit(dist_df)
            
            model_path = os.path.join(PRICE_MODEL_DIR, f'prophet_{district}.pkl')
            joblib.dump(model, model_path)
            print(f"Saved price model for {district} at {model_path}")
        
        print("All district price models trained!")
    
    except Exception as e:
        print("Price training error:", str(e))

def forecast_price(district, periods=30):
    model_path = os.path.join(PRICE_MODEL_DIR, f'prophet_{district}.pkl')
    if not os.path.exists(model_path):
        return [{'error': f'No trained model for {district}. Run train_price_model() first.'}]
    
    model = joblib.load(model_path)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
    result = result.round(2)
    
    return result.to_dict('records')

# =============================================
# Run Training & Test (optional - run once)
# =============================================
if __name__ == '__main__':
    print("=== Coconut Farm Models ===")
    
    # Uncomment only if you want to re-train
    # print("\n1. Training Yield Model...")
    # train_yield_model()
    
    # print("\n2. Training District-wise Price Models...")
    # train_price_model()
    
    # Test Yield
    print("\n=== Test Yield Prediction ===")
    test_input = {
        'crop_year': 2025,
        'district_name': 'Coimbatore',
        'area': 5000,
        'season': 'Kharif',
        'crop': 'Coconut'
    }
    yield_pred = predict_yield(test_input)
    if yield_pred is not None:
        print("Predicted Yield (total):", yield_pred)
        print("Recommendations:", yield_recommendations(yield_pred))
    else:
        print("Yield test failed!")
    
    # Test Price
    print("\n=== Test Price Forecast (Coimbatore, 15 days) ===")
    price_fc = forecast_price('Coimbatore', 15)
    if 'error' not in price_fc[0]:
        for row in price_fc:
            print(f"{row['ds']}: ₹{row['yhat']}")
    else:
        print("Price forecast error:", price_fc[0]['error'])