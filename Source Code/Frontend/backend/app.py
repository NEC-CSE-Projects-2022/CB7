





# backend/app.py
import os
import traceback
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ML tools (some optional imports may be done at runtime)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import compute_class_weight

# optional: imblearn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

# Config / paths
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(APP_ROOT, "saved_models")
os.makedirs(MODEL_FOLDER, exist_ok=True)
DATA_FILE = os.path.join(MODEL_FOLDER, "your_merged_ce_data.csv")
MODEL_PATH = os.path.join(MODEL_FOLDER, "Dropout_MLP_model.h5")
SCALER_PATH = os.path.join(MODEL_FOLDER, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_FOLDER, "label_encoder.pkl")

app = Flask(__name__)
CORS(app, origins="*")

# ---------- Utility: safe prints for debugging ----------
def log(*args):
    print(*args, flush=True)

# ---------- Load dataset ----------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found. Please upload your_merged_ce_data.csv to {MODEL_FOLDER}")

df = pd.read_csv(DATA_FILE)
log("Loaded dataset rows:", df.shape[0])
log("City list:", df["City_Name"].unique())
log("Columns:", df.columns.tolist())

# ---------- Helper: feature mapping (based on your CSV) ----------
# We will map friendly names to actual columns found in your CSV.
COLUMN_MAP = {
    "city": "City_Name",
    "current_waste": None,  # try multiple options below
    "waste_tpd": None,
    "recycling_rate": None,
    "population_density": None,
    "municipal_efficiency": None,
    "sdg_score": None,
    "waste_per_capita": None,
    "circular_score": None,
    "ce_action": None
}

# best-guess mapping based on observed column names in your dataset
candidates = {
    "current_waste": ["Waste Generated (Tons/Day)", "Waste_TPD", "Waste_TPD", "Waste_TPD"],
    "waste_tpd": ["Waste_TPD", "Waste Generated (Tons/Day)"],
    "recycling_rate": ["Recycling Rate (%)", "Recycling_Efficiency", "Recycling_Efficiency"],
    "population_density": ["Population Density"],
    "municipal_efficiency": ["Municipal Efficiency Score (1-10)"],
    "sdg_score": ["SDG 12", "Composite Score"],
    "waste_per_capita": ["Waste_per_capita", "Waste_kg_per_capita_per_day"],
    "circular_score": ["Circular_Score", "Circular_Score"],
    "ce_action": ["CE_Action", "CE_Action", "CE_Action"]
}

for key, tries in candidates.items():
    for t in tries:
        if t in df.columns:
            COLUMN_MAP[key] = t
            break

log("Column mapping used:", COLUMN_MAP)

# ---------- Prepare X and y for training if needed ----------
# We will use numeric columns (a subset) as features; but prefer scaler.feature_names_in_ if available later.
NUMERIC_FEATURES = df.select_dtypes(include=["number"]).columns.tolist()
log("Numeric features detected:", NUMERIC_FEATURES)

# y target column
if COLUMN_MAP["ce_action"] is None:
    raise KeyError("CE action column not detected. Ensure your dataset has 'CE_Action' or similar column.")
TARGET_COL = COLUMN_MAP["ce_action"]

# ---------- Retrain logic ----------
def retrain_and_save_model(force_retrain=False):
    """
    Retrains using SMOTE balancing (if available) and saves model, scaler, encoder.
    """
    # retrain if files missing or forced
    need = force_retrain or not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH))
    if not need:
        log("Existing model artifacts found; skipping retrain.")
        return

    log("Starting retraining... (this may take a few minutes)")

    X = df.select_dtypes(include=["number"]).copy()
    # Ensure target not in features
    if TARGET_COL in X.columns:
        X = X.drop(columns=[TARGET_COL])

    y = df[TARGET_COL].astype(str).copy()

    # Fill NaNs
    X = X.fillna(0)
    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    log("Scaler saved.")

    # Label encoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, ENCODER_PATH)
    log("Label encoder saved. Classes:", le.classes_)

    # Balance classes
    if HAS_SMOTE:
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X_scaled, y_enc)
        log("SMOTE applied. Balanced shape:", X_bal.shape, np.bincount(y_bal))
    else:
        # fallback: class weights training
        X_bal, y_bal = X_scaled, y_enc
        log("imblearn.SMOTE not installed; training without oversampling. Consider installing imblearn for better balance.")

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal if HAS_SMOTE else y_bal)

    # simple MLP (Dropout variant)
    model = Sequential([
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(len(le.classes_), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1, verbose=1)

    model.save(MODEL_PATH)
    log("Model trained and saved to", MODEL_PATH)

# call retrain if artifacts missing
retrain_and_save_model(force_retrain=False)

# ---------- Load artifacts (after retrain or if already exist) ----------
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
log("Loaded model, scaler, and encoder. Label classes:", label_encoder.classes_)

# ---------- API endpoints ----------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": label_encoder is not None,
        "cities_count": int(df["City_Name"].nunique())
    })

@app.route("/api/cities", methods=["GET"])
def cities():
    """Return sorted list of city names for the frontend dropdown"""
    cities = sorted(df["City_Name"].unique().tolist())
    return jsonify({"cities": cities})

@app.route("/api/forecast", methods=["POST"])
def forecast():
    """
    POST json: {"city": "Delhi"}
    """
    try:
        payload = request.get_json(force=True)
        city_name = payload.get("city", "").strip()
        if not city_name:
            return jsonify({"success": False, "message": "City not provided"}), 400

        # exact match (case-insensitive)
        city_data = df[df["City_Name"].str.lower() == city_name.lower()]
        if city_data.empty:
            # try contains as fallback
            city_data = df[df["City_Name"].str.contains(city_name, case=False, na=False)]
        if city_data.empty:
            return jsonify({"success": False, "message": "City not found in dataset"}), 404

        # Prepare features for prediction using scaler.feature_names_in_ if available
        try:
            required = list(scaler.feature_names_in_)
        except Exception:
            required = df.select_dtypes(include=["number"]).columns.tolist()
            if TARGET_COL in required:
                required.remove(TARGET_COL)

        # Ensure required columns present
        for c in required:
            if c not in city_data.columns:
                city_data[c] = 0.0

        X_city = city_data[required].fillna(0.0)
        X_scaled = scaler.transform(X_city)

        preds = model.predict(X_scaled)
        pred_classes = np.argmax(preds, axis=1)
        # use last row prediction for city (more recent)
        last_pred = pred_classes[-1]
        action_label = label_encoder.inverse_transform([last_pred])[0]

        # produce a simple confidence (max prob of last sample)
        confidence = float(np.max(preds[-1]) * 100)

        # extract dataset-derived values safely
        def safe_get(series, candidates, default=0.0):
            for c in candidates:
                if c in city_data.columns:
                    return float(city_data[c].iloc[-1])
            return default

        current_waste = safe_get(city_data, ["Waste Generated (Tons/Day)", "Waste_TPD", "Waste_TPD", "Waste_TPD"], 0.0)
        recycling_rate = safe_get(city_data, ["Recycling Rate (%)", "Recycling_Efficiency"], 0.0)
        municipal_eff = safe_get(city_data, ["Municipal Efficiency Score (1-10)"], 0.0)
        population_density = safe_get(city_data, ["Population Density"], 0.0)
        sdg_score = safe_get(city_data, ["SDG 12", "Composite Score"], 0.0)
        waste_per_capita = safe_get(city_data, ["Waste_per_capita", "Waste_kg_per_capita_per_day"], 0.0)
        circular_score = safe_get(city_data, ["Circular_Score"], 0.0)

        # simple mathematically predicted waste (3.5% growth) — replace with your LSTM if available
        predicted_waste = round(current_waste * 1.035, 2) if current_waste > 0 else 0.0
        growth_rate = round(((predicted_waste - current_waste) / current_waste) * 100, 2) if current_waste > 0 else 0.0

        result = {
            "success": True,
            "city": city_name,
            "circular_economy_action": action_label,
            "confidence": confidence,
            "waste_forecast": {
                "current_waste": current_waste,
                "predicted_waste": predicted_waste,
                "growth_rate": growth_rate
            },
            "recycling_rate": recycling_rate,
            "municipal_efficiency": municipal_eff,
            "population_density": population_density,
            "sdg_score": sdg_score,
            "waste_per_capita": waste_per_capita,
            "circular_score": circular_score,
            "recommendations": {
                "Rethink": [
                    "Launch awareness campaigns and reduce waste generation.",
                    "Promote segregation at source and responsible consumption."
                ],
                "Redesign": [
                    "Upgrade waste collection infrastructure (smart bins, logistics).",
                    "Implement EPR and product redesign incentives."
                ],
                "Reuse": [
                    "Support repair, refurbish and upcycling startups.",
                    "Build industrial symbiosis and material exchange platforms."
                ]
            }.get(action_label, [])
        }
        return jsonify(result), 200

    except Exception as e:
        log("ERROR in /api/forecast:", e)
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

# Optionally expose basic city info (for charts)
@app.route("/api/cityInfo", methods=["GET"])
def city_info():
    # returns simple arrays for plotting (waste vs pop density)
    try:
        x = df["City_Name"].tolist()
        y_waste = (df["Waste_TPD"].fillna(0).tolist() if "Waste_TPD" in df.columns else df.get("Waste Generated (Tons/Day)", pd.Series([0]*len(df))).fillna(0).tolist())
        y_pop = df["Population Density"].fillna(0).tolist() if "Population Density" in df.columns else [0]*len(df)
        return jsonify({"cities": x, "waste": y_waste, "population_density": y_pop}), 200
    except Exception as e:
        log("ERROR in /api/cityInfo:", e)
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    log("Starting SmartWasteNet Flask API on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)



