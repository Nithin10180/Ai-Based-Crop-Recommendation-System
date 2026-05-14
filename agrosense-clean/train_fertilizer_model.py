"""
Fertilizer Recommendation Model Trainer
Generates a trained Random Forest model + label encoder + scaler
saved to models/fertilizer_model.pkl etc.
Run once: python train_fertilizer_model.py
"""

import numpy as np
import joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Synthetic training data ─────────────────────────────────────────────────
# Features: crop_enc, soil_enc, temperature, humidity, rainfall, N, P, K, pH, stage_enc
# Label:    fertilizer name

FERTILIZERS = [
    'Urea', 'DAP', 'NPK 10-26-26', 'NPK 17-17-17', 'NPK 20-20-0',
    'Muriate of Potash (MOP)', 'Single Super Phosphate (SSP)',
    'Ammonium Sulphate', 'NPK 12-32-16', 'NPK 13-0-45',
]

CROPS = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
    'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee',
]
SOILS  = ['clay', 'sandy', 'loamy', 'silt', 'black', 'red']
STAGES = ['seedling', 'vegetative', 'flowering', 'fruiting', 'maturity']

crop_le  = LabelEncoder().fit(CROPS)
soil_le  = LabelEncoder().fit(SOILS)
stage_le = LabelEncoder().fit(STAGES)
fert_le  = LabelEncoder().fit(FERTILIZERS)

rng = np.random.default_rng(42)
n   = 6000

crop_idx  = rng.integers(0, len(CROPS),  n)
soil_idx  = rng.integers(0, len(SOILS),  n)
stage_idx = rng.integers(0, len(STAGES), n)
temp      = rng.uniform(15, 45, n)
humidity  = rng.uniform(20, 95, n)
rainfall  = rng.uniform(20, 300, n)
N         = rng.uniform(0, 140, n)
P         = rng.uniform(5, 145, n)
K         = rng.uniform(5, 205, n)
pH        = rng.uniform(3.5, 9.9, n)

X = np.column_stack([crop_idx, soil_idx, temp, humidity, rainfall, N, P, K, pH, stage_idx])

# Rule-based label generation (agronomically informed)
def label(crop_i, soil_i, t, h, r, ni, pi, ki, ph_v, stg_i):
    crop  = CROPS[crop_i]
    stage = STAGES[stg_i]
    soil  = SOILS[soil_i]

    # Nitrogen deficient → Urea or Ammonium Sulphate
    if ni < 25:
        return 'Urea' if soil != 'sandy' else 'Ammonium Sulphate'
    # P deficient → DAP or SSP
    if pi < 20:
        return 'DAP' if stage in ('seedling', 'vegetative') else 'Single Super Phosphate (SSP)'
    # K deficient → MOP
    if ki < 20:
        return 'Muriate of Potash (MOP)'

    # Crop-specific
    if crop in ('rice', 'jute', 'blackgram', 'mungbean'):
        return 'Urea' if stage == 'vegetative' else 'NPK 10-26-26'
    if crop in ('maize', 'chickpea', 'lentil'):
        return 'NPK 20-20-0' if stage != 'flowering' else 'NPK 10-26-26'
    if crop == 'cotton':
        return 'NPK 12-32-16'
    if crop in ('mango', 'apple', 'orange', 'grapes'):
        return 'NPK 13-0-45' if stage in ('flowering', 'fruiting') else 'NPK 17-17-17'
    if crop in ('banana', 'papaya', 'coconut'):
        return 'Muriate of Potash (MOP)' if ki < 80 else 'NPK 17-17-17'
    if crop == 'coffee':
        return 'NPK 17-17-17'
    if crop in ('pomegranate', 'watermelon', 'muskmelon'):
        return 'NPK 13-0-45' if stage == 'fruiting' else 'DAP'
    if crop in ('pigeonpeas', 'mothbeans', 'kidneybeans'):
        return 'Single Super Phosphate (SSP)' if stage == 'seedling' else 'NPK 10-26-26'

    return 'NPK 17-17-17'

y_labels = [label(int(crop_idx[i]), int(soil_idx[i]), temp[i], humidity[i],
                  rainfall[i], N[i], P[i], K[i], pH[i], int(stage_idx[i]))
            for i in range(n)]
y = fert_le.transform(y_labels)

# ── Train ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train_s, y_train)
acc = accuracy_score(y_test, model.predict(X_test_s))
print(f"Fertilizer model accuracy: {acc*100:.2f}%")

# ── Save ─────────────────────────────────────────────────────────────────────
joblib.dump(model,    os.path.join(MODELS_DIR, 'fertilizer_model.pkl'))
joblib.dump(scaler,   os.path.join(MODELS_DIR, 'fertilizer_scaler.pkl'))
joblib.dump(fert_le,  os.path.join(MODELS_DIR, 'fertilizer_label_encoder.pkl'))
joblib.dump(crop_le,  os.path.join(MODELS_DIR, 'fertilizer_crop_le.pkl'))
joblib.dump(soil_le,  os.path.join(MODELS_DIR, 'fertilizer_soil_le.pkl'))
joblib.dump(stage_le, os.path.join(MODELS_DIR, 'fertilizer_stage_le.pkl'))

print("All fertilizer model files saved to models/")
