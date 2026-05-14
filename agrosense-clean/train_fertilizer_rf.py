"""
AgroSense AI — Fertilizer Recommendation Model Trainer
Trains a Random Forest classifier + supporting encoders.

Run:  python train_fertilizer_rf.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE, "Fertilizer_Recommendation.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset…")
df = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(df):,}   Columns: {list(df.columns)}")
print(f"  Fertilizers: {df['Fertilizer'].nunique()} classes")
print(f"\n{df['Fertilizer'].value_counts().to_string()}\n")

# ── Encode categoricals ───────────────────────────────────────────────────────
crop_le  = LabelEncoder()
soil_le  = LabelEncoder()
fert_le  = LabelEncoder()

df["Crop_enc"]       = crop_le.fit_transform(df["Crop"])
df["Soil_enc"]       = soil_le.fit_transform(df["Soil_Type"])
df["Fertilizer_enc"] = fert_le.fit_transform(df["Fertilizer"])

# ── Features and target ───────────────────────────────────────────────────────
FEATURES = ["Crop_enc", "Soil_enc", "Nitrogen", "Phosphorus", "Potassium"]
TARGET   = "Fertilizer_enc"

X = df[FEATURES].values
y = df[TARGET].values

# ── Scale numeric features ────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# ── Train Random Forest ───────────────────────────────────────────────────────
print("\nTraining Random Forest…")
rf = RandomForestClassifier(
    n_estimators    = 300,
    max_depth       = None,
    min_samples_split = 2,
    min_samples_leaf  = 1,
    max_features    = "sqrt",
    class_weight    = "balanced",
    random_state    = 42,
    n_jobs          = -1,
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"  RF Test Accuracy : {rf_acc*100:.2f}%")

# Cross-validation
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")
print(f"  5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Train Gradient Boosting ───────────────────────────────────────────────────
print("\nTraining Gradient Boosting…")
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"  GB Test Accuracy : {gb_acc*100:.2f}%")

# ── Detailed report ───────────────────────────────────────────────────────────
print("\n── Classification Report (Random Forest) ──")
y_pred = rf.predict(X_test)
target_names = fert_le.inverse_transform(np.unique(y))
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# ── Feature importance ────────────────────────────────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("── Feature Importances ──")
for feat, imp in feat_imp.items():
    print(f"  {feat:<20} {imp:.4f}")

# ── Save models and encoders ──────────────────────────────────────────────────
print("\nSaving models…")
joblib.dump(rf,      os.path.join(MODEL_DIR, "fert_rf_model.pkl"))
joblib.dump(gb,      os.path.join(MODEL_DIR, "fert_gb_model.pkl"))
joblib.dump(scaler,  os.path.join(MODEL_DIR, "fert_scaler_v2.pkl"))
joblib.dump(crop_le, os.path.join(MODEL_DIR, "fert_crop_le_v2.pkl"))
joblib.dump(soil_le, os.path.join(MODEL_DIR, "fert_soil_le_v2.pkl"))
joblib.dump(fert_le, os.path.join(MODEL_DIR, "fert_label_encoder_v2.pkl"))

# Save class lists for the API
import json
meta = {
    "crops":        list(crop_le.classes_),
    "soil_types":   list(soil_le.classes_),
    "fertilizers":  list(fert_le.classes_),
    "rf_accuracy":  round(rf_acc * 100, 2),
    "gb_accuracy":  round(gb_acc * 100, 2),
    "cv_accuracy":  round(cv_scores.mean() * 100, 2),
    "features":     FEATURES,
}
with open(os.path.join(MODEL_DIR, "fert_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"  ✅  fert_rf_model.pkl        ({os.path.getsize(os.path.join(MODEL_DIR,'fert_rf_model.pkl'))//1024} KB)")
print(f"  ✅  fert_gb_model.pkl")
print(f"  ✅  fert_scaler_v2.pkl")
print(f"  ✅  fert_crop_le_v2.pkl      classes: {list(crop_le.classes_)}")
print(f"  ✅  fert_soil_le_v2.pkl      classes: {list(soil_le.classes_)}")
print(f"  ✅  fert_label_encoder_v2.pkl classes: {list(fert_le.classes_)}")
print(f"  ✅  fert_meta.json")
print("\n✅  Training complete!")
print(f"    Best model : Random Forest ({rf_acc*100:.2f}% accuracy)")
