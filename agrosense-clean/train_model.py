"""
train_model.py — Train and save all ML models for AgroSense AI
Run this once before starting the Flask app if models are not pre-built.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, 'Crop_recommendation.csv')
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"   Shape: {df.shape} | Crops: {df['label'].nunique()}")

X = df.drop('label', axis=1)
y = df['label']

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── Random Forest ──────────────────────────────
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"   ✅ Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# ── Gradient Boosting ──────────────────────────
print("\n🚀 Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"   ✅ Accuracy: {gb_acc:.4f} ({gb_acc*100:.2f}%)")

# ── Logistic Regression ────────────────────────
print("\n📊 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"   ✅ Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

# ── Save artifacts ─────────────────────────────
print("\n💾 Saving models...")
joblib.dump(rf,     os.path.join(MODELS_DIR, 'rf_model.pkl'))
joblib.dump(gb,     os.path.join(MODELS_DIR, 'gb_model.pkl'))
joblib.dump(lr,     os.path.join(MODELS_DIR, 'lr_model.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(le,     os.path.join(MODELS_DIR, 'label_encoder.pkl'))

print("\n✅ All models saved to /models/")
print("\n📊 Classification Report (Random Forest):")
print(classification_report(y_test, rf.predict(X_test), target_names=le.classes_))
