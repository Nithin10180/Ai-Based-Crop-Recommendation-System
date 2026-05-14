# 🌿 AgroSense AI — Precision Agriculture Intelligence

AI-powered smart agriculture platform with **four integrated modules** — trained ML models + Claude Vision AI.

---

## 🚀 Quick Start (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key (required for Disease Detection)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Start the server
python server.py
```

Then open **http://localhost:5000** in your browser.

---

## 🧩 Four Modules

| # | Module | Nav Button | ML Endpoint | Accuracy |
|---|---|---|---|---|
| 1 | 🌾 **Crop Prediction** | Crop ▼ → Prediction | `POST /predict` | 99.55% RF |
| 2 | 🌱 **Fertilizer Advisor** | Fertilizer ▼ (separate) | `POST /predict-fertilizer` | 98.83% RF |
| 3 | 🔬 **Disease Detection** | Crop ▼ → Disease | `POST /api/claude` | Vision AI |
| 4 | 📋 **History** | History button | — | — |

> The **Fertilizer** button is now a **separate nav button** (amber colored), independent of the Crop dropdown.

---

## 🤖 ML Models

### Crop Prediction
| Model | Accuracy |
|---|---|
| Random Forest | **99.55%** |
| XGBoost | 99.31% |
| Naive Bayes | 99.09% |
| Gradient Boosting | 98.86% |
| SVM | 97.18% |
| K-Nearest Neighbors | 95.83% |
| Logistic Regression | 94.09% |
| Decision Tree | 90.27% |

**Input:** N, P, K (kg/ha), temperature (°C), humidity (%), pH, rainfall (mm)
**Output:** Best crop, confidence, top-3 alternatives, season

### Fertilizer Recommendation
| Model | Accuracy |
|---|---|
| Random Forest | **98.83%** (test), 98.93% (5-fold CV) |
| Gradient Boosting | 99.33% |

**Input:** crop_type, soil_type, N, P, K (kg/ha)
**Output:** Fertilizer name, NPK ratio, reason, application guide, timing, precautions, nutrient analysis, top-3 alternatives

**Fertilizers:** Urea, DAP, MOP, NPK 17-17-17, NPK 20-20-0, NPK 10-26-26, SSP, Ammonium Sulphate, NPK 12-32-16, NPK 13-0-45

**Crops:** Wheat, Rice, Maize, Sugarcane, Cotton, Groundnut, Soybean, Sunflower, Potato, Tomato, Onion, Chickpea, Mustard, Barley, Mango, Banana, Coffee, Tea, Jute, Tobacco

**Soils:** Sandy, Loamy, Clay, Silt, Black (Regur), Red Laterite

---

## 📡 API Reference

### `POST /predict`
```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 20.9, "humidity": 82.0,
  "ph": 6.5, "rainfall": 202.9,
  "model": "random_forest"
}
```

### `POST /predict-fertilizer`
```json
{
  "crop": "Rice",
  "soil_type": "Clay",
  "N": 15, "P": 12, "K": 20,
  "model": "random_forest"
}
```

### `POST /api/claude`
Secure server-side proxy to Anthropic API. Accepts standard Anthropic messages payload (used for Disease Detection). Your API key stays on the server — never exposed to the browser.

### `GET /api/meta`
Returns model load status and accuracy metadata.

### `GET /api/crops`
Returns lists of supported crops, soil types, fertilizers.

---

## 🐛 Bug Fixes (latest)

- ✅ **JSON parse error fixed** — `safeParseJSON()` robustly handles truncated AI responses with bracket counting and string repair
- ✅ **Fertilizer is now a separate nav button** (amber, independent of Crop dropdown)
- ✅ **Disease detection** uses `max_tokens:1200` + compact prompt to avoid truncation
- ✅ **Error banners** shown inline instead of alert dialogs
- ✅ **Model files** correctly placed in `models/` directory

---

## 📁 Project Structure

```
agrosense-v2/
├── server.py                     # Flask backend — all API endpoints
├── index.html                    # Complete frontend SPA (all 4 modules)
├── requirements.txt
├── README.md
├── Crop_recommendation.csv       # 2200-row crop dataset
├── Fertilizer_Recommendation.csv # 3000-row generated fertilizer dataset
├── generate_fertilizer_data.py   # Regenerates fertilizer dataset
├── train_fertilizer_rf.py        # Trains RF + GB fertilizer models
├── train_model.py                # Trains crop models
└── models/
    ├── rf_model.pkl              ─┐
    ├── gb_model.pkl               │ Crop recommendation
    ├── lr_model.pkl               │ ML models
    ├── scaler.pkl                 │
    ├── label_encoder.pkl         ─┘
    ├── fert_rf_model.pkl         ─┐
    ├── fert_gb_model.pkl          │ Fertilizer recommendation
    ├── fert_scaler_v2.pkl         │ ML models
    ├── fert_crop_le_v2.pkl        │
    ├── fert_soil_le_v2.pkl        │
    ├── fert_label_encoder_v2.pkl  │
    └── fert_meta.json            ─┘
```

---

## 🔁 Retrain Models

```bash
# Regenerate fertilizer dataset (3000 rows)
python generate_fertilizer_data.py

# Retrain fertilizer models (RF + GB)
python train_fertilizer_rf.py

# Retrain crop models
python train_model.py
```

---

## 🔑 Disease Detection API Key

Get your key at: **https://console.anthropic.com**

The key is used server-side only via `/api/claude` proxy — never sent to the browser.
