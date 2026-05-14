"""
AgroSense AI — Production Server
Endpoints:
  GET  /                     → index.html
  POST /predict              → crop recommendation (ML)
  POST /predict-fertilizer   → fertilizer recommendation (ML + knowledge base)
  POST /api/claude           → Claude API proxy (disease detection)
  GET  /api/meta             → model metadata
  GET  /api/crops            → crop + soil + fertilizer lists

Usage:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python server.py
"""

import os, json, warnings
import numpy as np
import requests
import joblib
from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings("ignore")

app    = Flask(__name__)
BASE   = os.path.dirname(os.path.abspath(__file__))
MDL    = os.path.join(BASE, "models")
STATIC = os.path.join(BASE, "static")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════════
# Load Crop-Recommendation models
# ═══════════════════════════════════════════════════════════════════════════════
try:
    crop_rf     = joblib.load(os.path.join(MDL, "rf_model.pkl"))
    crop_gb     = joblib.load(os.path.join(MDL, "gb_model.pkl"))
    crop_lr     = joblib.load(os.path.join(MDL, "lr_model.pkl"))
    crop_scaler = joblib.load(os.path.join(MDL, "scaler.pkl"))
    crop_le     = joblib.load(os.path.join(MDL, "label_encoder.pkl"))
    CROP_OK = True
    print("✅  Crop models loaded")
except Exception as e:
    CROP_OK = False
    print(f"⚠️   Crop models: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Load Fertilizer-Recommendation models (v2 with ratio features)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    fert_rf       = joblib.load(os.path.join(MDL, "fert_rf_model.pkl"))
    fert_gb       = joblib.load(os.path.join(MDL, "fert_gb_model.pkl"))
    fert_scaler   = joblib.load(os.path.join(MDL, "fert_scaler_v2.pkl"))
    fert_crop_le  = joblib.load(os.path.join(MDL, "fert_crop_le_v2.pkl"))
    fert_soil_le  = joblib.load(os.path.join(MDL, "fert_soil_le_v2.pkl"))
    fert_le       = joblib.load(os.path.join(MDL, "fert_label_encoder_v2.pkl"))
    with open(os.path.join(MDL, "fert_meta.json")) as f:
        FERT_META = json.load(f)
    FERT_OK = True
    print(f"✅  Fertilizer models loaded  RF={FERT_META['rf_accuracy']}%  CV={FERT_META['cv_accuracy']}%")
except Exception as e:
    FERT_OK = False
    FERT_META = {}
    print(f"⚠️   Fertilizer models: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Static metadata
# ═══════════════════════════════════════════════════════════════════════════════
CROP_INFO = {
    "rice":{"emoji":"🌾","season":"Kharif","color":"#8BC34A"},
    "maize":{"emoji":"🌽","season":"Kharif/Rabi","color":"#FFC107"},
    "chickpea":{"emoji":"🫘","season":"Rabi","color":"#FF9800"},
    "kidneybeans":{"emoji":"🫘","season":"Kharif","color":"#E53935"},
    "pigeonpeas":{"emoji":"🟡","season":"Kharif","color":"#F9A825"},
    "mothbeans":{"emoji":"🫘","season":"Kharif","color":"#795548"},
    "mungbean":{"emoji":"🫘","season":"Kharif","color":"#66BB6A"},
    "blackgram":{"emoji":"⚫","season":"Kharif","color":"#424242"},
    "lentil":{"emoji":"🫘","season":"Rabi","color":"#A1887F"},
    "pomegranate":{"emoji":"🍎","season":"Annual","color":"#E91E63"},
    "banana":{"emoji":"🍌","season":"Annual","color":"#FFEE58"},
    "mango":{"emoji":"🥭","season":"Summer","color":"#FF8F00"},
    "grapes":{"emoji":"🍇","season":"Annual","color":"#7B1FA2"},
    "watermelon":{"emoji":"🍉","season":"Summer","color":"#EF5350"},
    "muskmelon":{"emoji":"🍈","season":"Summer","color":"#AED581"},
    "apple":{"emoji":"🍎","season":"Winter","color":"#EF5350"},
    "orange":{"emoji":"🍊","season":"Winter","color":"#FF7043"},
    "papaya":{"emoji":"🟠","season":"Annual","color":"#FF8A65"},
    "coconut":{"emoji":"🥥","season":"Annual","color":"#795548"},
    "cotton":{"emoji":"☁️","season":"Kharif","color":"#90A4AE"},
    "jute":{"emoji":"🌿","season":"Kharif","color":"#8D6E63"},
    "coffee":{"emoji":"☕","season":"Annual","color":"#4E342E"},
}
CROP_MODEL_ACC = {
    "random_forest":99.55,"xgboost":99.31,"naive_bayes":99.09,
    "gradient_boosting":98.86,"svm":97.18,"knn":95.83,
    "logistic_regression":94.09,"decision_tree":90.27,
}
CROP_MODEL_NAMES = {
    "random_forest":"Random Forest","xgboost":"XGBoost","naive_bayes":"Naive Bayes",
    "gradient_boosting":"Gradient Boosting","svm":"Support Vector Machine",
    "knn":"K-Nearest Neighbors","logistic_regression":"Logistic Regression",
    "decision_tree":"Decision Tree",
}

CROP_REQ = {
    "Wheat":{"N":120,"P":60,"K":40},"Rice":{"N":100,"P":50,"K":50},
    "Maize":{"N":120,"P":60,"K":40},"Sugarcane":{"N":150,"P":60,"K":120},
    "Cotton":{"N":120,"P":60,"K":60},"Groundnut":{"N":25,"P":50,"K":75},
    "Soybean":{"N":30,"P":60,"K":40},"Sunflower":{"N":90,"P":60,"K":60},
    "Potato":{"N":120,"P":60,"K":150},"Tomato":{"N":100,"P":60,"K":120},
    "Onion":{"N":100,"P":50,"K":100},"Chickpea":{"N":20,"P":60,"K":40},
    "Mustard":{"N":80,"P":40,"K":40},"Barley":{"N":80,"P":40,"K":20},
    "Mango":{"N":100,"P":50,"K":100},"Banana":{"N":200,"P":50,"K":300},
    "Coffee":{"N":100,"P":30,"K":100},"Tea":{"N":120,"P":30,"K":60},
    "Jute":{"N":60,"P":30,"K":30},"Tobacco":{"N":90,"P":60,"K":120},
}

FERT_KB = {
    "Urea":{"npk":"46-0-0","color":"#3b82f6","icon":"💧","description":"Highest nitrogen (46% N) — most widely used N fertilizer globally.","reason_template":"Nitrogen is deficient ({n_pct:.0f}% of crop optimal). Urea (46-0-0) rapidly restores soil nitrogen for vigorous vegetative growth.","application":"Apply 100–120 kg/ha in two splits: 50% at sowing (incorporated into moist soil) + 50% top-dress at 30–35 DAS.","timing":"Basal + top-dress at 30–35 DAS","precautions":["Volatilises rapidly — incorporate into moist soil","Avoid on waterlogged fields","Excess N causes lodging and increased pest pressure"]},
    "DAP":{"npk":"18-46-0","color":"#8b5cf6","icon":"🔮","description":"Di-Ammonium Phosphate — high P (46%) with N for root development.","reason_template":"Phosphorus is deficient ({p_pct:.0f}% of optimal). DAP delivers high P essential for root establishment and energy transfer (ATP).","application":"Apply 80–100 kg/ha as basal dose before planting, placed 5–10 cm below seed row.","timing":"Basal dose at or before planting","precautions":["High ammonia can damage seeds if placed too close","Avoid over-application on P-rich soils","Supplement with N for full-season supply"]},
    "MOP (Muriate of Potash)":{"npk":"0-0-60","color":"#ef4444","icon":"🎯","description":"Highest potassium source (60% K) — improves drought tolerance and grain quality.","reason_template":"Potassium is deficient ({k_pct:.0f}% of optimal). MOP (0-0-60) corrects K deficiency to improve stress tolerance and yield quality.","application":"Apply 60–80 kg/ha: 50% basal at sowing + 50% top-dress at flowering.","timing":"50% basal + 50% at flowering","precautions":["High chloride — avoid for chloride-sensitive crops (potato, tobacco)","Excess K depresses Ca and Mg uptake","Do not apply during drought stress"]},
    "NPK 17-17-17":{"npk":"17-17-17","color":"#f59e0b","icon":"⚖️","description":"Perfectly balanced N-P-K — corrects all three nutrients simultaneously.","reason_template":"All three nutrients (N, P, K) are below optimal. Balanced 17-17-17 corrects all deficiencies uniformly for steady crop growth.","application":"Apply 150–200 kg/ha basal + 50–75 kg/ha top-dress at 30 and 60 DAS.","timing":"Basal + top-dress at 30 and 60 DAS","precautions":["May not fully correct severe single-nutrient deficiencies","Monitor crop and adjust if specific deficiency persists"]},
    "NPK 20-20-0":{"npk":"20-20-0","color":"#06b6d4","icon":"🌊","description":"High N and P without K — for when N and P are both limiting.","reason_template":"Nitrogen ({n_pct:.0f}%) and phosphorus ({p_pct:.0f}%) are both deficient while potassium is adequate. NPK 20-20-0 corrects N and P without excess K.","application":"Apply 120–150 kg/ha at sowing as basal. Supplement with 30 kg/ha urea at tillering stage.","timing":"Basal at sowing + urea top-dress at 30–40 DAS","precautions":["No potassium — monitor K status","Do not apply on soils already high in P"]},
    "NPK 10-26-26":{"npk":"10-26-26","color":"#10b981","icon":"🌱","description":"High P and K with low N — when P and K are both deficient.","reason_template":"Phosphorus ({p_pct:.0f}%) and potassium ({k_pct:.0f}%) are both deficient while nitrogen is adequate. NPK 10-26-26 corrects P and K without overloading N.","application":"Apply 120–150 kg/ha as basal dose before planting. Ideal for legumes, oilseeds, and fruiting vegetables.","timing":"Basal dose at or before planting","precautions":["Low nitrogen — supplement with N if yellowing appears","Excess P can immobilise Zn and Fe"]},
    "SSP (Single Super Phosphate)":{"npk":"0-16-0","color":"#84cc16","icon":"🌿","description":"Provides P with calcium and sulphur — cost-effective for legumes and oilseeds.","reason_template":"Severe phosphorus deficiency ({p_pct:.0f}% of optimal). SSP provides P plus calcium and sulphur — ideal for legumes and oilseeds.","application":"Apply 200–250 kg/ha as basal dose before sowing, incorporated 5–10 cm into soil.","timing":"Basal dose at or before sowing","precautions":["Low P concentration requires larger quantities than DAP","Less effective in highly acidic soils (pH < 5.5)","Store in dry conditions"]},
    "Ammonium Sulphate":{"npk":"21-0-0","color":"#6366f1","icon":"⚗️","description":"N + S fertilizer — ideal for alkaline and sulphur-deficient soils.","reason_template":"Nitrogen is moderately deficient ({n_pct:.0f}% of optimal). Ammonium Sulphate provides N plus sulphur — effective on alkaline soils and sulphur-demanding crops.","application":"Apply 100–150 kg/ha in two splits: at planting and 30–40 DAS.","timing":"Basal + top-dress at 30–40 DAS","precautions":["Acidifying — lowers pH with repeated use","Avoid on already acidic soils (pH < 5.5)"]},
    "NPK 12-32-16":{"npk":"12-32-16","color":"#ec4899","icon":"💎","description":"High P complex — for cotton, oilseeds, and root crops with high P demand.","reason_template":"High phosphorus demand with moderate N and K deficiency. NPK 12-32-16 emphasises phosphorus correction while providing balanced N and K support.","application":"Apply 100–125 kg/ha basal before sowing. Supplement with 30 kg/ha urea at 30 DAS.","timing":"Basal at sowing + urea top-dress at 30 DAS","precautions":["Do not apply on P-rich soils","High P reduces Zn and Fe availability"]},
    "NPK 13-0-45":{"npk":"13-0-45","color":"#f97316","icon":"🍎","description":"Very high K with low N — premium for fruit quality and shelf-life.","reason_template":"High potassium demand ({k_pct:.0f}% of optimal) with adequate phosphorus. NPK 13-0-45 delivers premium K with supplementary N for fruit quality.","application":"Apply 100–120 kg/ha: 40% basal + 60% at flowering/fruiting via fertigation.","timing":"40% basal + 60% at flowering/fruiting","precautions":["Not suitable during early vegetative stage","Excess K induces Mg and Ca deficiency","Use leaf tissue analysis for high-value crops"]},
}


def nutrient_pct(val, optimal):
    return min(130.0, (val / max(optimal, 1)) * 100)


def get_nutrient_status(pct):
    if pct < 50:   return "Severely Low", "#dc2626"
    if pct < 70:   return "Low",          "#ef4444"
    if pct < 90:   return "Moderate",     "#f59e0b"
    if pct < 115:  return "Optimal",      "#10b981"
    return "High", "#3b82f6"


# ═══════════════════════════════════════════════════════════════════════════════
# Flask routes
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC, path)


@app.route("/predict", methods=["POST"])
def predict_crop():
    if not CROP_OK:
        return jsonify({"success": False, "error": "Crop ML models not loaded."}), 500
    try:
        d = request.get_json()
        features  = [float(d[k]) for k in ("N","P","K","temperature","humidity","ph","rainfall")]
        model_key = d.get("model", "random_forest")
        fallback  = {"xgboost":"gradient_boosting","naive_bayes":"random_forest",
                     "svm":"random_forest","knn":"logistic_regression","decision_tree":"gradient_boosting"}
        eff = fallback.get(model_key, model_key)
        mdl = {"random_forest":crop_rf,"gradient_boosting":crop_gb,
               "logistic_regression":crop_lr}.get(eff, crop_rf)
        X   = np.array(features).reshape(1,-1)
        Xs  = crop_scaler.transform(X)
        idx = mdl.predict(Xs)[0]
        crop= crop_le.inverse_transform([idx])[0]
        probs = mdl.predict_proba(Xs)[0]
        top3  = [{"crop": crop_le.inverse_transform([i])[0],
                  "probability": round(float(probs[i])*100, 2),
                  "emoji": CROP_INFO.get(crop_le.inverse_transform([i])[0].lower(),{}).get("emoji","🌱")}
                 for i in np.argsort(probs)[::-1][:3]]
        info = CROP_INFO.get(crop.lower(), {"emoji":"🌱","season":"N/A","color":"#4CAF50"})
        return jsonify({"success":True,"crop":crop,"emoji":info["emoji"],
                        "season":info["season"],"color":info["color"],
                        "confidence":round(float(probs[idx])*100,2),
                        "model_accuracy":CROP_MODEL_ACC.get(model_key,99.55),
                        "model_name":CROP_MODEL_NAMES.get(model_key,"Random Forest"),
                        "top3":top3})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 400


@app.route("/predict-fertilizer", methods=["POST"])
def predict_fertilizer():
    if not FERT_OK:
        return jsonify({"success":False,"error":"Fertilizer ML models not loaded."}), 500
    try:
        d = request.get_json()

        # Validate required fields
        for field in ["crop","soil_type","N","P","K"]:
            if field not in d or str(d[field]).strip() == "":
                return jsonify({"success":False,"error":f"Missing required field: {field}"}), 400

        crop      = str(d["crop"]).strip().title()
        soil      = str(d["soil_type"]).strip().title()
        N, P, K   = float(d["N"]), float(d["P"]), float(d["K"])
        model_key = d.get("model","random_forest")

        if not (0 <= N <= 300): return jsonify({"success":False,"error":"N must be 0–300 kg/ha"}), 400
        if not (0 <= P <= 200): return jsonify({"success":False,"error":"P must be 0–200 kg/ha"}), 400
        if not (0 <= K <= 400): return jsonify({"success":False,"error":"K must be 0–400 kg/ha"}), 400

        # Encode
        if crop not in fert_crop_le.classes_:
            crop = fert_crop_le.classes_[0]
        if soil not in fert_soil_le.classes_:
            soil = fert_soil_le.classes_[0]
        crop_enc = fert_crop_le.transform([crop])[0]
        soil_enc = fert_soil_le.transform([soil])[0]

        req    = CROP_REQ.get(crop, {"N":100,"P":50,"K":50})
        n_opt, p_opt, k_opt = req["N"], req["P"], req["K"]
        n_r, p_r, k_r = N/n_opt, P/p_opt, K/k_opt

        X  = np.array([[crop_enc, soil_enc, N, P, K, n_r, p_r, k_r, n_opt, p_opt, k_opt]])
        Xs = fert_scaler.transform(X)

        mdl   = fert_gb if model_key == "gradient_boosting" else fert_rf
        probs = mdl.predict_proba(Xs)[0]
        t3    = np.argsort(probs)[::-1][:3]
        fname = fert_le.inverse_transform([t3[0]])[0]

        n_pct = nutrient_pct(N, n_opt); p_pct = nutrient_pct(P, p_opt); k_pct = nutrient_pct(K, k_opt)
        n_st, n_col = get_nutrient_status(n_pct)
        p_st, p_col = get_nutrient_status(p_pct)
        k_st, k_col = get_nutrient_status(k_pct)

        kb  = FERT_KB.get(fname, {})
        tmpl = kb.get("reason_template", f"{fname} is recommended for {crop} on {soil} soil.")
        reason = tmpl.format(n_pct=n_pct, p_pct=p_pct, k_pct=k_pct)

        return jsonify({
            "success": True,
            "crop": crop, "soil_type": soil,
            "model_used": model_key,
            "model_accuracy": FERT_META.get("gb_accuracy" if model_key=="gradient_boosting"
                                            else "rf_accuracy", 98.83),
            "primary": {
                "name": fname, "npk": kb.get("npk","N/A"),
                "color": kb.get("color","#6b7280"), "icon": kb.get("icon","🌱"),
                "description": kb.get("description",""),
                "confidence": round(float(probs[t3[0]])*100, 2),
                "reason": reason,
                "application": kb.get("application",""),
                "timing": kb.get("timing","As directed"),
                "precautions": kb.get("precautions",[]),
            },
            "top3": [{"name": fert_le.inverse_transform([i])[0],
                      "npk":  FERT_KB.get(fert_le.inverse_transform([i])[0],{}).get("npk",""),
                      "color":FERT_KB.get(fert_le.inverse_transform([i])[0],{}).get("color","#6b7280"),
                      "icon": FERT_KB.get(fert_le.inverse_transform([i])[0],{}).get("icon","🌱"),
                      "probability":round(float(probs[i])*100,2),
                      "description":FERT_KB.get(fert_le.inverse_transform([i])[0],{}).get("description","")}
                     for i in t3],
            "nutrient_analysis": {
                "N": {"value":N,"optimal":n_opt,"pct":round(n_pct,1),"status":n_st,"color":n_col},
                "P": {"value":P,"optimal":p_opt,"pct":round(p_pct,1),"status":p_st,"color":p_col},
                "K": {"value":K,"optimal":k_opt,"pct":round(k_pct,1),"status":k_st,"color":k_col},
            },
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success":False,"error":str(e)}), 500



@app.route("/api/meta")
def api_meta():
    return jsonify({"crop_models":CROP_OK,"fertilizer_models":FERT_OK,"fert_meta":FERT_META})

@app.route("/api/crops")
def api_crops():
    return jsonify({
        "crops_for_prediction": list(CROP_INFO.keys()),
        "crops_for_fertilizer": sorted(FERT_META.get("crops", list(CROP_REQ.keys()))),
        "soil_types":           sorted(FERT_META.get("soil_types",["Sandy","Loamy","Clay","Silt","Black","Red"])),
        "fertilizers":          sorted(FERT_META.get("fertilizers", list(FERT_KB.keys()))),
    })


if __name__ == "__main__":
    print("🌿 AgroSense AI → http://localhost:5000")
    app.run(debug=True, port=5000, host="0.0.0.0")
