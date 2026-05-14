"""
AgroSense AI — Fertilizer Dataset Generator
Generates a realistic, rule-based fertilizer recommendation dataset.

Run:  python generate_fertilizer_data.py
Output: Fertilizer_Recommendation.csv
"""

import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

# ── Crop requirements: (N_optimal, P_optimal, K_optimal) ─────────────────────
CROP_REQUIREMENTS = {
    "Wheat":        (120, 60, 40),
    "Rice":         (100, 50, 50),
    "Maize":        (120, 60, 40),
    "Sugarcane":    (150, 60, 120),
    "Cotton":       (120, 60, 60),
    "Groundnut":    (25,  50, 75),
    "Soybean":      (30,  60, 40),
    "Sunflower":    (90,  60, 60),
    "Potato":       (120, 60, 150),
    "Tomato":       (100, 60, 120),
    "Onion":        (100, 50, 100),
    "Chickpea":     (20,  60, 40),
    "Mustard":      (80,  40, 40),
    "Barley":       (80,  40, 20),
    "Mango":        (100, 50, 100),
    "Banana":       (200, 50, 300),
    "Coffee":       (100, 30, 100),
    "Tea":          (120, 30, 60),
    "Jute":         (60,  30, 30),
    "Tobacco":      (90,  60, 120),
}

# ── Soil nutrient retention modifiers (fraction of applied nutrient retained) ─
SOIL_MODIFIERS = {
    "Sandy":  {"N": 0.6, "P": 0.5, "K": 0.5},
    "Loamy":  {"N": 1.0, "P": 1.0, "K": 1.0},
    "Clay":   {"N": 1.1, "P": 1.2, "K": 1.3},
    "Silt":   {"N": 0.9, "P": 0.9, "K": 0.9},
    "Black":  {"N": 1.0, "P": 1.1, "K": 1.2},
    "Red":    {"N": 0.8, "P": 0.7, "K": 0.8},
}

SOIL_TYPES = list(SOIL_MODIFIERS.keys())
CROPS      = list(CROP_REQUIREMENTS.keys())

# ── Fertilizer decision rules ─────────────────────────────────────────────────
#   Each rule: (condition_fn, fertilizer, reason, application)
FERTILIZER_RULES = [
    {
        "fertilizer":   "Urea",
        "npk":          "46-0-0",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: n < 0.60 * n_opt and p >= 0.70 * p_opt and k >= 0.70 * k_opt,
        "reason":       "Soil nitrogen is critically low. Urea provides the highest nitrogen concentration (46%) to rapidly correct deficiency.",
        "application":  "Apply 100–120 kg/ha in two split doses: half at sowing, half at 30–35 DAS. Incorporate into moist soil to reduce volatilisation losses.",
    },
    {
        "fertilizer":   "DAP",
        "npk":          "18-46-0",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: p < 0.60 * p_opt and n >= 0.60 * n_opt and k >= 0.60 * k_opt,
        "reason":       "Phosphorus level is deficient. DAP (Di-Ammonium Phosphate) supplies high phosphorus (46%) plus nitrogen to support root development and energy transfer.",
        "application":  "Apply 80–100 kg/ha as basal dose, incorporated 5–10 cm below the seed level. Best applied before planting.",
    },
    {
        "fertilizer":   "MOP (Muriate of Potash)",
        "npk":          "0-0-60",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: k < 0.60 * k_opt and n >= 0.60 * n_opt and p >= 0.60 * p_opt,
        "reason":       "Potassium is deficient. MOP provides the highest potassium (60%) to improve drought tolerance, disease resistance, and fruit/grain quality.",
        "application":  "Apply 60–80 kg/ha — half as basal dose at sowing and half as top-dressing at flowering. Avoid applying during dry spells.",
    },
    {
        "fertilizer":   "NPK 17-17-17",
        "npk":          "17-17-17",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: n < 0.75 * n_opt and p < 0.75 * p_opt and k < 0.75 * k_opt,
        "reason":       "All three primary nutrients (N, P, K) are below optimal levels. Balanced NPK 17-17-17 corrects all deficiencies simultaneously.",
        "application":  "Apply 150–200 kg/ha as basal dose and repeat 50–75 kg/ha as top-dressing at 30 and 60 DAS. Suitable for broadcast and fertigation.",
    },
    {
        "fertilizer":   "NPK 20-20-0",
        "npk":          "20-20-0",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: n < 0.70 * n_opt and p < 0.70 * p_opt and k >= 0.80 * k_opt,
        "reason":       "Nitrogen and phosphorus are both low while potassium is adequate. NPK 20-20-0 targets the N and P deficiency without excess K.",
        "application":  "Apply 120–150 kg/ha at sowing as basal dose. Supplement with 30 kg/ha urea as top-dressing at tillering/knee-high stage.",
    },
    {
        "fertilizer":   "NPK 10-26-26",
        "npk":          "10-26-26",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: p < 0.65 * p_opt and k < 0.65 * k_opt and n >= 0.70 * n_opt,
        "reason":       "Phosphorus and potassium are both low while nitrogen is sufficient. NPK 10-26-26 corrects P and K without adding excess nitrogen.",
        "application":  "Apply 120–150 kg/ha as basal dose before planting. Ideal for fruiting crops and legumes in P- and K-deficient soils.",
    },
    {
        "fertilizer":   "SSP (Single Super Phosphate)",
        "npk":          "0-16-0",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: p < 0.50 * p_opt and n >= 0.80 * n_opt and k >= 0.80 * k_opt,
        "reason":       "Severe phosphorus deficiency. SSP provides phosphorus along with calcium and sulphur, beneficial for legumes and oilseeds.",
        "application":  "Apply 200–250 kg/ha as basal dose incorporated into the soil before sowing. Can be applied in furrows near the seed zone.",
    },
    {
        "fertilizer":   "Ammonium Sulphate",
        "npk":          "21-0-0",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: n < 0.65 * n_opt and p >= 0.80 * p_opt and k >= 0.80 * k_opt,
        "reason":       "Nitrogen is low. Ammonium Sulphate provides nitrogen plus sulphur — ideal for alkaline soils and sulphur-deficient conditions.",
        "application":  "Apply 100–150 kg/ha in two splits: at planting and 30–40 DAS. Particularly effective on alkaline/calcareous soils.",
    },
    {
        "fertilizer":   "NPK 12-32-16",
        "npk":          "12-32-16",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: p < 0.55 * p_opt and k < 0.75 * k_opt and n < 0.85 * n_opt,
        "reason":       "High phosphorus demand with moderate N and K deficiency. NPK 12-32-16 emphasises P correction while supplying balanced N and K.",
        "application":  "Apply 100–125 kg/ha as basal dose. Supplement with urea (30 kg/ha) at 30 DAS if nitrogen remains low.",
    },
    {
        "fertilizer":   "NPK 13-0-45",
        "npk":          "13-0-45",
        "condition":    lambda n, p, k, n_opt, p_opt, k_opt: k < 0.55 * k_opt and p >= 0.85 * p_opt and n < 0.80 * n_opt,
        "reason":       "High potassium demand with adequate phosphorus. NPK 13-0-45 is a premium K-source with supplementary nitrogen for fruiting and quality crops.",
        "application":  "Apply 100–120 kg/ha — 40% as basal and 60% as top-dressing at flowering and fruit development. Best via drip fertigation.",
    },
]

# ── Row generator ─────────────────────────────────────────────────────────────
def generate_row(crop, soil):
    n_opt, p_opt, k_opt = CROP_REQUIREMENTS[crop]
    mod = SOIL_MODIFIERS[soil]

    # Randomise actual NPK values around 0.3–1.2× optimal
    n_actual = round(np.random.uniform(0.25 * n_opt, 1.20 * n_opt))
    p_actual = round(np.random.uniform(0.25 * p_opt, 1.20 * p_opt))
    k_actual = round(np.random.uniform(0.25 * k_opt, 1.20 * k_opt))

    # Apply soil modifier (effective nutrient availability)
    n_eff = n_actual * mod["N"]
    p_eff = p_actual * mod["P"]
    k_eff = k_actual * mod["K"]

    # Match rules in order
    matched = None
    random.shuffle(FERTILIZER_RULES)          # shuffle to avoid bias
    for rule in FERTILIZER_RULES:
        if rule["condition"](n_eff, p_eff, k_eff, n_opt, p_opt, k_opt):
            matched = rule
            break

    if matched is None:
        # All nutrients adequate → use balanced fertilizer
        matched = {
            "fertilizer":  "NPK 17-17-17",
            "npk":         "17-17-17",
            "reason":      "Nutrients are near-optimal. Balanced NPK 17-17-17 maintains soil fertility for sustained yield.",
            "application": "Apply 100 kg/ha as basal dose. Monitor and top-dress only if deficiency symptoms appear.",
        }

    return {
        "Crop":        crop,
        "Soil_Type":   soil,
        "Nitrogen":    int(n_actual),
        "Phosphorus":  int(p_actual),
        "Potassium":   int(k_actual),
        "Fertilizer":  matched["fertilizer"],
        "NPK_Ratio":   matched["npk"],
        "Reason":      matched["reason"],
        "Application": matched["application"],
    }


def generate_dataset(n_rows: int = 3000) -> pd.DataFrame:
    rows = []
    per_combo = max(1, n_rows // (len(CROPS) * len(SOIL_TYPES)))
    for crop in CROPS:
        for soil in SOIL_TYPES:
            for _ in range(per_combo):
                rows.append(generate_row(crop, soil))
    # Top up to n_rows
    while len(rows) < n_rows:
        rows.append(generate_row(random.choice(CROPS), random.choice(SOIL_TYPES)))
    random.shuffle(rows)
    return pd.DataFrame(rows[:n_rows])


if __name__ == "__main__":
    print("Generating fertilizer dataset…")
    df = generate_dataset(3000)
    out = "Fertilizer_Recommendation.csv"
    df.to_csv(out, index=False)
    print(f"✅  Saved {len(df)} rows → {out}")
    print("\nFertilizer distribution:")
    print(df["Fertilizer"].value_counts().to_string())
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
