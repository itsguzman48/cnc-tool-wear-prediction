"""Script de verificación en vivo del modelo sobre los 18 experimentos reales.

Evalúa la precisión del modelo entrenado frente a la realidad física documentada
en el banco de pruebas de la Universidad de Michigan.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd

from src.features import extraer_features_segmento


def verificar_todos_los_experimentos():
    model_path = BASE_DIR / "models" / "cnc_tool_wear_xgb.pkl"
    cols_path = BASE_DIR / "models" / "feature_columns.json"
    train_path = BASE_DIR / "data" / "raw" / "train.csv"

    modelo = joblib.load(model_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    df_train = pd.read_csv(train_path)

    print("=" * 86)
    print(f"{'Exp':<4} | {'Estado Real':<12} | {'Predicción IA':<14} | {'Prob. Desgaste':<15} | {'Corriente S1':<14} | {'Resultado':<10}")
    print("=" * 86)

    aciertos = 0
    total = len(df_train)

    for _, row in df_train.iterrows():
        exp_id = int(row["No"])
        real_label = str(row["tool_condition"]).strip().lower()
        feedrate = float(row["feedrate"])
        clamp = float(row["clamp_pressure"])

        csv_file = BASE_DIR / "data" / "raw" / f"experiment_{exp_id:02d}.csv"
        if not csv_file.exists():
            continue

        df_exp = pd.read_csv(csv_file)

        # Filtrar pasadas de corte
        if "Machining_Process" in df_exp.columns:
            df_corte = df_exp[df_exp["Machining_Process"].str.startswith("Layer", na=False)]
        else:
            df_corte = df_exp

        if len(df_corte) == 0:
            df_corte = df_exp

        # Segmentar en ventanas de 50 muestras (5 segundos de corte) tal como fue entrenado el modelo
        window_size = 50
        step_size = 50
        probabilidades_ventanas = []

        for start_idx in range(0, len(df_corte) - window_size + 1, step_size):
            ventana = df_corte.iloc[start_idx : start_idx + window_size]
            feats = extraer_features_segmento(ventana, exp_id=exp_id, feedrate=feedrate, clamp=clamp, label=0)
            X_win = pd.DataFrame([feats])[feature_cols].fillna(0.0)
            p_worn = float(modelo.predict_proba(X_win)[0][1])
            probabilidades_ventanas.append(p_worn)

        if not probabilidades_ventanas:
            feats = extraer_features_segmento(df_corte, exp_id=exp_id, feedrate=feedrate, clamp=clamp, label=0)
            X_win = pd.DataFrame([feats])[feature_cols].fillna(0.0)
            probabilidades_ventanas.append(float(modelo.predict_proba(X_win)[0][1]))

        # Consenso de probabilidad de desgaste sobre todas las pasadas de corte
        prob_worn = float(np.mean(probabilidades_ventanas))
        pred_label = "worn" if prob_worn >= 0.50 else "unworn"

        es_acierto = (pred_label == real_label)
        if es_acierto:
            aciertos += 1

        marca = "CORRECTO [OK]" if es_acierto else "FALLO [X]"
        s1_curr = df_corte["S1_OutputCurrent"].mean()

        print(
            f"{exp_id:<4} | {real_label.upper():<12} | {pred_label.upper():<14} | {prob_worn * 100:>12.1f} % | {s1_curr:>11.2f} A | {marca} ({len(probabilidades_ventanas)} ventanas)"
        )

    print("=" * 86)
    print(f"Efectividad global: {aciertos}/{total} experimentos clasificados correctamente ({aciertos / total * 100:.1f}%)")
    print("=" * 86)


if __name__ == "__main__":
    verificar_todos_los_experimentos()
