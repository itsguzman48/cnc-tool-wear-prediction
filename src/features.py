"""Módulo de ingeniería de características (Feature Engineering).

Extrae representaciones estadísticas en el dominio del tiempo a partir de las
señales multivariables de la fresadora CNC, segmentando por ventanas temporales
y manteniendo la trazabilidad del experimento para evitar data leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import iqr, skew


def calcular_rms(serie: pd.Series) -> float:
    """Calcula el valor cuadrático medio (Root Mean Square) de una señal."""
    return float(np.sqrt(np.mean(np.square(serie))))


def validar_telemetria_csv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Comprueba que el conjunto de datos contenga las señales de sensores mínimas requeridas.

    Retorna una tupla (es_valido, columnas_faltantes).
    """
    columnas_requeridas = [
        "S1_OutputCurrent",
        "Z1_CurrentFeedback",
        "Z1_ActualAcceleration",
    ]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    return (len(faltantes) == 0, faltantes)


def extraer_features_segmento(
    segmento: pd.DataFrame, exp_id: int, feedrate: float, clamp: float, label: int
) -> dict:
    """Calcula estadísticas temporales y dinámicas para una ventana o fase de corte."""
    feats = {
        "exp_id": exp_id,
        "feedrate": feedrate,
        "clamp_pressure": clamp,
        "target": label,
    }

    # Señales clave del husillo y servomotores
    senales = [
        "S1_OutputCurrent",
        "S1_OutputPower",
        "S1_CurrentFeedback",
        "Z1_CurrentFeedback",
        "Z1_ActualAcceleration",
        "X1_CurrentFeedback",
        "X1_ActualAcceleration",
        "Y1_CurrentFeedback",
        "Y1_ActualAcceleration",
    ]

    for col in senales:
        if col in segmento.columns:
            s = segmento[col].dropna()
            if len(s) == 0:
                continue
            feats[f"{col}_mean"] = float(s.mean())
            feats[f"{col}_std"] = float(s.std() if len(s) > 1 else 0.0)
            feats[f"{col}_max"] = float(s.max())
            feats[f"{col}_min"] = float(s.min())
            feats[f"{col}_rms"] = calcular_rms(s)
            feats[f"{col}_iqr"] = float(iqr(s)) if len(s) > 1 else 0.0
            feats[f"{col}_skew"] = float(skew(s)) if len(s) > 2 else 0.0

    return feats


def construir_dataset_segmentado(
    data_dir: Path, window_size: int = 50, step_size: int = 25
) -> pd.DataFrame:
    """Procesa los 18 experimentos extrayendo ventanas solapadas durante las fases de corte.

    Filtrar las fases de mecanizado activas ('Layer') descarta las operaciones en vacío
    (posicionamiento y retracción), centrando el aprendizaje en el esfuerzo de corte real.
    """
    df_train = pd.read_csv(data_dir / "train.csv")
    label_map = {"unworn": 0, "worn": 1}
    df_train["target"] = df_train["tool_condition"].map(label_map)

    filas = []

    for _, row in df_train.iterrows():
        exp_id = int(row["No"])
        feedrate = float(row["feedrate"])
        clamp = float(row["clamp_pressure"])
        label = int(row["target"])

        file_path = data_dir / f"experiment_{exp_id:02d}.csv"
        if not file_path.exists():
            continue

        df_exp = pd.read_csv(file_path)

        # Aislamiento de pasadas de corte
        if "Machining_Process" in df_exp.columns:
            df_corte = df_exp[
                df_exp["Machining_Process"].str.startswith("Layer", na=False)
            ]
        else:
            df_corte = df_exp

        if len(df_corte) < window_size:
            df_corte = df_exp

        # Segmentación por ventanas deslizantes
        for start_idx in range(0, len(df_corte) - window_size + 1, step_size):
            ventana = df_corte.iloc[start_idx : start_idx + window_size]
            features = extraer_features_segmento(
                ventana, exp_id, feedrate, clamp, label
            )
            filas.append(features)

    df_dataset = pd.DataFrame(filas).fillna(0.0)
    return df_dataset


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    df = construir_dataset_segmentado(raw_dir)
    print(
        f"Dataset generado exitosamente: {df.shape[0]} muestras, {df.shape[1]} columnas."
    )
    print("Muestras por condición (target):")
    print(df["target"].value_counts())
