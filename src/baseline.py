"""Módulo de evaluación y modelos baseline con GroupKFold.

Garantiza la ausencia de data leakage agrupando estrictamente las particiones
de validación cruzada por experimento (exp_id).
"""

import json
from pathlib import Path
import sys

# Asegurar raíz en sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import construir_dataset_segmentado


def evaluar_modelo_group_cv(modelo, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> dict:
    """Evalúa un estimador mediante StratifiedGroupKFold para garantizar balance de clases sin data leakage."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        # Probabilidades si el modelo lo soporta
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        aucs.append(auc)

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "roc_auc_mean": float(np.mean(aucs)),
    }


def ejecutar_evaluacion_baseline(data_dir: Path, reports_dir: Path) -> dict:
    """Ejecuta el experimento baseline y guarda el reporte comparativo."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = construir_dataset_segmentado(data_dir)

    feature_cols = [c for c in df.columns if c not in ("exp_id", "target")]
    X = df[feature_cols].values
    y = df["target"].values
    groups = df["exp_id"].values

    modelos = {
        "Dummy (Mayoritario)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Baseline Lineal)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest (Baseline Árboles)": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        ),
    }

    resultados = {}
    print("=" * 60)
    print("EVALUACIÓN BASELINE CON GROUP-K-FOLD (n_splits=5)")
    print("=" * 60)

    for nombre, modelo in modelos.items():
        res = evaluar_modelo_group_cv(modelo, X, y, groups, n_splits=5)
        resultados[nombre] = res
        print(f"\nModelo: {nombre}")
        print(f"  - Accuracy: {res['accuracy_mean']:.3f} (+/- {res['accuracy_std']:.3f})")
        print(f"  - F1-Score: {res['f1_mean']:.3f} (+/- {res['f1_std']:.3f})")
        print(f"  - ROC-AUC:  {res['roc_auc_mean']:.3f}")

    # Guardar métricas en JSON
    json_path = reports_dir / "baseline_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
    print(f"\nReporte baseline guardado en: {json_path}")
    return resultados


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "data" / "raw"
    rep_path = base_dir / "reports"
    ejecutar_evaluacion_baseline(raw_path, rep_path)
