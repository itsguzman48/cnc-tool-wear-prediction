"""Módulo de modelado avanzado con XGBoost e interpretabilidad con SHAP.

Entrena un modelo regularizado sobre las features dinámicas de corte, evalúa con
StratifiedGroupKFold y genera explicaciones explicables de caja blanca con SHAP.
"""

import json
import sys
from pathlib import Path

# Asegurar raíz en sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.features import construir_dataset_segmentado


def entrenar_y_evaluar_xgboost(
    data_dir: Path, models_dir: Path, reports_dir: Path
) -> dict:
    """Entrena XGBoost con GroupKFold, guarda el artefacto de producción y genera explicaciones SHAP."""
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = construir_dataset_segmentado(data_dir)

    # Seleccionar features predictivas (excluyendo metadatos y target)
    feature_cols = [c for c in df.columns if c not in ("exp_id", "target")]
    X = df[feature_cols]
    y = df["target"]
    groups = df["exp_id"].values

    # Hiperparámetros de regularización para evitar memorizar patrones específicos de máquina
    params = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    }

    # 1. Validación cruzada agrupada
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_test, y_proba))
        except ValueError:
            aucs.append(0.5)

    print("=" * 60)
    print("XGBOOST - VALIDACIÓN CRUZADA ESTRATIFICADA POR GRUPO")
    print("=" * 60)
    print(f"Accuracy: {np.mean(accs):.3f} (+/- {np.std(accs):.3f})")
    print(f"F1-Score: {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")
    print(f"ROC-AUC:  {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")

    # 2. Entrenar modelo final de producción sobre todas las muestras
    modelo_final = xgb.XGBClassifier(**params)
    modelo_final.fit(X, y)

    # Guardar modelo y lista de features
    joblib.dump(modelo_final, models_dir / "cnc_tool_wear_xgb.pkl")
    with open(models_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nModelo final exportado en: {models_dir / 'cnc_tool_wear_xgb.pkl'}")

    # 3. Interpretabilidad Global con SHAP
    print("\nCalculando valores SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(modelo_final)
    shap_values = explainer(X)

    # Gráfico 1: Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=12, show=False)
    plt.title(
        "Importancia Global de Variables e Impacto en Desgaste (SHAP Beeswarm)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Gráfico 2: Bar Plot de Importancia Media
    plt.figure(figsize=(10, 5))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title(
        "Top 10 Señales más Influyentes en la Predicción (SHAP Mean |Values|)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Gráficos SHAP generados en: {figures_dir}")

    metricas_xgb = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "roc_auc_mean": float(np.mean(aucs)),
    }
    with open(reports_dir / "xgb_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas_xgb, f, indent=2)

    return metricas_xgb


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_p = base_dir / "data" / "raw"
    mod_p = base_dir / "models"
    rep_p = base_dir / "reports"
    entrenar_y_evaluar_xgboost(raw_p, mod_p, rep_p)
