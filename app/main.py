"""Aplicación interactiva de diagnóstico de desgaste de herramienta CNC con Streamlit.

Permite a operarios e ingenieros de planta analizar la telemetría de mecanizado en tiempo real,
inferir el estado de desgaste (worn/unworn) con XGBoost y auditar la decisión mediante SHAP.
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
import pandas as pd
import shap
import streamlit as st

from src.features import extraer_features_segmento

# Configuración de página
st.set_page_config(
    page_title="CNC Tool Wear AI Diagnostic",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS modernos
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1a1c23;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .status-worn {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 24px;
    }
    .status-unworn {
        color: #00cc96;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_artefactos():
    """Carga el modelo XGBoost y las columnas de features."""
    models_dir = BASE_DIR / "models"
    model_path = models_dir / "cnc_tool_wear_xgb.pkl"
    cols_path = models_dir / "feature_columns.json"

    if not model_path.exists():
        st.error(
            f"Modelo no encontrado en {model_path}. Ejecuta primero src/models.py."
        )
        return None, None

    modelo = joblib.load(model_path)
    with open(cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return modelo, feature_cols


def main():
    st.title("⚙️ CNC Tool Wear Prediction & Explainability Dashboard")
    st.markdown(
        "Sistema de mantenimiento predictivo industrial basado en Machine Learning (XGBoost + SHAP) "
        "para la detección temprana de desgaste en fresadoras CNC."
    )

    modelo, feature_cols = cargar_artefactos()
    if modelo is None:
        return

    # Barra lateral
    st.sidebar.header("🕹️ Selección de Experimento")
    modo_seleccion = st.sidebar.radio(
        "Modo de entrada de datos:",
        ["Experimentos de referencia", "Subir archivo CSV"],
    )

    data_raw = BASE_DIR / "data" / "raw"
    data_sample = BASE_DIR / "data" / "sample"
    df_exp = None
    feedrate = 6.0
    clamp = 4.0

    if modo_seleccion == "Experimentos de referencia":
        muestras = {
            "Experimento 01 (Herramienta Nueva - Unworn)": (1, 6.0, 4.0, "unworn"),
            "Experimento 02 (Herramienta Nueva - Unworn)": (2, 20.0, 4.0, "unworn"),
            "Experimento 06 (Herramienta Desgastada - Worn)": (6, 6.0, 4.0, "worn"),
            "Experimento 18 (Herramienta Desgastada - Worn)": (18, 3.0, 2.5, "worn"),
        }
        eleccion = st.sidebar.selectbox(
            "Seleccionar serie temporal:", list(muestras.keys())
        )
        exp_id, feedrate, clamp, _expected_label = muestras[eleccion]
        file_path = data_sample / f"experiment_{exp_id:02d}.csv"
        if not file_path.exists():
            file_path = data_raw / f"experiment_{exp_id:02d}.csv"
        if file_path.exists():
            df_exp = pd.read_csv(file_path)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Subir CSV de telemetría de mecanizado", type=["csv"]
        )
        feedrate = st.sidebar.number_input(
            "Velocidad de avance (Feedrate mm/s)",
            min_value=1.0,
            max_value=50.0,
            value=6.0,
        )
        clamp = st.sidebar.number_input(
            "Presión de fijación (bar)", min_value=1.0, max_value=10.0, value=4.0
        )
        if uploaded_file is not None:
            df_exp = pd.read_csv(uploaded_file)
            exp_id = 99

    if df_exp is None:
        st.info(
            "Por favor, selecciona o sube un archivo CSV para iniciar el diagnóstico."
        )
        return

    # Fases de mecanizado activas
    if "Machining_Process" in df_exp.columns:
        df_corte = df_exp[df_exp["Machining_Process"].str.startswith("Layer", na=False)]
    else:
        df_corte = df_exp

    if len(df_corte) == 0:
        df_corte = df_exp

    # Segmentación en ventanas de 50 muestras y consenso predictivo
    window_size = 50
    step_size = 50
    probabilidades = []
    ventanas_feats = []

    for start_idx in range(0, len(df_corte) - window_size + 1, step_size):
        ventana = df_corte.iloc[start_idx : start_idx + window_size]
        wf = extraer_features_segmento(ventana, exp_id=exp_id, feedrate=feedrate, clamp=clamp, label=0)
        ventanas_feats.append(wf)
        X_win = pd.DataFrame([wf])[feature_cols].fillna(0.0)
        probabilidades.append(float(modelo.predict_proba(X_win)[0][1]))

    if probabilidades:
        prob_worn = float(np.mean(probabilidades))
        y_pred = 1 if prob_worn >= 0.50 else 0
        feats_dict = pd.DataFrame(ventanas_feats).mean().to_dict()
    else:
        feats_dict = extraer_features_segmento(
            df_corte, exp_id=exp_id, feedrate=feedrate, clamp=clamp, label=0
        )
        X_tmp = pd.DataFrame([feats_dict])[feature_cols].fillna(0.0)
        y_pred = int(modelo.predict(X_tmp)[0])
        prob_worn = float(modelo.predict_proba(X_tmp)[0][1])

    X_input = pd.DataFrame([feats_dict])[feature_cols].fillna(0.0)

    # Panel de métricas principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if y_pred == 1:
            st.metric(
                "Diagnóstico del Modelo",
                "WORN (Desgaste)",
                delta="- Reemplazo necesario",
                delta_color="inverse",
            )
        else:
            st.metric(
                "Diagnóstico del Modelo",
                "UNWORN (Afilada)",
                delta="Operación Segura",
                delta_color="normal",
            )
    with col2:
        st.metric("Probabilidad de Desgaste", f"{prob_worn * 100:.1f}%")
    with col3:
        st.metric(
            "Corriente Media Husillo (S1)",
            f"{df_corte['S1_OutputCurrent'].mean():.2f} A",
        )
    with col4:
        st.metric("Duración Total Muestreo", f"{len(df_exp) * 0.1:.1f} s")

    st.markdown("---")

    # Pestañas de análisis
    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Señales de Telemetría",
            "🔍 Explicación SHAP (XAI)",
            "ℹ️ Rigor y Metodología",
        ]
    )

    with tab1:
        st.subheader("Telemetría de los Servomotores y Husillo")
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = df_exp.index * 0.1

        # Corriente de salida del husillo S1
        axes[0].plot(
            t,
            df_exp["S1_OutputCurrent"],
            color="#2b5c8f",
            lw=1.2,
            label="Corriente Husillo S1",
        )
        axes[0].set_ylabel("S1 Current (A)")
        axes[0].set_title("Evolución Temporal de Carga del Husillo (S1 Output Current)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        # Aceleración del Eje Z
        axes[1].plot(
            t,
            df_exp["Z1_ActualAcceleration"],
            color="#d95f02",
            lw=1.0,
            alpha=0.8,
            label="Aceleración Eje Z",
        )
        axes[1].set_ylabel("Z Acceleration (mm/s²)")
        axes[1].set_xlabel("Tiempo transcurrido (s)")
        axes[1].set_title("Vibración Dinámica Axial en Eje Z")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Interpretabilidad Local (SHAP Waterfall)")
        st.write(
            "Desglose de cómo cada sensor físico específico de esta pieza contribuyó a desviar "
            "la predicción hacia desgaste (rojo / positivo) o hacia filo óptimo (azul / negativo)."
        )

        explainer = shap.TreeExplainer(modelo)
        shap_values_inst = explainer(X_input)

        fig_shap, _ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values_inst[0], max_display=10, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)

    with tab3:
        st.subheader("Garantías Metodológicas del Proyecto")
        st.markdown(
            """
            - **Prevención de Data Leakage:** El pipeline fue validado mediante **StratifiedGroupKFold**, 
              garantizando que ninguna serie temporal del mismo experimento se comparta entre entrenamiento y prueba.
            - **Aislamiento de Fases de Mecanizado:** Se filtran automáticamente las etapas de aproximación y retracción 
              (`Prep`, `Starting`, `Repositioning`), concentrando el análisis en los cortes activos (`Layer 1-3`).
            - **Arquitectura:** Modelo entrenado con XGBoost regularizado y auditable con valores Shapley aditivos (SHAP).
            """
        )


if __name__ == "__main__":
    main()
