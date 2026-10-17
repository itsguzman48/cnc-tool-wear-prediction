"""Módulo de Análisis Exploratorio de Datos (EDA) para el desgaste de herramientas CNC.

Examina las 18 series temporales de la fresadora y la distribución de etiquetas
en train.csv para comprender las dinámicas de corte y el comportamiento de los motores.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuración estética consistente
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 120


def cargar_metadatos(data_dir: Path) -> pd.DataFrame:
    """Carga y procesa las etiquetas de los 18 experimentos."""
    df_train = pd.read_csv(data_dir / "train.csv")
    df_train["target"] = (df_train["tool_condition"] == "worn").astype(int)
    return df_train


def inspeccionar_series(data_dir: Path) -> pd.DataFrame:
    """Recorre los 18 CSVs y extrae métricas de duración y balance."""
    registros = []
    for exp_id in range(1, 19):
        file_path = data_dir / f"experiment_{exp_id:02d}.csv"
        if not file_path.exists():
            continue
        df_exp = pd.read_csv(file_path)
        registros.append({
            "No": exp_id,
            "filas": len(df_exp),
            "duracion_segundos": len(df_exp) * 0.1,  # Frecuencia de muestreo: 100 ms
            "fases_unicas": df_exp["Machining_Process"].nunique() if "Machining_Process" in df_exp.columns else 0,
            "corriente_husillo_media": df_exp["S1_OutputCurrent"].mean() if "S1_OutputCurrent" in df_exp.columns else np.nan,
            "potencia_husillo_media": df_exp["S1_OutputPower"].mean() if "S1_OutputPower" in df_exp.columns else np.nan,
        })
    return pd.DataFrame(registros)


def generar_graficos_eda(data_dir: Path, output_dir: Path) -> None:
    """Genera las visualizaciones clave del comportamiento físico de los motores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train = cargar_metadatos(data_dir)
    df_resumen = inspeccionar_series(data_dir)
    df_merged = df_train.merge(df_resumen, on="No")

    # 1. Gráfico de distribución de clases y parámetros operativos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.countplot(
        data=df_train,
        x="tool_condition",
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[0],
        hue="tool_condition",
        legend=False,
    )
    axes[0].set_title("Distribución de Clases (Tool Condition)")
    axes[0].set_xlabel("Condición de la Herramienta")
    axes[0].set_ylabel("Número de Experimentos")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2.0, p.get_height() / 2),
                         ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    sns.scatterplot(
        data=df_merged,
        x="feedrate",
        y="clamp_pressure",
        hue="tool_condition",
        palette=["#2b5c8f", "#d95f02"],
        s=120,
        ax=axes[1],
    )
    axes[1].set_title("Parámetros de Operación vs. Desgaste")
    axes[1].set_xlabel("Velocidad de Avance (Feedrate mm/s)")
    axes[1].set_ylabel("Presión de Fijación (bar)")
    plt.tight_layout()
    fig.savefig(output_dir / "distribucion_etiquetas_train.png", dpi=150)
    plt.close(fig)

    # 2. Serie temporal comparativa entre herramienta nueva (Exp 01) y desgastada (Exp 06)
    df_unworn = pd.read_csv(data_dir / "experiment_01.csv")
    df_worn = pd.read_csv(data_dir / "experiment_06.csv")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    # Husillo S1 Output Current
    axes[0].plot(df_unworn.index * 0.1, df_unworn["S1_OutputCurrent"], label="Exp 01 (Unworn)", color="#2b5c8f", alpha=0.85)
    axes[0].plot(df_worn.index * 0.1, df_worn["S1_OutputCurrent"], label="Exp 06 (Worn)", color="#d95f02", alpha=0.75)
    axes[0].set_ylabel("Corriente Husillo S1 (A)")
    axes[0].set_title("Comparativa de Consumo de Corriente del Husillo (S1) a lo largo del Mecanizado")
    axes[0].legend(loc="upper right")

    # Eje Z Actual Acceleration
    axes[1].plot(df_unworn.index * 0.1, df_unworn["Z1_ActualAcceleration"], label="Exp 01 (Unworn)", color="#2b5c8f", alpha=0.7)
    axes[1].plot(df_worn.index * 0.1, df_worn["Z1_ActualAcceleration"], label="Exp 06 (Worn)", color="#d95f02", alpha=0.7)
    axes[1].set_ylabel("Aceleración Eje Z (mm/s²)")
    axes[1].set_xlabel("Tiempo transcurrido (segundos)")
    axes[1].set_title("Respuesta Vibracional y Dinámica del Eje Z (Fuerza de Empuje Vertical)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "comparativa_senales_husillo.png", dpi=150)
    plt.close(fig)

    # 3. Consumo medio por fase de mecanizado (Filtrado de fases de corte)
    df_worn_cut = df_worn[df_worn["Machining_Process"].str.startswith("Layer", na=False)]
    df_unworn_cut = df_unworn[df_unworn["Machining_Process"].str.startswith("Layer", na=False)]

    fases_unworn = df_unworn_cut.groupby("Machining_Process")["S1_OutputCurrent"].mean()
    fases_worn = df_worn_cut.groupby("Machining_Process")["S1_OutputCurrent"].mean()
    fases_df = pd.DataFrame({"Nueva (Unworn)": fases_unworn, "Desgastada (Worn)": fases_worn}).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    fases_df.plot(kind="bar", ax=ax, color=["#2b5c8f", "#d95f02"])
    ax.set_title("Corriente Media del Husillo en Fases Activas de Corte (Layers)")
    ax.set_ylabel("Media S1_OutputCurrent (A)")
    ax.set_xlabel("Fase de Corte")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "perfil_fases_mecanizado.png", dpi=150)
    plt.close(fig)

    print("Visualizaciones generadas exitosamente en reports/figures/")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "data" / "raw"
    figs_path = base_dir / "reports" / "figures"
    generar_graficos_eda(raw_path, figs_path)
