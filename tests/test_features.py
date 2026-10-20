"""Pruebas unitarias para el módulo de extracción de características y cálculo de métricas."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    calcular_rms,
    extraer_features_segmento,
    validar_telemetria_csv,
)


def test_calcular_rms_constante() -> None:
    """Verifica que el RMS de una señal constante coincide exactamente con su valor absoluto."""
    serie = pd.Series([5.0, 5.0, 5.0, 5.0])
    assert abs(calcular_rms(serie) - 5.0) < 1e-6


def test_calcular_rms_ceros() -> None:
    """Verifica que el RMS de una señal nula sea cero."""
    serie = pd.Series([0.0, 0.0, 0.0])
    assert calcular_rms(serie) == 0.0


def test_extraer_features_segmento_basico() -> None:
    """Verifica que se extraigan todas las métricas esperadas a partir de un segmento sintético."""
    datos = {
        "S1_OutputCurrent": [300.0, 310.0, 320.0, 305.0],
        "S1_OutputPower": [0.001, 0.002, 0.0015, 0.0018],
        "Z1_CurrentFeedback": [0.0, 1.2, 0.8, 0.5],
        "Z1_ActualAcceleration": [10.0, -10.0, 15.0, -12.0],
    }
    df_seg = pd.DataFrame(datos)
    feats = extraer_features_segmento(
        df_seg, exp_id=1, feedrate=6.0, clamp=4.0, label=0
    )

    # Validar metadatos obligatorios
    assert feats["exp_id"] == 1
    assert feats["feedrate"] == 6.0
    assert feats["clamp_pressure"] == 4.0
    assert feats["target"] == 0

    # Validar métricas extraídas
    assert "S1_OutputCurrent_mean" in feats
    assert "S1_OutputCurrent_rms" in feats
    assert "Z1_ActualAcceleration_std" in feats
    assert feats["S1_OutputCurrent_mean"] == pytest.approx(308.75)


def test_extraer_features_con_nulos() -> None:
    """Verifica que la presencia de valores nulos o columnas ausentes no lance excepciones."""
    datos = {
        "S1_OutputCurrent": [300.0, np.nan, 320.0, 310.0],
    }
    df_seg = pd.DataFrame(datos)
    feats = extraer_features_segmento(
        df_seg, exp_id=2, feedrate=20.0, clamp=3.0, label=1
    )

    assert feats["exp_id"] == 2
    assert "S1_OutputCurrent_mean" in feats
    assert not np.isnan(feats["S1_OutputCurrent_mean"])


def test_esquema_columnas_modelo_coherente() -> None:
    """Verifica que el esquema de columnas serializado en models sea consistente."""
    base_dir = Path(__file__).resolve().parent.parent
    cols_file = base_dir / "models" / "feature_columns.json"

    if cols_file.exists():
        with open(cols_file, "r", encoding="utf-8") as f:
            cols = json.load(f)
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert "feedrate" in cols
        assert "clamp_pressure" in cols


def test_validar_telemetria_csv_correcta() -> None:
    """Verifica que un conjunto de datos con las señales críticas sea aceptado."""
    df = pd.DataFrame(
        {
            "S1_OutputCurrent": [300.0, 305.0],
            "Z1_CurrentFeedback": [0.5, 0.6],
            "Z1_ActualAcceleration": [10.0, 12.0],
            "Machining_Process": ["Layer 1 Up", "Layer 1 Up"],
        }
    )
    valido, faltantes = validar_telemetria_csv(df)
    assert valido is True
    assert len(faltantes) == 0


def test_validar_telemetria_csv_columnas_faltantes() -> None:
    """Verifica que se detecten y reporten las columnas críticas ausentes."""
    df = pd.DataFrame({"X1_ActualPosition": [1.0, 2.0]})
    valido, faltantes = validar_telemetria_csv(df)
    assert valido is False
    assert "S1_OutputCurrent" in faltantes
    assert "Z1_CurrentFeedback" in faltantes
    assert "Z1_ActualAcceleration" in faltantes
