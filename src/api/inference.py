"""Carregamento do modelo treinado e execução da classificação."""

from pathlib import Path

import joblib
import pandas as pd

from src.api.metrics import triage_classifications_total, triage_inference_duration_seconds

# Necessário para o joblib conseguir desserializar as classes customizadas
# (ThresholdedBinaryClassifier / TextMetaFeatures) salvas dentro do pipeline.
from src.binary_triage.features import TextMetaFeatures  # noqa: F401
from src.binary_triage.model_wrapper import ThresholdedBinaryClassifier


def load_model(model_path: Path) -> ThresholdedBinaryClassifier:
    return joblib.load(model_path)


def classify(model: ThresholdedBinaryClassifier, texto: str) -> dict:
    x = pd.DataFrame({"texto": [texto]})
    with triage_inference_duration_seconds.time():
        resultado = model.predict_detailed(x)[0]
    resultado["threshold_usado"] = model.threshold
    triage_classifications_total.labels(classificacao=resultado["classificacao"]).inc()
    return resultado
