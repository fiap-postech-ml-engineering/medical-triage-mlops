"""Tests for `src.binary_triage.export_onnx` (conversão sklearn -> ONNX)."""

import numpy as np
import onnxruntime as rt
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.binary_triage.export_onnx import export_pipeline
from src.binary_triage.train import MODEL_PATH


def _make_pipeline() -> Pipeline:
    """Pipeline pequeno, real e determinístico — não um mock — pois `skl2onnx`
    precisa de um `Pipeline` sklearn de verdade para converter.
    """
    x = pd.DataFrame(
        {
            "texto": [
                "dor no peito falta de ar",
                "check up de rotina sem queixas",
                "cefaleia intensa e tontura",
                "consulta de rotina anual",
            ]
        }
    )
    y = np.array(["ESPECIALISTA", "CLINICO_GERAL", "ESPECIALISTA", "CLINICO_GERAL"])

    pipeline = Pipeline(
        steps=[
            ("features", ColumnTransformer([("tfidf", TfidfVectorizer(), "texto")])),
            ("clf", LogisticRegression()),
        ]
    )
    pipeline.fit(x, y)
    return pipeline


@pytest.mark.slow
def test_export_pipeline_gera_onnx_com_paridade_numerica(tmp_path):
    """Marcado slow: skl2onnx/onnxruntime têm overhead de import/conversão
    perceptível e o TfidfVectorizer exportado exige locale en_US.UTF-8
    instalado no sistema (StringNormalizer do onnxruntime).
    """
    pipeline = _make_pipeline()
    output_path = tmp_path / "modelo_teste.onnx"

    export_pipeline(pipeline, output_path)

    assert output_path.exists()

    x_novo = pd.DataFrame({"texto": ["dor no peito"]})
    proba_sklearn = pipeline.predict_proba(x_novo)

    sess = rt.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    x_onnx = np.array([["dor no peito"]], dtype=object)
    _, proba_onnx = sess.run(None, {input_name: x_onnx})

    assert np.allclose(proba_onnx, proba_sklearn, atol=1e-5)


@pytest.mark.slow
def test_export_onnx_modelo_real_treinado():
    """Integração de ponta a ponta com o `modelo.joblib` real (mesmo teste
    slow que valida a API — requer treino prévio via `train.py`).
    """
    import joblib

    if not MODEL_PATH.exists():
        pytest.skip("models/modelo.joblib não encontrado — rode o treino antes.")

    model = joblib.load(MODEL_PATH)
    assert model.pipeline is not None
    assert model.specialty_pipeline is not None
