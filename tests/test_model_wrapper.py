"""Tests for `ThresholdedBinaryClassifier`."""

import numpy as np

from src.binary_triage.model_wrapper import ThresholdedBinaryClassifier


class _FakeBinaryPipeline:
    """Pipeline fake que devolve probabilidades fixas para 3 linhas."""

    classes_ = np.array(["CLINICO_GERAL", "ESPECIALISTA"])

    def __init__(self):
        self.predict_proba_calls = 0

    def predict_proba(self, x):
        self.predict_proba_calls += 1
        p_pos = np.array([0.10, 0.55, 0.90])
        return np.column_stack([1.0 - p_pos, p_pos])


class _FakeSpecialtyPipeline:
    classes_ = np.array(["cardiologia", "oncologia"])

    def predict_proba(self, x):
        return np.array(
            [
                [0.5, 0.5],
                [0.3, 0.7],
                [0.2, 0.8],
            ]
        )


def _make_classifier(threshold=0.40, with_specialty=False):
    return ThresholdedBinaryClassifier(
        pipeline=_FakeBinaryPipeline(),
        threshold=threshold,
        specialty_pipeline=_FakeSpecialtyPipeline() if with_specialty else None,
    )


def test_predict_aplica_threshold_customizado():
    clf = _make_classifier(threshold=0.40)
    result = clf.predict(["a", "b", "c"])
    assert list(result) == ["CLINICO_GERAL", "ESPECIALISTA", "ESPECIALISTA"]


def test_predict_proba_soma_um_e_mapeia_classes():
    clf = _make_classifier()
    proba = clf.predict_proba(["a", "b", "c"])
    classes = list(clf.classes_)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.allclose(proba[:, classes.index("ESPECIALISTA")], [0.10, 0.55, 0.90])


def test_predict_detailed_consistente_com_predict_e_predict_proba():
    clf = _make_classifier(threshold=0.40, with_specialty=True)
    x = ["a", "b", "c"]

    esperado_classificacao = list(clf.predict(x))
    esperado_p_esp = clf.predict_proba(x)[:, list(clf.classes_).index("ESPECIALISTA")]

    detalhado = clf.predict_detailed(x)

    assert [row["classificacao"] for row in detalhado] == esperado_classificacao
    assert np.allclose([row["probabilidade_especialista"] for row in detalhado], esperado_p_esp)
    assert detalhado[0]["especialidade_provavel"] is None
    assert detalhado[1]["especialidade_provavel"] == "oncologia"
    assert detalhado[2]["especialidade_provavel"] == "oncologia"


def test_predict_detailed_chama_pipeline_predict_proba_uma_unica_vez():
    pipeline = _FakeBinaryPipeline()
    clf = ThresholdedBinaryClassifier(pipeline=pipeline, threshold=0.40)

    clf.predict_detailed(["a", "b", "c"])

    assert pipeline.predict_proba_calls == 1
