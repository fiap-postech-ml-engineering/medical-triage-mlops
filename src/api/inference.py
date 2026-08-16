"""Isola o carregamento do classificador de especialidade e a lógica de servir triagem.

Quando o artefato do modelo ainda não existe em disco (`make train` / `make
onnx-export` não rodou), o serviço opera em modo degradado: a API sobe
normalmente, o healthcheck responde, mas /classify retorna 503 com uma
mensagem clara em vez de derrubar a aplicação inteira.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.config import BASE_DIR, get_settings

logger = logging.getLogger(__name__)


class TriageService:
    """Carrega o classificador (sklearn ou ONNX) e prevê a especialidade recomendada."""

    def __init__(self) -> None:
        self._sklearn_model = None
        self._onnx_session = None
        self._onnx_labels: list[str] | None = None
        self._use_onnx: bool = False

    def load(self) -> None:
        """Tenta carregar o modelo configurado. Nunca levanta exceção — loga o
        motivo da falha e deixa o serviço em modo degradado."""
        settings = get_settings()
        self._use_onnx = settings.USE_ONNX

        try:
            if self._use_onnx:
                self._load_onnx(settings)
            else:
                self._load_sklearn(settings)
        except Exception:
            self._sklearn_model = None
            self._onnx_session = None
            logger.exception(
                "Modelo indisponível (USE_ONNX=%s) — API sobe em modo degradado, "
                "/classify retornará 503 até `make train`%s ser executado.",
                self._use_onnx,
                " / `make onnx-export`" if self._use_onnx else "",
            )

    def _load_sklearn(self, settings) -> None:
        import joblib

        model_path = BASE_DIR / settings.MODEL_PATH
        self._sklearn_model = joblib.load(model_path)
        logger.info("Pipeline sklearn carregado de %s", model_path)

    def _load_onnx(self, settings) -> None:
        import onnxruntime as ort

        model_path = BASE_DIR / settings.ONNX_MODEL_PATH
        labels_path = BASE_DIR / settings.ONNX_LABELS_PATH

        self._onnx_session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._onnx_labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        logger.info("Sessão ONNX carregada de %s", model_path)

    @property
    def is_ready(self) -> bool:
        if self._use_onnx:
            return self._onnx_session is not None
        return self._sklearn_model is not None

    def classify(self, texto: str) -> tuple[str, float, dict[str, float]]:
        """Retorna (especialidade_recomendada, confianca, probabilidades_por_classe)."""
        if not self.is_ready:
            raise RuntimeError("Modelo ainda não disponível — rode `make train`.")

        if self._use_onnx:
            return self._classify_onnx(texto)
        return self._classify_sklearn(texto)

    def _classify_sklearn(self, texto: str) -> tuple[str, float, dict[str, float]]:
        model = self._sklearn_model
        proba = model.predict_proba([texto])[0]
        classes = model.classes_
        probabilidades = {str(c): float(p) for c, p in zip(classes, proba, strict=True)}
        idx = int(np.argmax(proba))
        return str(classes[idx]), float(proba[idx]), probabilidades

    def _classify_onnx(self, texto: str) -> tuple[str, float, dict[str, float]]:
        session = self._onnx_session
        input_name = session.get_inputs()[0].name
        # TfidfVectorizer convertido via skl2onnx espera shape [N, 1] de strings.
        outputs = session.run(None, {input_name: np.array([[texto]], dtype=object)})

        # Conversores skl2onnx para classificadores produzem 2 saídas:
        # [0] rótulo previsto, [1] lista de dicts {classe: probabilidade} (zipmap).
        label = str(outputs[0][0])
        probabilidades_raw = outputs[1][0]
        probabilidades = {str(k): float(v) for k, v in probabilidades_raw.items()}
        confianca = probabilidades.get(label, max(probabilidades.values()))
        return label, confianca, probabilidades


triage_service = TriageService()
