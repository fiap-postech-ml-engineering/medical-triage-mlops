"""Exporta os pipelines sklearn do modelo treinado para ONNX.

Converte `model.pipeline` (decisão CLINICO_GERAL/ESPECIALISTA) e
`model.specialty_pipeline` (qual especialidade) — ambos `Pipeline`
scikit-learn puro (TfidfVectorizer -> LogisticRegression) — para o formato
ONNX, usando `skl2onnx`.

Fica de fora do grafo ONNX: a lógica de threshold do
`ThresholdedBinaryClassifier.predict()` (threshold calibrado em vez do 0.5
padrão) não é exportável, pois não é uma operação nativa do classificador —
precisa ser reaplicada em Python puro sobre as probabilidades que a sessão
ONNX devolve.

Uso: `uv run python -m src.binary_triage.export_onnx` (a partir da raiz do
projeto, requer `models/modelo.joblib` já treinado).
"""

import logging
from pathlib import Path

import joblib
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

from src.binary_triage.model_wrapper import ThresholdedBinaryClassifier

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = OUT_DIR / "modelo.joblib"
ONNX_BINARY_PATH = OUT_DIR / "modelo.onnx"
ONNX_SPECIALTY_PATH = OUT_DIR / "specialty.onnx"

INITIAL_TYPE = [("texto", StringTensorType([None, 1]))]


def export_pipeline(pipeline, output_path: Path) -> Path:
    # Stub de skl2onnx declara `initial_types` mais estrito do que a API real aceita
    # (StringTensorType é o uso documentado/correto aqui, não um erro de tipo real).
    onx = to_onnx(pipeline, initial_types=INITIAL_TYPE, options={"zipmap": False})  # ty: ignore[invalid-argument-type]
    output_path.write_bytes(onx.SerializeToString())
    logger.info(">>> Pipeline exportado para %s", output_path)
    return output_path


def main() -> None:
    model: ThresholdedBinaryClassifier = joblib.load(MODEL_PATH)

    export_pipeline(model.pipeline, ONNX_BINARY_PATH)

    if model.specialty_pipeline is not None:
        export_pipeline(model.specialty_pipeline, ONNX_SPECIALTY_PATH)
    else:
        logger.warning("Modelo não tem specialty_pipeline — pulando exportação.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
