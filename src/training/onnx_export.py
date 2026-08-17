"""Converte o pipeline sklearn treinado (TF-IDF + RandomForest) para ONNX.

Otimização de latência exigida pela Etapa 4 do Tech Challenge: o runtime ONNX
evita o overhead do dispatcher Python do scikit-learn na inferência.

Uso: `make onnx-export` (ou `uv run python -m src.training.onnx_export`).
Requer `models/classifier.joblib` já existente (`make train`).
"""

import json
import logging

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from src.config import BASE_DIR, get_settings

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()

    model_path = BASE_DIR / settings.MODEL_PATH
    logger.info("Carregando pipeline sklearn de %s", model_path)
    pipeline = joblib.load(model_path)

    initial_type = [("input", StringTensorType([None, 1]))]
    logger.info("Convertendo pipeline para ONNX...")
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=17,
        options={id(pipeline): {"zipmap": True}},
    )

    onnx_path = BASE_DIR / settings.ONNX_MODEL_PATH
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())
    logger.info("Modelo ONNX salvo em %s", onnx_path)

    labels_path = BASE_DIR / settings.ONNX_LABELS_PATH
    labels = list(pipeline.classes_)
    labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    logger.info("Classes salvas em %s: %s", labels_path, labels)


if __name__ == "__main__":
    main()
