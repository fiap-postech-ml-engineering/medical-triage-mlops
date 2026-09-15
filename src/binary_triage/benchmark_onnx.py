"""Compara latência de inferência: sklearn puro vs. sessão ONNX Runtime.

Mede p50/p95/p99 de uma única chamada por vez (caso de uso de uma API
servindo requisições uma a uma, não em lote), sobre o pipeline binário
(`model.pipeline` / `models/modelo.onnx`). Requer que `models/modelo.joblib`
e `models/modelo.onnx` já existam (`train.py` e `export_onnx.py`).

Uso: `uv run python -m src.binary_triage.benchmark_onnx` (a partir da raiz
do projeto).
"""

import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd

from src.binary_triage.export_onnx import ONNX_BINARY_PATH
from src.binary_triage.model_wrapper import ThresholdedBinaryClassifier
from src.binary_triage.train import MODEL_PATH, REPORTS_DIR

logger = logging.getLogger(__name__)

BENCHMARK_PATH = REPORTS_DIR / "onnx_benchmark.json"

WARMUP_ROUNDS = 20
N_ROUNDS = 300

SAMPLE_TEXTO = (
    "Patient presents with chest pain and shortness of breath, ECG shows "
    "ST elevation and elevated troponin."
)


def _percentiles(latencias_s: list[float]) -> dict:
    latencias_ms = np.array(latencias_s) * 1000
    return {
        "p50_ms": float(np.percentile(latencias_ms, 50)),
        "p95_ms": float(np.percentile(latencias_ms, 95)),
        "p99_ms": float(np.percentile(latencias_ms, 99)),
    }


def benchmark_sklearn(pipeline, x_df: pd.DataFrame, n_rounds: int) -> list[float]:
    latencias = []
    for _ in range(n_rounds):
        inicio = time.perf_counter()
        pipeline.predict_proba(x_df)
        latencias.append(time.perf_counter() - inicio)
    return latencias


def benchmark_onnx(sess: rt.InferenceSession, x_onnx: np.ndarray, n_rounds: int) -> list[float]:
    input_name = sess.get_inputs()[0].name
    latencias = []
    for _ in range(n_rounds):
        inicio = time.perf_counter()
        sess.run(None, {input_name: x_onnx})
        latencias.append(time.perf_counter() - inicio)
    return latencias


def main() -> None:
    model: ThresholdedBinaryClassifier = joblib.load(MODEL_PATH)
    pipeline = model.pipeline
    sess = rt.InferenceSession(str(ONNX_BINARY_PATH), providers=["CPUExecutionProvider"])

    x_df = pd.DataFrame({"texto": [SAMPLE_TEXTO]})
    x_onnx = np.array([[SAMPLE_TEXTO]], dtype=object)

    benchmark_sklearn(pipeline, x_df, WARMUP_ROUNDS)
    benchmark_onnx(sess, x_onnx, WARMUP_ROUNDS)

    latencias_sklearn = benchmark_sklearn(pipeline, x_df, N_ROUNDS)
    latencias_onnx = benchmark_onnx(sess, x_onnx, N_ROUNDS)

    resultado = {
        "n_rounds": N_ROUNDS,
        "sklearn": _percentiles(latencias_sklearn),
        "onnx": _percentiles(latencias_onnx),
    }
    resultado["speedup_p50"] = resultado["sklearn"]["p50_ms"] / resultado["onnx"]["p50_ms"]

    logger.info("Resultado do benchmark: %s", json.dumps(resultado, indent=2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(BENCHMARK_PATH).write_text(json.dumps(resultado, indent=2))
    logger.info(">>> Relatório salvo em %s", BENCHMARK_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
