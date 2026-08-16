"""Compara a latência de inferência: pipeline sklearn original vs. runtime ONNX.

Entregável da Etapa 4 (otimização de latência): mede p50/p95/p99 de N
inferências repetidas para os dois artefatos e imprime um resumo comparável.

Uso: `make latency-compare` (ou `uv run python -m src.training.latency_compare`).
Requer `models/classifier.joblib` (`make train`) e `models/classifier.onnx`
(`make onnx-export`) já existentes.
"""

import logging
import statistics
import time

import joblib
import numpy as np
import onnxruntime as ort

from src.config import BASE_DIR, get_settings

logger = logging.getLogger(__name__)

SAMPLE_TEXT = "Paciente relata dor torácica intensa com irradiação para o braço esquerdo."
N_RUNS = 200


def _percentile(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100)[int(p) - 1]


def _summary(label: str, latencies_ms: list[float]) -> None:
    logger.info(
        "%-12s p50=%.3fms  p95=%.3fms  p99=%.3fms  mean=%.3fms",
        label,
        _percentile(latencies_ms, 50),
        _percentile(latencies_ms, 95),
        _percentile(latencies_ms, 99),
        statistics.mean(latencies_ms),
    )


def benchmark_sklearn(model_path) -> list[float]:
    pipeline = joblib.load(model_path)
    pipeline.predict([SAMPLE_TEXT])  # warm-up

    latencies = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        pipeline.predict_proba([SAMPLE_TEXT])
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def benchmark_onnx(model_path) -> list[float]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    payload = np.array([[SAMPLE_TEXT]], dtype=object)
    session.run(None, {input_name: payload})  # warm-up

    latencies = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        session.run(None, {input_name: payload})
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def main() -> None:
    settings = get_settings()
    sklearn_path = BASE_DIR / settings.MODEL_PATH
    onnx_path = BASE_DIR / settings.ONNX_MODEL_PATH

    logger.info("Rodando %d inferências por artefato...", N_RUNS)
    sklearn_latencies = benchmark_sklearn(sklearn_path)
    onnx_latencies = benchmark_onnx(onnx_path)

    _summary("sklearn", sklearn_latencies)
    _summary("onnx", onnx_latencies)

    speedup = statistics.mean(sklearn_latencies) / statistics.mean(onnx_latencies)
    logger.info("Ganho médio de latência com ONNX: %.2fx", speedup)


if __name__ == "__main__":
    main()
