"""Script de inferência do classificador CLINICO_GERAL vs ESPECIALISTA.

Uso:
    python predict.py "Paciente relata dor persistente no joelho..."

Carrega `modelo.joblib` (Pipeline completo: TF-IDF + LogisticRegression,
já com o threshold calibrado embutido) e imprime a classificação e as
probabilidades de cada classe. Não é necessário recriar nenhum
pré-processamento manualmente — tudo está dentro do artefato.

IMPORTANTE: esta é uma ferramenta de apoio à triagem baseada em padrões de
texto observados no dataset de treino. Não substitui avaliação médica.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Necessário para o joblib conseguir desserializar as classes customizadas
# (ThresholdedBinaryClassifier / TextMetaFeatures) salvas dentro do pipeline.
from binary_triage.features import TextMetaFeatures  # noqa: F401,E402
from binary_triage.model_wrapper import ThresholdedBinaryClassifier  # noqa: F401,E402

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "modelo.joblib"


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_texto(model, texto: str) -> dict:
    x = pd.DataFrame({"texto": [texto]})
    detalhado = model.predict_detailed(x)[0]
    detalhado["threshold_usado"] = model.threshold
    return detalhado


def main():
    if len(sys.argv) < 2:
        print('Uso: python predict.py "texto da observação médica"')
        sys.exit(1)

    texto = sys.argv[1]
    model = load_model()
    resultado = predict_texto(model, texto)

    print(f"Classificação: {resultado['classificacao']}")
    print(f"Probabilidade Especialista: {resultado['probabilidade_especialista']:.2f}")
    print(f"Probabilidade Clínico Geral: {resultado['probabilidade_clinico_geral']:.2f}")
    print(f"(threshold de decisão usado: {resultado['threshold_usado']:.2f})")

    if resultado["especialidade_provavel"]:
        probs = resultado["probabilidades_especialidade"]
        print(f"\nEspecialidade provável: {resultado['especialidade_provavel']}")
        for esp, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            print(f"  {esp}: {p:.2f}")


if __name__ == "__main__":
    main()
