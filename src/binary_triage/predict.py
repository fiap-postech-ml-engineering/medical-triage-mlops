"""CLI de inferência do classificador CLINICO_GERAL vs ESPECIALISTA.

Uso:
    uv run python -m src.binary_triage.predict "Paciente relata dor persistente..."

Carrega o `modelo.joblib` treinado (ThresholdedBinaryClassifier, já com o
threshold calibrado embutido) e imprime a classificação binária e, quando
aplicável, a especialidade provável. Não é necessário recriar nenhum
pré-processamento manualmente — tudo está dentro do artefato salvo.

IMPORTANTE: esta é uma ferramenta de apoio à triagem baseada em padrões de
texto observados no dataset de treino. Não substitui avaliação médica.
"""

from pathlib import Path

import joblib
import pandas as pd
from rich.console import Console
import typer

# Necessário para o joblib conseguir desserializar as classes customizadas
# (ThresholdedBinaryClassifier / TextMetaFeatures) salvas dentro do pipeline.
from src.binary_triage.features import TextMetaFeatures  # noqa: F401
from src.binary_triage.model_wrapper import ThresholdedBinaryClassifier  # noqa: F401

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "modelo.joblib"

app = typer.Typer(add_completion=False)
console = Console()


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_texto(model, texto: str) -> dict:
    x = pd.DataFrame({"texto": [texto]})
    detalhado = model.predict_detailed(x)[0]
    detalhado["threshold_usado"] = model.threshold
    return detalhado


@app.command()
def main(texto: str) -> None:
    """Classifica um laudo médico em CLINICO_GERAL ou ESPECIALISTA."""
    model = load_model()
    resultado = predict_texto(model, texto)

    console.print(f"Classificação: [bold]{resultado['classificacao']}[/bold]")
    console.print(f"Probabilidade Especialista: {resultado['probabilidade_especialista']:.2f}")
    console.print(f"Probabilidade Clínico Geral: {resultado['probabilidade_clinico_geral']:.2f}")
    console.print(f"(threshold de decisão usado: {resultado['threshold_usado']:.2f})")

    if resultado["especialidade_provavel"]:
        probs = resultado["probabilidades_especialidade"]
        console.print(
            f"\nEspecialidade provável: [bold]{resultado['especialidade_provavel']}[/bold]"
        )
        for esp, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            console.print(f"  {esp}: {p:.2f}")


if __name__ == "__main__":
    app()
