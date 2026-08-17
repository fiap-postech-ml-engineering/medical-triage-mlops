"""Features estruturadas derivadas do texto do laudo.

O dataset (`data/raw/laudos.csv`) tem apenas duas colunas (`texto`,
`especialidade`) — não existem variáveis estruturadas independentes
(idade, sexo, duração, etc.). Para não descartar sinal e para atender ao
pedido de combinar texto + variáveis estruturadas, este módulo deriva um
pequeno conjunto de features numéricas A PARTIR do texto: tamanho da
observação e contagem de termos associados a cada especialidade.

As listas de termos abaixo vêm da EDA (`binary_triage/eda.py`, seção 6 —
top termos por especialidade) e não são um vocabulário inventado: são os
termos que mais diferenciam cada especialidade nesse corpus específico.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ONCOLOGY_TERMS = {
    "cancer", "tumor", "tumors", "tumour", "carcinoma", "malignant", "malignancy",
    "metastatic", "metastasis", "neoplasm", "neoplastic", "oncology", "chemotherapy",
    "biopsy", "sarcoma", "lymphoma", "leukemia", "melanoma",
}
CARDIO_TERMS = {
    "coronary", "cardiac", "heart", "myocardial", "ventricular", "artery", "arterial",
    "hypertension", "angina", "arrhythmia", "infarction", "atrial", "aortic",
    "cardiovascular", "cardiomyopathy", "systolic", "diastolic",
}
NEURO_TERMS = {
    "brain", "cerebral", "neurological", "neurologic", "stroke", "seizure", "epilepsy",
    "spinal", "nerve", "neuron", "cognitive", "dementia", "cerebrovascular", "paralysis",
    "parkinson", "meningitis",
}
GASTRO_TERMS = {
    "liver", "hepatic", "hepatitis", "gastric", "intestinal", "bowel", "colon",
    "colonic", "pancreatic", "cirrhosis", "esophageal", "duodenal", "colorectal",
    "gastrointestinal", "biliary",
}

SPECIALTY_KEYWORD_SETS = {
    "oncologia": ONCOLOGY_TERMS,
    "cardiologia": CARDIO_TERMS,
    "neurologia": NEURO_TERMS,
    "gastroenterologia": GASTRO_TERMS,
}

META_FEATURE_NAMES = [
    "n_chars",
    "n_words",
    "avg_word_len",
    "n_digits",
    "n_sentences",
    "kw_oncologia",
    "kw_cardiologia",
    "kw_neurologia",
    "kw_gastroenterologia",
    "kw_total_especialista",
    "kw_any_especialista",
]

_word_re = re.compile(r"[a-zA-Z]+")
_digit_re = re.compile(r"\d")


def _count_keywords(text_lower: str, vocab: set[str]) -> int:
    tokens = _word_re.findall(text_lower)
    return sum(1 for tok in tokens if tok in vocab)


def extract_meta_features(texts: pd.Series) -> pd.DataFrame:
    texts = texts.fillna("")
    rows = []
    for t in texts:
        t_lower = t.lower()
        words = _word_re.findall(t_lower)
        n_words = len(words)
        n_chars = len(t)
        avg_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
        n_digits = len(_digit_re.findall(t))
        n_sentences = max(t.count(".") + t.count("!") + t.count("?"), 1)

        kw_onco = _count_keywords(t_lower, ONCOLOGY_TERMS)
        kw_cardio = _count_keywords(t_lower, CARDIO_TERMS)
        kw_neuro = _count_keywords(t_lower, NEURO_TERMS)
        kw_gastro = _count_keywords(t_lower, GASTRO_TERMS)
        kw_total = kw_onco + kw_cardio + kw_neuro + kw_gastro

        rows.append(
            [
                n_chars,
                n_words,
                avg_word_len,
                n_digits,
                n_sentences,
                kw_onco,
                kw_cardio,
                kw_neuro,
                kw_gastro,
                kw_total,
                1 if kw_total > 0 else 0,
            ]
        )
    return pd.DataFrame(rows, columns=META_FEATURE_NAMES, index=texts.index)


class TextMetaFeatures(BaseEstimator, TransformerMixin):
    """Transformer sklearn-compatible: recebe a coluna de texto (Series/array 1D)
    e devolve a matriz de features estruturadas derivadas do texto."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        elif not isinstance(x, pd.Series):
            x = pd.Series(np.asarray(x).ravel())
        return extract_meta_features(x).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(META_FEATURE_NAMES)
