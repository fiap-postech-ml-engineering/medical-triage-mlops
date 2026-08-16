"""Wrapper que embute o threshold de decisão dentro do artefato salvo.

`ThresholdedBinaryClassifier` guarda um Pipeline sklearn já treinado
(pré-processamento + vetorização + classificador) e o threshold de
probabilidade escolhido para a classe positiva (ESPECIALISTA). Isso permite
que `modelo.joblib` seja autossuficiente: `predict()` já aplica o threshold
calibrado em vez do padrão 0.5 do scikit-learn, sem o usuário do modelo
precisar saber qual threshold foi escolhido nem recriar lógica externa.

Opcionalmente carrega também `specialty_pipeline`: um classificador multiclasse
(oncologia/cardiologia/neurologia/gastroenterologia) treinado só nas linhas
ESPECIALISTA. A decisão binária não depende dele — ele só responde "para qual
especialidade, dado que não é Clínico Geral", via `predict_specialty` /
`predict_detailed`. Sem isso, o modelo dizia "precisa de especialista" sem
dizer qual, o que é pouco acionável numa triagem real.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdedBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, pipeline, threshold: float = 0.5, positive_label: str = "ESPECIALISTA",
                 negative_label: str = "CLINICO_GERAL", specialty_pipeline=None):
        self.pipeline = pipeline
        self.threshold = threshold
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.specialty_pipeline = specialty_pipeline

    @property
    def classes_(self):
        return np.array([self.negative_label, self.positive_label])

    def _positive_proba(self, X):
        proba = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.classes_)
        pos_idx = classes.index(self.positive_label)
        return proba[:, pos_idx]

    def predict_proba(self, X):
        p_pos = self._positive_proba(X)
        p_neg = 1.0 - p_pos
        neg_idx, pos_idx = list(self.classes_).index(self.negative_label), list(self.classes_).index(
            self.positive_label
        )
        out = np.zeros((len(p_pos), 2))
        out[:, pos_idx] = p_pos
        out[:, neg_idx] = p_neg
        return out

    def predict(self, X):
        p_pos = self._positive_proba(X)
        return np.where(p_pos >= self.threshold, self.positive_label, self.negative_label)

    def decision_function(self, X):
        return self._positive_proba(X)

    def predict_specialty_proba(self, X):
        """Probabilidade de cada especialidade (oncologia/cardiologia/...),
        independente da decisão binária. Só faz sentido interpretar quando
        `predict(X)` == ESPECIALISTA para a mesma linha."""
        if self.specialty_pipeline is None:
            raise ValueError("Este modelo não tem specialty_pipeline carregado.")
        return self.specialty_pipeline.predict_proba(X)

    @property
    def specialty_classes_(self):
        if self.specialty_pipeline is None:
            return None
        return self.specialty_pipeline.classes_

    def predict_detailed(self, X):
        """Retorna, por linha: classificação binária, probabilidades, e —
        quando classificado como ESPECIALISTA e o sub-modelo está disponível —
        a especialidade mais provável com a distribuição completa."""
        classificacao = self.predict(X)
        proba_bin = self.predict_proba(X)
        classes = list(self.classes_)
        p_esp = proba_bin[:, classes.index(self.positive_label)]
        p_cg = proba_bin[:, classes.index(self.negative_label)]

        specialty_proba, sp_classes = None, None
        if self.specialty_pipeline is not None:
            specialty_proba = self.specialty_pipeline.predict_proba(X)
            sp_classes = list(self.specialty_pipeline.classes_)

        results = []
        for i in range(len(classificacao)):
            row = {
                "classificacao": classificacao[i],
                "probabilidade_especialista": float(p_esp[i]),
                "probabilidade_clinico_geral": float(p_cg[i]),
                "especialidade_provavel": None,
                "probabilidades_especialidade": None,
            }
            if specialty_proba is not None:
                probs = {sp_classes[j]: float(specialty_proba[i, j]) for j in range(len(sp_classes))}
                row["probabilidades_especialidade"] = probs
                if classificacao[i] == self.positive_label:
                    row["especialidade_provavel"] = max(probs, key=probs.get)
            results.append(row)
        return results
