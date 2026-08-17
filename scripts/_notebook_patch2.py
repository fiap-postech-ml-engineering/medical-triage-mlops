import json

path = "notebooks/01_eda_medical_abstracts.ipynb"
nb = json.load(open(path, encoding="utf-8"))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


new_cells = [
    md(
        [
            "## 10. Tuning de hiperparâmetros (TF-IDF + LogisticRegression)\n",
            "\n",
            "A `LogisticRegression (balanced)` da seção 9 usou os hiperparâmetros default do TF-IDF do projeto (`max_features=5000`, `ngram_range=(1,2)`, sem remoção de stopwords, sem `sublinear_tf`) e `C=1.0`. Nenhum desses foi ajustado ainda. Busca aleatória (`RandomizedSearchCV`, 3-fold, otimizando `f1_macro`) sobre o vetorizador + o classificador juntos, no mesmo `x_train`/`y_train`.",
        ]
    ),
    code(
        [
            "from scipy.stats import loguniform\n",
            "from sklearn.model_selection import RandomizedSearchCV\n",
            "\n",
            "tuning_pipeline = Pipeline(\n",
            "    steps=[\n",
            '        ("tfidf", TfidfVectorizer()),\n',
            '        ("clf", LogisticRegression(max_iter=2000, random_state=settings.RANDOM_SEED)),\n',
            "    ]\n",
            ")\n",
            "\n",
            "param_distributions = {\n",
            '    "tfidf__max_features": [5000, 10000, 20000, None],\n',
            '    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],\n',
            '    "tfidf__min_df": [1, 2, 3],\n',
            '    "tfidf__sublinear_tf": [True, False],\n',
            '    "tfidf__stop_words": [None, "english"],\n',
            '    "clf__C": loguniform(1e-2, 1e2),\n',
            '    "clf__class_weight": [None, "balanced"],\n',
            "}\n",
            "\n",
            "search = RandomizedSearchCV(\n",
            "    tuning_pipeline,\n",
            "    param_distributions=param_distributions,\n",
            "    n_iter=25,\n",
            "    cv=3,\n",
            '    scoring="f1_macro",\n',
            "    random_state=settings.RANDOM_SEED,\n",
            "    n_jobs=-1,\n",
            "    verbose=1,\n",
            ")\n",
            "search.fit(x_train, y_train)\n",
            "\n",
            'print("\\nMelhores parâmetros:", search.best_params_)\n',
            'print(f"Melhor f1_macro (CV, treino): {search.best_score_:.3f}")',
        ]
    ),
    code(
        [
            "tuned_preds = search.best_estimator_.predict(x_test)\n",
            'print(f"f1_macro (holdout): {f1_score(y_test, tuned_preds, average=\'macro\'):.3f}")\n',
            'print(f"accuracy (holdout): {accuracy_score(y_test, tuned_preds):.3f}")\n',
            "print()\n",
            "print(classification_report(y_test, tuned_preds))",
        ]
    ),
    md(
        [
            "**Se o resultado ajustado superar a `LogisticRegression (balanced)` da seção 9**, os `search.best_params_` acima viram os novos defaults de `TFIDF_MAX_FEATURES`/`RF_N_ESTIMATORS` (renomear para algo genérico) em `src/config.py` e de `build_pipeline` em `src/training/train.py` — junto com a troca de `RandomForestClassifier` para `LogisticRegression`, se você topar aplicar em produção."
        ]
    ),
]

insert_at = len(nb["cells"]) - 1
nb["cells"][insert_at:insert_at] = new_cells

json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("cells now:", len(nb["cells"]))
