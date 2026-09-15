# Pipeline de Retreino (Airflow)

## 🗺️ Visão geral

O retreino do classificador de triagem é orquestrado por uma DAG do
**Apache Airflow** (`dags/triage_training_dag.py`), agendada para rodar
`@weekly`. A DAG reaproveita diretamente os módulos já usados no treino
manual (`src/binary_triage/dataset.py` e `src/binary_triage/train.py`) — não
há lógica duplicada entre o fluxo manual e o orquestrado.

| Task | Chama | O que faz |
|------|-------|-----------|
| `load_data` | `prepare_dataset(force=False)` | Garante que `data/raw/laudos.csv` existe, baixando do Kaggle via `kagglehub` só se ainda não existir |
| `train_and_save` | `train.main()` | Roda o pipeline completo (comparação de candidatos, tuning de threshold, refit, avaliação, interpretabilidade) e sobrescreve `models/modelo.joblib` |

## 🤔 O porquê da escolha dessa arquitetura

- **TaskFlow API** (`@dag`/`@task`, ver `airflow.sdk`) em vez de `PythonOperator` clássico — é o padrão recomendado pelo Airflow moderno (3.x) para pipelines Python puro como este, sem operators de sistemas externos, e mantém o código enxuto.
- **`load_data` com `force=False`** — o dataset de origem (Medical Abstracts TC Corpus) é estático; forçar novo download a cada execução agendada gastaria rede/tempo sem qualquer ganho, já que o conteúdo não muda entre execuções.
- **`train_and_save` chama `train.main()` inteiro**, sem uma versão "enxuta" separada só para retreino automatizado — o pipeline já é determinístico (`RANDOM_SEED` fixo) e a reavaliação de candidatos/threshold a cada retreino é o comportamento correto esperado de um retreino real (dados novos poderiam justificar um threshold diferente). Criar um segundo caminho de treino agora seria complexidade prematura sem um requisito real de performance da DAG.

## 🔁 Qual o funcionamento da DAG

1. `load_data` roda primeiro; se `data/raw/laudos.csv` já existir, não faz nada além de retornar o caminho.
2. `train_and_save` roda em seguida (dependência `load_data() >> train_and_save()`), sem receber o path via XCom explicitamente — ambas as tasks operam sobre o mesmo caminho padrão (`RAW_PATH`/`MODEL_PATH`) já definido nos módulos de origem.
3. Ao final, `models/modelo.joblib` é sobrescrito com o modelo recém-treinado, e o `ThresholdedBinaryClassifier` resultante já embute o novo threshold calibrado.

## ⚙️ Ambiente local de teste

- `AIRFLOW_HOME=.airflow/` — diretório local, fora do controle de versão (`.gitignore`), usado só para testar a DAG sem interferir no restante do repositório.
- Banco de metadados: SQLite local (`airflow db migrate`), suficiente para teste local — não representa o executor/banco usado em um deploy real do Airflow.
- Validação: `airflow dags test triage_training_dag` executa a DAG inteira de ponta a ponta (sem subir scheduler/webserver), rodando `load_data` e `train_and_save` em sequência.

## ⚠️ Limitações atuais

| Limitação | Impacto |
|---|---|
| Sem Airflow rodando de fato em produção (scheduler/webserver via Docker) | A DAG foi validada localmente via CLI (`airflow dags test`), não em um ambiente orquestrado real |
| `train_and_save` não notifica nem versiona o modelo anterior | Um retreino que piora métricas sobrescreve `modelo.joblib` sem rollback automático |
| Sem alertas de falha configurados | Uma falha na DAG não dispara notificação (e-mail/Slack) |
