# Monitoramento e Observabilidade

## 🗺️ Visão geral

A API expõe métricas no formato Prometheus em `GET /metrics`, coletadas por
um Prometheus (`observability/prometheus.yml`) e visualizadas em um
dashboard Grafana provisionado automaticamente
(`observability/grafana/`). Ambos sobem via `docker-compose.yml`, junto
com a API.

| Métrica | Tipo | Labels | Descrição |
|---|---|---|---|
| `http_requests_total` | Counter | `method`, `path`, `status_code` | Total de requisições HTTP recebidas |
| `http_request_duration_seconds` | Histogram | `method`, `path` | Duração da requisição HTTP, incluindo overhead de rede/serialização |
| `triage_classifications_total` | Counter | `classificacao` | Total de laudos classificados, por classe (`ESPECIALISTA`/`CLINICO_GERAL`) |
| `triage_inference_duration_seconds` | Histogram | — | Duração isolada de `model.predict_detailed()`, sem o overhead HTTP |

## 🤔 O porquê da escolha dessas métricas

- `triage_classifications_total` permite observar a distribuição de triagens ao longo do tempo — uma mudança abrupta na proporção `ESPECIALISTA`/`CLINICO_GERAL` pode indicar deriva nos dados de entrada em relação ao dataset de treino.
- `triage_inference_duration_seconds` é medida separadamente de `http_request_duration_seconds` porque a segunda inclui overhead de rede/serialização Pydantic — isolar a latência pura da inferência é o que será comparado, na próxima branch, contra a versão ONNX do modelo.
- Instrumentação com `prometheus-client` puro (já dependência direta do projeto), em vez de uma lib de auto-instrumentação de terceiros, para ter controle total sobre nomes/labels das métricas de negócio.

## 🔁 Como subir a stack

```bash
docker compose up --build
```

- API: `http://localhost:8000` (`/health`, `/classify`, `/metrics`)
- Prometheus: `http://localhost:9090` (scrape configurado para `api:8000/metrics` a cada 15s)
- Grafana: `http://localhost:3000` (login `admin`/`admin`, ou acesso anônimo como Viewer) — dashboard "Medical Triage - Overview" já provisionado, com painéis de:
  - Classificações por classe (taxa/5m)
  - Latência de inferência (p50/p95)
  - Requests HTTP por status (taxa/5m)
  - Latência HTTP total (p50/p95)

## ⚠️ Limitações atuais

| Limitação | Impacto |
|---|---|
| Sem alertas configurados (Alertmanager) | Uma degradação de latência ou queda de recall não dispara notificação |
| Dashboard não versiona mudanças feitas manualmente na UI | Edições no Grafana em runtime não voltam para o JSON provisionado |
| Sem métrica de drift de dados de entrada | `triage_classifications_total` só sinaliza mudança de proporção, não mudança na distribuição do texto em si |
