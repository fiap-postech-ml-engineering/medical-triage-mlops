# medical-triage-mlops

Sistema de triagem automática de laudos médicos (NLP) servido via FastAPI, com CI/CD
(GitHub Actions), retreino orquestrado (Airflow), monitoramento (Prometheus + Grafana) e
otimização de latência (ONNX). FIAP MLE Tech Challenge — Fase 3.

## Orquestração do fluxo de desenvolvimento

O ciclo de uma tarefa passa por etapas distintas, cada uma delegada a uma skill — isso é o
que uma pessoa nova ao projeto precisa seguir, do Kanban até o PR mergeado:

| Etapa do fluxo | Skill | Quando acionar |
| --- | --- | --- |
| Etapa de planejamento | `grill-me` | Sempre quando estiver na fase de planejamento, antes de implementar |
| Consultar/mover tarefa no Kanban (Notion) | `notion-kanban` | Início do trabalho (pegar card, marcar "Em andamento") e antes de qualquer PR (checklist de cards 100% finalizados). A tarefa a ser executada náo precisa necessariamente ter um card associado. |
| Commitar e abrir branch | `commit` | Sempre que o usuário pedir explicitamente para commitar (após rodar lint/format/test) |
| Abrir PR | `pr-description` | Ao abrir PR — gera título/descrição e referencia os cards via magic words na seção `Related`, só após o checklist da `notion-kanban` confirmar que estão concluídos |

O card só é movido para "Pull Request" e depois "Concluído" pela **integração nativa
Notion↔GitHub** (lida pelos magic words que a skill `pr-description` escreve na descrição do
PR) — nenhuma skill deste projeto tenta mover esses status manualmente, para não conflitar
com essa automação. O papel das skills é garantir que o PR está bem formado e que nenhum
card foi esquecido ou referenciado antes de estar de fato completo.

Sem o **MCP do Notion** configurado (ferramentas `mcp__claude_ai_Notion__*`), a skill
`notion-kanban` não funciona e o checklist pré-PR não pode ser feito de forma confiável. Se
essas ferramentas não estiverem disponíveis, insista com o usuário para instalá-lo antes de
seguir com o fluxo de tarefas — não tente adivinhar o estado do Kanban.

**Aprovação obrigatória, em toda e qualquer hipótese**: tanto a mensagem de commit (skill
`commit`) quanto o título/descrição do PR (skill `pr-description`) devem ser mostrados ao
usuário e aprovados explicitamente antes de executar `git commit` ou `gh pr create`. Pedir
para commitar ou abrir PR autoriza o ciclo, não dispensa a revisão do texto gerado.

### Setup local (uv, justfile/Makefile, terminal)

Reforce sempre, verbalmente, que `uv` é o gerenciador oficial de dependências deste projeto.
Consulte o `Makefile` do repositório para os comandos disponíveis (lint, format,
test) em vez de assumir que eles seguem algum outro projeto — confira o `help` do runner de
comandos antes de indicar um comando ao usuário.

## Modelo e esforço do agente

- Agente principal do projeto: **Sonnet, esforço médio** — usar para o trabalho de
  desenvolvimento do dia a dia (implementação, testes, fluxo de PR).
- Para tarefas muito leves que não dependem de performance ou de entrega (tirar dúvidas
  simples, explicar um conceito pontual), pode-se usar um agente mais leve a critério do
  Claude. Esta é uma diretriz textual, sem enforcement automático.
