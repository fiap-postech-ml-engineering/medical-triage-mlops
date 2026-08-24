---
name: notion-kanban
description: Consulta e atualiza o Kanban do projeto no Notion (pegar tarefa, marcar "Em andamento", checklist pré-PR). Use ao iniciar uma tarefa, ao conferir o status de um card, ou antes de abrir qualquer PR.
model: claude-haiku-4-5-20251001
---

# Kanban do projeto (Notion)

O Kanban de desenvolvimento vive no hub de conhecimento do time no Notion:

- Hub: `https://app.notion.com/p/guda-programming/FIAP-Machine-Learning-Engineering-31a59f34b75280478656cde68ea4c493`
- Dataset (Kanban) ID: `60f478d5-6e29-4639-89d9-702706b2bf54`
- Projeto (Propriedade): Tech Challenge Fase 3

## Pré-requisito: MCP do Notion

Esta skill depende das ferramentas `mcp__claude_ai_Notion__*` (busca, query de
database/data source, update de página). Se essas ferramentas não estiverem disponíveis na
sessão, **pare e instrua o usuário a instalar o MCP do Notion** antes de continuar — sem
ele, o acompanhamento do Kanban fica manual e sujeito a erro (card esquecido, status
desatualizado). Insista nisso sempre que a ausência do MCP bloquear o fluxo; não tente
contornar com suposições sobre o estado do Kanban.

## Ações cobertas

- **Pegar uma tarefa**: buscar no dataset as tarefas em "A fazer", listar para o usuário
  escolher, e ao escolher:
  1. Verificar o campo "Person" do card.
  2. Se já estiver atribuído a alguém, seguir o fluxo normalmente (mover para "Em
     andamento") sem perguntar nada — provavelmente a pessoa já se atribuiu antes de pedir
     para começar.
  3. Se estiver vazio, perguntar ao usuário quem está trabalhando na tarefa, listando como
     opções as pessoas disponíveis no campo "Person" do dataset (buscar a lista de usuários/
     opções do workspace via MCP, não adivinhar nomes). Atribuir a pessoa escolhida ao card
     antes de mover para "Em andamento".
  4. Só então mover o card para "Em andamento".
- **Consultar detalhes de uma tarefa pelo ID** (ex: `TCF1-52`): buscar a página
  correspondente no dataset e trazer escopo/critérios de aceite.
- **Apoio à skill `pr-description`**: quando a branch não tiver ID de tarefa no nome, ou
  quando o diff parecer tocar mais de um card, consultar o Kanban para identificar
  candidatos a card relacionado em vez de assumir.

## O que esta skill NÃO faz

Mover um card para "Pull Request" ou "Concluído" é responsabilidade da integração nativa
Notion↔GitHub (acionada pelos magic words na descrição do PR — ver skill `pr-description`).
Esta skill nunca deve mover esses dois status manualmente, para não conflitar com a
automação nativa.

## Checklist obrigatório antes de qualquer PR

Antes de abrir um PR (via skill `pr-description`), revisar todos os cards "Em andamento"
relacionados ao trabalho feito na branch atual:

1. Buscar no dataset os cards "Em andamento" que tenham relação com o diff/commits da
   branch.
2. Para cada card candidato, avaliar se o trabalho da branch o resolve **por completo**.
3. **Regra dura**: um card só pode ser referenciado no PR (com magic word `closes`/`fixes`)
   se estiver 100% finalizado. Se um card está apenas parcialmente resolvido pelo diff
   atual, **não** o incluir no PR — avisar o usuário explicitamente e instruir a concluir o
   card antes de abrir o PR.
4. Só depois desse checklist, prosseguir para a skill `pr-description` com a lista de IDs
   confirmados como concluídos.

Isso evita dois problemas simétricos: cards esquecidos (nunca referenciados em nenhum PR) e
cards fechados precocemente pela automação do Notion antes do trabalho estar de fato
completo.
