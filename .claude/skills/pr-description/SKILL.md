---
name: pr-description
description: Gera título e descrição em Markdown para Pull Requests deste projeto, no formato Conventional PR (Summary, Changes, Test plan, Related). Use ao abrir um PR ou quando o usuário pedir a descrição/mensagem do PR.
---

# Descrição de Pull Request

Gera o título e o corpo (Markdown) de um PR para o `medical-triage-mlops`, baseado nos
commits da branch atual em relação a `main`.

## Passo 1 — Coletar contexto

```bash
git branch --show-current
git log main..HEAD --oneline
git diff main...HEAD --stat
```

## Passo 2 — Identificar o(s) ID(s) de tarefa e checar o checklist do Kanban

Antes de montar qualquer coisa, use a skill `notion-kanban` para confirmar os cards que este
PR resolve:

1. Se o nome da branch seguir o padrão `<tipo>/<ID>-<slug>` (ex:
   `feat/TCF1-52-api-classificacao-laudo`), extraia o `<ID>` (ex: `TCF1-52`) como candidato.
2. Se a branch **não** tiver esse padrão, ou se o diff parecer tocar mais de um card,
   pergunte ao usuário (ou consulte o Kanban via `notion-kanban`) se há outros cards
   relacionados — não assuma que só existe um.
3. Para cada card candidato, rode o **checklist obrigatório pré-PR** da skill
   `notion-kanban`: só inclua no PR os cards que o trabalho da branch resolve **por
   completo**. Se algum card estiver apenas parcialmente resolvido, não o referencie no PR —
   avise o usuário e instrua a concluir o card primeiro.
4. Se nenhum card sobreviver ao checklist (branch não relacionada a nenhum card do Kanban),
   siga sem a seção `## Related`.

## Passo 3 — Montar o título

- Com ID confirmado: `[<ID>] <descrição curta no imperativo>` (se houver mais de um ID,
  `[<ID1>][<ID2>] <descrição>`)
- Sem ID: `<descrição curta no imperativo>`

A descrição curta resume o objetivo geral da branch (não é só o último commit). O ID no
título é só para legibilidade humana — **a integração com o Notion não lê o título**, lê a
descrição (Passo 4).

## Passo 4 — Montar a descrição (template)

```markdown
## Summary
- <1-3 bullets descrevendo o que foi feito e por quê>

## Changes
- <bullets das mudanças principais — módulos/arquivos afetados, decisões relevantes>

## Test plan
- [ ] <como validar — ex: `make check`, testes específicos, execução manual>

## Related
- Closes <ID>
```

Regras:
- A integração nativa Notion↔GitHub lê **magic words na descrição do PR**, não no título.
  Use um magic word por ID confirmado no checklist do Passo 2:
  - `Closes <ID>` (ou `Fixes`/`Resolves`/`Completes`) quando o card está 100% resolvido por
    este PR — o card é movido para concluído quando o PR for mergeado.
  - `Part of <ID>` (ou `Ref`/`Related to`/`Towards`) quando o PR só contribui para o card
    sem finalizá-lo — não fecha o card automaticamente.
  - Para múltiplos cards, uma linha por ID: `## Related` pode ter vários bullets, cada um
    com seu próprio magic word (ex: `Closes TCF1-52`, `Closes TCF1-53`).
- `## Related` só aparece se houver ao menos um ID confirmado no Passo 2. Se não houver,
  omita a seção inteira.
- `Summary` foca no "porquê"/objetivo; `Changes` foca no "o quê" (arquivos, módulos,
  decisões técnicas). Não repita a lista de commits literalmente — sintetize.
- `Test plan` deve ser um checklist verificável (comandos do `justfile`/`Makefile` deste
  projeto para lint/format/test, ou passos manuais quando aplicável).
- Linguagem: português, seguindo a convenção de commits do projeto.

## Passo 5 — Aprovação obrigatória antes de abrir o PR

**Em toda e qualquer hipótese**, mostrar o título e a descrição completos (já com a seção
`## Related` resolvida pelo checklist do Passo 2) ao usuário e esperar aprovação explícita
antes de rodar `gh pr create` ou qualquer comando que efetivamente abra o PR. Pedir para
abrir o PR autoriza o fluxo, não dispensa a revisão do conteúdo gerado — só prosseguir depois
que o usuário confirmar ou pedir ajustes no texto.
