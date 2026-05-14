# Nome do Projeto (Multi-Agent Workspace)

Workspace multi-root organizado em agentes especializados, ferramentas compartilhadas e infraestrutura.

## Estrutura

| Índice | Pasta                          | Descrição |
|--------|--------------------------------|-----------|
| 0      | **CORE-ROOT**                  | Raiz do projeto, configurações gerais e ambientes virtuais. |
| 1      | **SHARED**                     | Código e utilitários compartilhados entre todos os agentes. |
| 2      | **NISABA** (Orchestration)    | Agente de orquestração – coordena fluxos e decisões. |
| 3      | **THOTH** (Text)               | Agente de processamento de texto e NLP. |
| 4      | **EUTERPE** (Audio)            | Agente de processamento e análise de áudio. |
| 5      | **HERMES** (Semantics)         | Agente de análise semântica e ontologias. |
| 6      | **JANUS** (History)            | Agente de histórico, memória e versionamento de contexto. |
| 7      | **ERIS** (Audit)               | Agente de auditoria, logs e conformidade. |
| 8      | **TOOL-GLYPHAR** (OpenCV/Tesseract) | Ferramenta de OCR e visão computacional. |
| 9      | **TOOL-LYRA** (pyannote/Whisper)    | Ferramenta de diarização e transcrição de áudio. |
| 10     | **TOOL-NOMOS** (spaCy/LiLT)         | Ferramenta de parsing sintático e análise legal (NLP). |
| 11     | **INFRA-DOCKER** (Prefect/PostgreSQL/Cypher) | Configurações Docker, bancos e orquestração de pipelines. |
| 12     | **FLOWS**                      | Definições de fluxos (Prefect) para execução dos pipelines. |

## Configuração do Ambiente

1. Crie um ambiente virtual para cada agente dentro de `Venvs/` (ex.: `python -m venv Venvs/nisaba`).
2. Instale as dependências específicas de cada módulo.
3. Utilize o arquivo `.env` na raiz para variáveis de ambiente (chaves de API, conexões, etc.).
4. Abra o workspace com VS Code e selecione o interpretador correto para cada pasta (já configurado no `.code-workspace`).

## Convenções

- Código formatado com **Black**.
- Type checking básico ativado (`basic`).
- Importações resolvidas automaticamente a partir das pastas listadas em `python.analysis.extraPaths`.
- Ambientes virtuais excluídos da visualização do Explorer (`files.exclude`).
