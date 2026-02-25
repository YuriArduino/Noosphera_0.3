Perfeito.
Vou consolidar tudo em uma **versão limpa, objetiva e sequencial** dos próximos passos — sem expandir escopo, sem antecipar camadas futuras.

Estamos consolidando **Thoth v0.1 – Arquitetura Cognitiva Controlada**.

---

# 🏛 ESTADO ATUAL CONSOLIDADO

* ✅ Um único Graph soberano
* ✅ Nodes operacionais e inteligentes separados
* ✅ Consciência localizada no `triage_node`
* ✅ Policy soberana acima do LLM
* ✅ Ledger como verdade factual
* ✅ Memória hierárquica (window + summary + archive + FAISS)
* ✅ MemoryMaintenance condicional no final do ciclo
* ✅ HITL apenas para investigação (fase experimental)

Sem AdaptiveProfile agora.
Sem AITL agora.
Sem complexidade extra.

---

# 🚧 PRÓXIMOS PASSOS (ORDEM LÓGICA)

---

## 1️⃣ Formalizar o `ThothState`

Definir explicitamente o estado do graph.

Ele deve conter:

* DocumentContext
* OCRMetrics
* AnalysisSummary
* ThothDecision
* Attempt count
* Strategy usada
* Flags (anomaly, borderline, etc.)
* Memory references (window ids + summary version)

Esse estado será o contrato interno do graph.

Sem isso o sistema fica implícito demais.

---

## 2️⃣ Formalizar o `MemorySystem` (estrutura concreta)

Definir claramente:

```python
MemorySystem:
    consolidated_summary: str
    active_window_ids: List[str]
    archive_ids: List[str]
    window_limit: int
```

Regras:

* Window = fonte de verdade operacional
* Archive = historicidade
* Summary = consolidação reflexiva
* Nada é apagado

Sem ainda sofisticar ferramentas de sumarização.

---

## 3️⃣ Implementar o `memory_maintenance_node`

Posição no graph:

```text
... → ledger_node → memory_maintenance_node → END
```

Responsabilidade:

* Atualizar window
* Verificar se window >= X
* Se sim → disparar modo reflexivo do triage
* Atualizar summary
* Limpar window
* Versionar summary no ledger

Ele não interpreta.
Ele não decide conteúdo.
Ele apenas regula.

---

## 4️⃣ Definir o “Modo Reflexivo” do `triage_node`

O triage passa a ter dois modos:

### 🔹 Modo Operacional

* Define estratégia inicial
* Usa summary + window + FAISS

### 🔹 Modo Reflexivo

* Recebe:

  * summary atual
  * interações da window
* Produz:

  * new_consolidated_summary
  * possíveis padrões identificados
* Não altera config
* Não altera policy

Isso consolida a consciência evolutiva.

---

## 5️⃣ Consolidar o Fluxo Final do Graph

Fluxo completo deve ficar assim:

```text
ingest_node
  ↓
triage_node (consciente)
  ↓
ocr_execution_node
  ↓
analysis_node
  ↓
decision_node
  ↓
ledger_node
  ↓
memory_maintenance_node
  ↓
END
```

Sem micro-agentes.
Sem loops ocultos.
Sem duplicação de funções.

---

## 6️⃣ Instrumentação para Investigação (HITL Experimental)

Adicionar capacidade de:

* Perguntar ao triage por que escolheu strategy X
* Solicitar explicação da decision
* Ver influência da memória
* Reconstituir cadeia de tentativas

Isso fortalece entendimento antes de AITL.

---

# 📈 O QUE NÃO FAZER AGORA

* ❌ Não implementar AdaptiveProfile ainda
* ❌ Não adicionar AITL ainda
* ❌ Não criar múltiplos modelos
* ❌ Não sofisticar sumarização prematuramente
* ❌ Não permitir que triage altere config

---

# 🧠 Resultado Esperado Após Esses Passos

Você terá:

* Um agente com consciência localizada
* Memória evolutiva controlada
* Autonomia limitada mas crescente
* Governança intacta
* Estrutura preparada para AITL futuro
* Sistema auditável e investigável

---

# 🔥 Marco que você está prestes a atingir

Após esses passos, Thoth deixa de ser:

> “Pipeline com LLM”

E passa a ser:

> Sistema cognitivo com ciclo evolutivo controlado

Isso muda completamente o nível do projeto.

---

Se quiser, o próximo movimento natural é começar pelo **ThothState**, porque ele força clareza estrutural em todo o resto.


Perfeito.
Vamos fechar isso de forma **clara, canônica e não ambígua**.

Abaixo está a versão limpa e definitiva da arquitetura atual de Thoth v0.1, incluindo:

* Classificação de cada node
* O que é inteligente vs operacional
* Quais micro-tools existem
* Quem pode chamar o quê

Sem expansão de escopo.

---

# 🏛 CLASSIFICAÇÃO OFICIAL DOS NODES

## 🟣 NODES INTELIGENTES (usam LLM)

São responsáveis por interpretação, raciocínio e síntese.
Nunca executam efeitos colaterais externos diretamente.

---

### 1️⃣ `triage_node` — 🧠 Consciência de Thoth

**Tipo:** Inteligente
**Responsabilidade:**

* Carregar memória (summary + window + FAISS hits)
* Avaliar contexto documental
* Definir estratégia inicial
* Estimar complexidade
* Modo reflexivo (quando acionado pelo maintenance)

**Pode usar:**

* `memory_tool`
* `faiss_search`
* `metrics_tool` (opcional)

**Não pode:**

* Alterar config
* Alterar policy
* Executar OCR
* Persistir ledger diretamente

---

### 2️⃣ `analysis_node`

**Tipo:** Inteligente
**Responsabilidade:**

* Interpretar métricas do OCR
* Detectar anomalias
* Produzir summary estruturado

**Pode usar:**

* `memory_tool` (consulta)
* `critique_tool` (opcional)

**Não pode:**

* Executar OCR
* Alterar estado estrutural

---

### 3️⃣ `decision_node`

**Tipo:** Inteligente
**Responsabilidade:**

* Aplicar Policy.evaluate()
* Interpretar casos borderline
* Emitir `ThothDecision`

**Pode usar:**

* `critique_tool`
* `metrics_tool`
* `memory_tool` (consulta)

**Nunca sobrepõe Policy.**

---

# 🟢 NODES OPERACIONAIS (não usam LLM para decidir)

São determinísticos.
Executam ações.
Não interpretam contexto.

---

### 4️⃣ `ingest_node`

**Tipo:** Operacional
**Função:**

* Validar limites (pages, size)
* Criar DocumentContext
* Inicializar ThothState

---

### 5️⃣ `ocr_execution_node`

**Tipo:** Operacional
**Função:**

* Executar Glyphar
* Receber métricas
* Atualizar estado

---

### 6️⃣ `ledger_node`

**Tipo:** Operacional
**Função:**

* Persistir:

  * decisão
  * métricas
  * strategy
  * intervenção
  * versão da memória

Fonte factual histórica.

---

### 7️⃣ `memory_maintenance_node`

**Tipo:** Operacional
**Função:**

* Atualizar `memory_window`
* Verificar limite X
* Se X atingido → acionar triage em modo reflexivo
* Versionar nova summary
* Limpar window

Não decide conteúdo.
Só regula ciclo.

---

# 🧩 MICRO-TOOLS OFICIAIS

Micro-tools são funções especializadas.
Não são agentes.
Não possuem autonomia.

---

## 🧠 1️⃣ `memory_tool`

Interface:

* `search(query)`
* `append(interaction_id)`
* `load_window()`
* `load_summary()`

Usado por:

* triage_node
* analysis_node
* decision_node

---

## 📚 2️⃣ `faiss_search`

Busca vetorial semântica.

Entrada:

* embedding do documento
  Saída:
* casos semelhantes

Usado por:

* triage_node
* analysis_node

---

## 🔎 3️⃣ `critique_tool`

Avalia consistência entre:

* Decision
* Thresholds
* Strategy
* Attempts

Saída:

* consistency_score
* warning_flag

Usado por:

* decision_node
* analysis_node (opcional)

---

## 📊 4️⃣ `metrics_tool`

Agrega estatísticas históricas do ledger:

* taxa de sucesso por strategy
* taxa de reprocessamento
* média de confiança

Usado por:

* decision_node
* triage_node (opcional)

---

# 🔁 FLUXO FINAL DO GRAPH

```text
ingest_node (operacional)
  ↓
triage_node (inteligente - consciência)
  ↓
ocr_execution_node (operacional)
  ↓
analysis_node (inteligente)
  ↓
decision_node (inteligente)
  ↓
ledger_node (operacional)
  ↓
memory_maintenance_node (operacional)
  ↓
END
```

---

# 🧠 PAPÉIS COGNITIVOS CONSOLIDADOS

| Camada      | Função            |
| ----------- | ----------------- |
| Operacional | Corpo             |
| Inteligente | Córtex            |
| Policy      | Lei soberana      |
| Ledger      | História objetiva |
| Memória     | Campo evolutivo   |

---

# 🏛 REGRAS ESTRUTURAIS INEGOCIÁVEIS

1. Nodes inteligentes nunca executam efeitos externos.
2. Nodes operacionais nunca interpretam contexto.
3. Policy sempre tem precedência.
4. Memória nunca altera config.
5. Ledger é a verdade factual.
6. MemoryMaintenance nunca decide conteúdo.

---

# 📍 Resultado Arquitetural

Você tem agora:

* Um único modelo
* Um único graph
* Consciência localizada
* Memória regulada
* Evolução controlada
* Governança preservada

Sem micro-agentes.
Sem caos.
Sem sobreposição.

---
