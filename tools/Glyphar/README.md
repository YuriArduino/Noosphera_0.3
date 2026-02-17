# 📜 Glyphar — OCR Adaptativo para Documentos Psicanalíticos

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production](https://img.shields.io/badge/status-production-green.svg)]()

**Glyphar** é um pipeline OCR adaptativo e otimizado para extração de textos de documentos psicanalíticos, integrado ao agente **Thoth** via LangGraph + FastAPI + LLMStudio.

> **Filosofia de Design:** *"Bom o suficiente para correção LLM" > "OCR perfeito"*
> Priorizamos velocidade e robustez sobre ganhos marginais de acurácia.

---

## 🎯 Visão Geral

```
┌─────────────────────────────────────────────────────────────────┐
│                    GLYPHAR OCR PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📥 INPUT → 📖 File I/O → 🔍 Quality Assessment → 🎯 Strategy   │
│                                                                  │
│  🎯 Strategy → 🖼️ Preprocessing → 🧠 Layout Detection → 🔤 OCR  │
│                                                                  │
│  🔤 OCR → 📊 Statistics → 📤 OCROutput → 🤖 LLM Correction      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Casos de Uso Principais

| Cenário | Estratégia | Velocidade | Acurácia |
|---------|-----------|-----------|----------|
| PDFs digitais (nativos) | `fast_scan` | ⚡⚡⚡ 1.5s/pág | 85-90% |
| Documentos críticos | `high_accuracy` | ⚡⚡ 2.8s/pág | 90-95% |
| Scans degradados | `noisy_documents` | ⚡ 3.5s/pág | 82-90% |

---

## ✨ Funcionalidades Principais

### 🔍 Análise de Qualidade Adaptativa
- **QualityAssessor** avalia cada página em <3ms
- Métricas: `sharpness` (Laplacian), `contrast` (Michelson), `quality_score`
- Classificação: EXCELLENT | GOOD | FAIR | POOR
- **60-70% dos documentos modernos** pulam pré-processamento pesado

### 🎯 Otimização Dinâmica de Configuração
```python
# ConfigStrategy.decide() seleciona automaticamente:
engine_config = ConfigStrategy.decide(
    layout_type="single",      # ou "double", "complex"
    quality={
        "is_clean_digital": False,
        "sharpness": 85.0,
        "contrast": 0.25,
    }
)
# Result: EngineConfig(pre_type="adaptive", psm=6, scale=1.3, oem=3)
```

### 📐 Detecção de Layout
| Detector | Precisão | Tempo | Uso |
|----------|---------|-------|-----|
| **ColumnLayoutDetector** | 98.7% (single), 96.3% (double) | ~2ms | 95% dos documentos |
| **AdvancedLayoutDetector** | 88% (multi/complex) | ~15ms | Fallback especializado |

### 🖼️ Pipeline de Pré-Processamento (8 Estratégias)
```yaml
execution_order:
  - "polarity_correction"    # Corrige inversão (texto branco em fundo escuro)
  - "grayscale"              # Converte para luminância
  - "shadow_removal"         # Remove sombras (CLAHE + background division)
  - "denoise"                # Reduz ruído (NLM, bilateral, median)
  - "deskew"                 # Corrige inclinação (±15°)
  - "smart_crop"             # Remove margens vazias
  - "threshold"              # Binarização (Otsu ou Adaptive)
```

### 🧠 Engine Tesseract Gerenciado
- **3 perfis**: `fast` (LSTM), `standard` (LSTM+legacy), `best` (all)
- **Fallback progressivo**: PSM 6 → PSM 11 → PSM 3 (legacy)
- **Cache LRU**: 1000 entradas, ~30% hit rate em batch
- **Dicionários de domínio**: 14 termos psicanalíticos (Freud, Lacan, inconsciente...)

### 📤 Output Imutável (OCROutput)
```python
output = pipeline.process("book.pdf")

# API response
JSONResponse(output.model_dump())

# LLM correction
llm_input = output.llm_ready_text()
# Structure:
# === OCR RESULTS - 320 PAGES ===
# === PAGE 1 | Confidence: 92.3% ===
# [text]
# === END OF DOCUMENT ===

# Dashboard summary
summary = output.summary()
# {file, file_hash, pages, page_hashes, words, average_confidence,
#  processing_time_s, needs_llm_correction}
```

---

## 🚀 Instalação

### Pré-requisitos

```bash
# Ubuntu/Debian
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-por tesseract-ocr-eng

# macOS
brew install poppler tesseract

# Windows
# Download: https://github.com/UB-Mannheim/tesseract/wiki
# https://github.com/oschwartz10612/poppler-windows
```

### Instalação do Pacote

```bash
# Clone o repositório
git clone https://github.com/noosphera/glyphar.git
cd glyphar

# Instale dependências
pip install -e .

# Ou com todas as extras
pip install -e ".[dev,test]"
```

### Verificação

```bash
# Verificar instalação
python -c "from glyphar import OCRPipeline; print('✅ Glyphar instalado')"

# Verificar Tesseract
tesseract --version

# Verificar Poppler
pdfinfo -v
```

---

## 📖 Quick Start

### Uso Básico

```python
from glyphar import OCRPipeline, OCRConfig
from glyphar.engines.managed.tesseract_managed import TesseractManagedEngine
from glyphar.layout.column_detector import ColumnLayoutDetector

# Configuração
config = OCRConfig(
    dpi=200,
    min_confidence=70.0,
    parallel=True,
    max_workers=4,
)

# Engine
engine = TesseractManagedEngine(
    tessdata_dir="resources/tessdata",
    languages="por+eng",
    model_type="fast",
    config=config,
)

# Pipeline
pipeline = OCRPipeline(
    engine=engine,
    layout_detector=ColumnLayoutDetector(),
    _preprocessing_strategies=[],  # Auto-selecionado pelo ConfigOptimizer
    config=config,
    include_llm_input=True,
)

# Processar documento
result = pipeline.process("documento.pdf", parallel=True, max_workers=8)

# Resultados
print(f"✅ {result.total_pages} páginas processadas")
print(f"⏱️  {result.statistics.total_processing_time_s:.1f}s")
print(f"📊 Acurácia média: {result.average_confidence:.1f}%")

# Correção LLM (se necessário)
if result.needs_llm_correction:
    llm_input = result.llm_ready_text()
    corrected = llm.correct(llm_input)
```

### Processamento em Lote

```python
from pathlib import Path

pdfs = sorted(Path("documents").glob("*.pdf"))

for pdf_path in pdfs:
    result = pipeline.process(str(pdf_path), parallel=True)

    # Salvar output
    (Path("output") / f"{pdf_path.stem}.json").write_text(
        result.model_dump(mode="json"),
        encoding="utf-8",
    )
    (Path("output") / f"{pdf_path.stem}.txt").write_text(
        result.full_text,
        encoding="utf-8",
    )
```

---

## ⚙️ Configuração

### Estrutura de Arquivos

```
docs/
├── capabilities/
│   ├── layout.yaml              # Detecção de layout
│   ├── preprocessing.yaml       # Pipeline de 8 estratégias
│   ├── analysis.yaml            # QualityAssessor
│   └── engine_modes.yaml        # Perfis fast/standard/best
│
├── tradeoffs/
│   ├── performance.yaml         # Benchmarks e targets
│   └── memory.yaml              # Gestão de memória
│
└── strategies/
    ├── fast_scan.yaml           # Velocidade > acurácia
    ├── high_accuracy.yaml       # Acurácia > velocidade
    └── noisy_documents.yaml     # Robustez para scans degradados
```

### runtime.yaml (Base)

```yaml
version: "1.0.0"

engine:
  model_type: "standard"
  language: "pt"
  enable_layout_analysis: true
  enable_preprocessing: true

pipeline:
  max_workers: 4
  batch_size: 8
  enable_parallelism: true

analysis:
  confidence_threshold: 85.0
  llm_correction_threshold: 92.0

limits:
  max_pages: 500
  max_file_size_mb: 100
  timeout_seconds: 300
```

### Uso com Estratégias

```bash
# Fast scan (PDFs digitais)
glyphar process document.pdf \
  --config runtime.yaml \
  --strategy docs/strategies/fast_scan.yaml

# Alta acurácia (documentos críticos)
glyphar process document.pdf \
  --config runtime.yaml \
  --strategy docs/strategies/high_accuracy.yaml

# Documentos ruidosos (scans degradados)
glyphar process document.pdf \
  --config runtime.yaml \
  --strategy docs/strategies/noisy_documents.yaml
```

---

## 🏗️ Arquitetura

### Componentes Principais

| Módulo | Responsabilidade | Arquivos Chave |
|--------|-----------------|----------------|
| **Core** | Orquestração do pipeline | `pipeline.py`, `runner.py`, `page_processor.py`, `file_processor.py` |
| **Engines** | Execução OCR | `tesseract_core.py`, `tesseract_managed.py`, `config_builder.py`, `fallback.py` |
| **Optimization** | Seleção adaptativa | `config_optimizer.py`, `config_strategy.py`, `image_preprocessor.py` |
| **Preprocessing** | Estratégias de imagem | 8 estratégias (polarity → threshold) |
| **Layout** | Detecção de estrutura | `column_detector.py`, `advanced_detector.py` |
| **Analysis** | Métricas de qualidade | `quality_assessor.py` |
| **File I/O** | Leitura de arquivos | `readers.py` (PDF + Image) |
| **Models** | Schemas Pydantic | `output.py`, `page.py`, `column.py`, `config.py`, `stats.py` |

### Fluxo de Execução

```
1. FileProcessor.process(file_path)
   ↓
2. read_pages() → List[NDArray[uint8]]
   ↓
3. run_parallel() ou run_sequential()
   ↓
4. PageProcessor.process(image, page_number, doc_prefix, doc_date)
   │
   ├─→ QualityAssessor.assess(image) → metrics
   ├─→ LayoutDetector.detect(image) → layout_type, regions
   ├─→ ConfigOptimizer.find_optimal_config(image, layout_type, metrics)
   │   ├─→ ConfigStrategy.decide(layout_type, metrics) → EngineConfig
   │   ├─→ ImagePreprocessor.apply(image, pre_type)
   │   ├─→ ImagePreprocessor.upscale(processed, scale)
   │   └─→ engine.recognize(processed, {psm, oem})
   │
   └─→ PageResult(id, page_number, columns, confidence, ...)
   ↓
5. OCROutput(file_metadata, pages, full_text, statistics, config, ...)
   ↓
6. output.summary() ou output.llm_ready_text()
```

---

## 📊 Performance

### Benchmarks (Intel i7, 200 DPI)

| Documento | Páginas | Estratégia | Tempo | Acurácia | Memória |
|-----------|---------|-----------|-------|----------|---------|
| Livro digital | 500 | `fast_scan` | 2 min | 85-90% | 300MB |
| Artigo acadêmico | 10 | `fast_scan` | 15s | 88-94% | 50MB |
| Scan degradado | 50 | `noisy_documents` | 3 min | 75-85% | 150MB |

### Comparação de Estratégias

| Métrica | `fast_scan` | `high_accuracy` | `noisy_documents` |
|---------|-------------|-----------------|-------------------|
| Velocidade (s/pág) | 1.5 | 2.8 | 3.5 |
| Acurácia | 85-90% | 90-95% | 82-90% |
| Memória (MB/pág) | 3 | 5 | 6 |
| Pré-processamento | Mínimo | Completo | Agressivo |
| Use Case | PDFs digitais | Críticos/arquivo | Scans degradados |

---

## 🔗 Integração com Agente Thoth

### LangGraph Tool Configuration

```python
# Thoth agent → Glyphar tool
from langgraph.graph import StateGraph
from glyphar import OCRPipeline

class ThothState(TypedDict):
    documents: List[str]
    ocr_results: List[OCROutput]
    corrected_texts: List[str]

def glyphar_tool(state: ThothState) -> ThothState:
    pipeline = OCRPipeline(...)

    for doc_path in state["documents"]:
        result = pipeline.process(doc_path)
        state["ocr_results"].append(result)

        if result.needs_llm_correction:
            corrected = llm.correct(result.llm_ready_text())
            state["corrected_texts"].append(corrected)

    return state

# Build graph
graph = StateGraph(ThothState)
graph.add_node("glyphar", glyphar_tool)
graph.set_entry_point("glyphar")
app = graph.compile()
```

### FastAPI Endpoint

```python
from fastapi import FastAPI, UploadFile
from glyphar import OCRPipeline

app = FastAPI()
pipeline = OCRPipeline(...)

@app.post("/process")
async def process_document(file: UploadFile):
    # Save uploaded file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Process
    result = pipeline.process(temp_path, parallel=True)

    # Return summary
    return result.summary()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### Docker Compose

```yaml
version: "3.8"

services:
  glyphar:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./resources/tessdata:/app/resources/tessdata
      - ./output:/app/output
    environment:
      - TESSDATA_PREFIX=/app/resources/tessdata
      - GLYPHAR_CONFIG=/app/container.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  llmstudio:
    image: llmstudio:latest
    ports:
      - "1234:1234"
```

---

## 🧪 Testes

### Executar Testes

```bash
# Testes unitários
pytest tests/unit/ -v

# Testes de integração
pytest tests/integration/ -v

# Teste completo do pipeline (requer PDFs em Test/Data/)
pytest tests/diagnostics/test_full_pipeline_diagnostic.py -v

# Com coverage
pytest --cov=glyphar --cov-report=html
```

### Output do Teste Diagnóstico

```json
// tests/output_data/full_pipeline/summary.json
{
  "pdf_count": 3,
  "results": [
    {
      "file": "PDF_A_Digital.pdf",
      "file_hash": "337f7ee9c65e39d29abd7610b48ad61465fb873b...",
      "pages": 3,
      "page_hashes": ["eba4f439...", "2838c746...", "a15f9c8a..."],
      "words": 846,
      "avg_confidence": 92.4,
      "processing_time_s": 4.36,
      "needs_llm_correction": false
    }
  ]
}
```

---

## 📚 Documentação

| Documento | Descrição |
|-----------|-----------|
| `docs/capabilities/layout.yaml` | Configuração de detecção de layout |
| `docs/capabilities/preprocessing.yaml` | Pipeline de 8 estratégias de pré-processamento |
| `docs/capabilities/analysis.yaml` | QualityAssessor e métricas |
| `docs/capabilities/engine_modes.yaml` | Perfis fast/standard/best |
| `docs/tradeoffs/performance.yaml` | Benchmarks e targets de performance |
| `docs/tradeoffs/memory.yaml` | Gestão de memória e limites |
| `docs/strategies/*.yaml` | Estratégias pré-configuradas |

---

## 🤝 Contribuindo

### Setup de Desenvolvimento

```bash
# Fork e clone
git clone https://github.com/your-username/glyphar.git
cd glyphar

# Instale em modo development
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install

# Rodar testes antes de commit
pytest tests/ -v
```

### Padrões de Código

```bash
# Formatação
black src/ tests/
isort src/ tests/

# Linting
pylint src/glyphar --rcfile=.pylintrc
mypy src/glyphar

# Segurança
bandit -r src/glyphar
```

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.

---

## 🙏 Agradecimentos

- **Tesseract OCR** — Engine OCR open-source
- **LangGraph** — Orquestração de agentes
- **Pydantic** — Validação e schemas
- **OpenCV** — Processamento de imagens
- **Projeto Noosphera** — Contexto psicanalítico

---

## 📞 Suporte

- **Issues**: https://github.com/noosphera/glyphar/issues
- **Discussions**: https://github.com/noosphera/glyphar/discussions
- **Email**: thoth@noosphera.ai

---

<div align="center">

**Glyphar** — Extração de texto adaptativa para análise psicanalítica

[⬆ Voltar ao topo](#-glyphar---ocr-adaptativo-para-documentos-psicanalíticos)

</div>
