# InSightDocs

**Multimodal Multi-Agent RAG pipeline for AI/ML research papers.**

InSightDocs ingests research PDFs — extracting text, figures, and tables — and serves answers through a stateful, confidence-aware retrieval-augmented generation pipeline powered by LangGraph multi-agent orchestration, LLM-based intent routing, document domain validation, and semantic FAQ caching.

---

## Table of Contents

- [InSightDocs](#insightdocs)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Architecture \& LangGraph Flow](#architecture--langgraph-flow)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the System](#running-the-system)
    - [1. Launch FastAPI Web Server](#1-launch-fastapi-web-server)
    - [2. Launch Streamlit Web Interface](#2-launch-streamlit-web-interface)
  - [API Reference](#api-reference)
  - [Ingestion Safeguards \& Pipeline](#ingestion-safeguards--pipeline)
    - [Document Ingestion Safeguards (`app/ingestion/document_validator.py`)](#document-ingestion-safeguards-appingestiondocument_validatorpy)
    - [Ingestion Flow](#ingestion-flow)
  - [Retrieval Pipeline](#retrieval-pipeline)
  - [Generation \& Multi-Agent Pipeline](#generation--multi-agent-pipeline)
    - [LangGraph Agent Architecture (`app/agents/`)](#langgraph-agent-architecture-appagents)
  - [Semantic FAQ Cache](#semantic-faq-cache)
  - [Confidence Scoring \& Fallback](#confidence-scoring--fallback)
  - [Prompts](#prompts)
  - [Testing \& Verification](#testing--verification)
    - [Verified Safeguards Status](#verified-safeguards-status)
  - [Configuration](#configuration)
  - [Data Directory Layout](#data-directory-layout)
  - [📖 Detailed Codebase Documentation](#-detailed-codebase-documentation)
  - [🔮 Future Roadmap \& Enhancements](#-future-roadmap--enhancements)

---

## Overview

InSightDocs is an enterprise-grade RAG system built specifically for AI/ML research paper Q&A. It processes complex multimodal content (page texts, figures with VLM descriptions, and tables with LLM descriptions) from PDFs and utilizes a stateful LangGraph multi-agent architecture for query classification, semantic caching, document retrieval, validation, and web search fallback.

Key capabilities:

- **LangGraph Multi-Agent Orchestration** — Graph-driven state workflow connecting Planner, RAG, FAQ, Support, Validation, and Web Search agents.
- **Multimodal Ingestion** — Extracts page text via Docling structural layout, figure images with PyMuPDF & VLM descriptions (Gemini/OpenAI), and tables with LLM markdown summaries (Groq).
- **Ingestion & Domain Safeguards** — Strict document validation checking file format (`.pdf`), size (≤10MB), length (≤100 pages), and lightweight LLM domain verification ensuring papers are strictly AI/ML/Deep Learning research before indexing.
- **Unified Structured LLM Planner Router** — Zero static regex or keyword rules; a single structured LLM classification call determines query route (`rag` vs `support`), visual intent (`needs_image`), web search necessity (`needs_web_search`), atomic structure (`is_atomic`), and domain (`ai_ml_technical`, `greeting`, `out_of_domain`).
- **Semantic FAQ Engine** — High-performance in-memory vector matching against cached Q&A with dynamic promotion based on frequency (`hit_count >= 3`) and confidence (`>= 0.75`), stored cleanly in `data/faq.json`.
- **Hybrid Retrieval & Reranking** — Dense + sparse BGE-M3 embeddings merged with RRF (Reciprocal Rank Fusion) and reranked using Cohere `rerank-v3.5`.
- **Confidence-Gated Validation & Escalation** — RAGAS-based answer faithfulness and relevancy judge gates final answers, falling back to Support Agent + Web Search when context confidence is low.
- **Multi-Provider Failover** — Resilient LLM failover order: Groq → Google Gemini → Azure OpenAI.
- **Interactive UI & REST API** — Streamlit app with step-by-step Flow Trace Reports and FastAPI web server.

---

## Architecture & LangGraph Flow

```
                              User Query
                                  │
                                  ▼
                        ┌──────────────────┐
                        │    FAQ Agent     │  ─── Match found ──► Return Cached FAQ Response
                        └────────┬─────────┘      (Confidence >= 0.75)
                                 │ No match
                                 ▼
                        ┌──────────────────┐
                        │  Planner Agent   │  ─── Structured LLM Intent Classification
                        └────────┬─────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │ route="rag"                   │ route="support"
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │ Retrieval (RRF)  │            │  Support Agent   │ ◄── Needs Web Search ──► Tavily Web Search
       │ + Cohere Rerank  │            │ (Tech Refusal /  │
       └────────┬─────────┘            │  Parametric LLM) │
                │                      └──────────────────┘
     Retrieval Confidence
                │
     < 0.50 ────┴──────────────────────────────┐
                │                              │
     >= 0.50    │                              │
                ▼                              │
       ┌──────────────────┐                    │
       │ Generation Agent │                    │
       └────────┬─────────┘                    │
                │                              │
     Retrieval Confidence                      │
     < 0.80     │      >= 0.80 (Fast Path)     │
                ├─────────────────┐            │
                ▼                 │            │
       ┌──────────────────┐       │            │
       │ Validation Agent │       │            │
       └────────┬─────────┘       │            │
                │                 │            │
        Validation Score          │            │
        < 0.70 ─┼─────────────────┼────────────┘
                │ >= 0.70         │
                ▼                 ▼
          Answer Returned & Evaluation Tracked
```

---

## Project Structure

```
InSightDocs/
├── api/                          # FastAPI web server
│   ├── main.py                   # FastAPI app factory & CORS
│   ├── routes.py                 # REST endpoints (/health, /ingest, /query, /faq, /documents, /status)
│   ├── schemas.py                # Pydantic request & response schemas
│   └── services.py               # Document ingestion & query execution services
│
├── app/                          # Core application logic
│   ├── agents/                   # LangGraph multi-agent system
│   │   ├── graph.py              # Compiled LangGraph workflow state graph
│   │   ├── nodes.py              # Graph execution nodes (planner, rag, support, validation, faq)
│   │   ├── edges.py              # Conditional edge routers between graph nodes
│   │   ├── state.py              # Graph state definition (AgentState)
│   │   ├── orchestrator.py       # High-level query execution orchestrator
│   │   ├── planner_agent.py      # Structured LLM query classifier & intent router
│   │   ├── support_agent.py      # Support agent with parametric fallback & refusal guardrails
│   │   ├── validation_agent.py   # RAGAS-style faithfulness & relevancy validation judge
│   │   ├── faq_agent.py          # Fast in-memory semantic FAQ caching engine
│   │   ├── web_search_tool.py    # Dynamic Tavily web search tool integration
│   │   ├── graph_mermaid.mmd     # Mermaid diagram source for graph architecture
│   │   └── graph_diagram.md      # Mermaid visual diagram documentation
│   │
│   ├── generation/               # Answer synthesis & evaluation
│   │   ├── generation_service.py # Core answer generation logic & provider failover
│   │   ├── query_utils.py        # Answer parsing, citation injection, & multimodal processing
│   │   └── interactive_evaluator.py # Real-time evaluation logging & trace collection
│   │
│   ├── ingestion/                # Document extraction, validation & indexing
│   │   ├── document_validator.py # Pre-ingestion safeguards (type, size, page count, LLM domain check)
│   │   ├── extractor_pipeline.py # Multimodal extraction pipeline orchestrator
│   │   ├── ingestor_pipeline.py  # Chunker, embedder, & Qdrant indexer
│   │   ├── document_loader.py    # PDF text parsing via Docling structural layout
│   │   ├── image_extractor.py    # Figure image extraction via PyMuPDF
│   │   ├── image_describer.py    # Figure description generator via VLM (Gemini/Azure)
│   │   ├── table_extractor.py    # Table extraction to Markdown
│   │   ├── table_describer.py    # Table description generator via LLM
│   │   ├── content_aggregator.py # Merges page artifacts into manifest.json
│   │   ├── chunker.py            # Splits document manifest into indexed chunks
│   │   ├── embedding_service.py  # BAAI/bge-m3 dense & sparse vector generation
│   │   ├── qdrant_ingestor.py    # Qdrant payload builder & vector batch loader
│   │   ├── status_tracker.py     # Checkpoint processing status tracking (completed/incomplete)
│   │   ├── metadata_builder.py   # Ingestion metadata builder
│   │   └── recovery.py           # Ingestion retry & error recovery logic
│   │
│   └── retriever/                # Retrieval engine
│       └── retrieval_service.py  # Hybrid search (Dense+Sparse RRF) & Cohere reranking
│
├── data/                         # Persistent application data
│   ├── faq.json                  # Semantic FAQ cache storage
│   ├── uploads/                  # Input PDF document storage
│   └── output/                   # Extracted manifests, images, tables & conversation logs
│
├── prompts/                      # Centralized LLM prompts
│   └── prompts.py                # Generation, visual description, table description, & agent prompts
│
├── utils/                        # Utilities & configuration
│   ├── settings.py               # Environment variables, thresholds, & application settings
│   ├── models_and_clients.py     # Singleton LLM, VLM, embedding, & vector store clients
│   └── logger_config.py          # Unified Loguru logging setup
│
├── tests/                        # Automated unit & integration test suites
│   ├── conftest.py               # Test fixtures & heavy client mocks
│   ├── test_domain_and_ingestion_safeguards.py # Ingestion & domain refusal tests
│   ├── test_faq_agent.py         # Semantic FAQ engine unit tests
│   ├── test_langgraph_workflow.py# LangGraph multi-agent execution tests
│   ├── test_ingest_unit.py       # Document ingestion unit tests
│   ├── test_retrieval_unit.py    # Hybrid retrieval & reranking tests
│   ├── test_generation_unit.py   # Answer synthesis unit tests
│   └── test_planner_query_correction.py # Planner router classification tests
│
├── app.py                        # Streamlit web UI with Flow Trace Reports
├── main.py                       # CLI application launcher
├── pyproject.toml                # Project dependencies & metadata
├── uv.lock                       # Dependency lockfile
└── README.md                     # Documentation
```

---

## Setup

### Prerequisites

- **Python 3.11+**
- **[`uv`](https://docs.astral.sh/uv/)** package manager
- **Qdrant Vector Database** (local instance or cloud cluster)
- **API Keys** for required providers:
  - Groq, Google Gemini, or Azure OpenAI (at least one LLM provider)
  - Cohere API Key (for hybrid reranking)
  - Tavily API Key (optional, for web search fallback)

### Installation

```bash
git clone <repo-url>
cd InSightDocs
uv sync
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Qdrant Vector DB
QDRANT_ENDPOINT=https://your-qdrant-instance.cloud
QDRANT_API_KEY=your_qdrant_api_key

# Primary LLM Providers (tested in order: Groq -> Gemini -> Azure OpenAI)
GROQ_API_KEY=your_groq_api_key
GROQ_API_KEY2=your_groq_api_key_2           # optional secondary key

GOOGLE_API_KEY3=your_google_gemini_api_key

AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_MODEL_NAME=your_deployment_name
AZURE_OPENAI_MODEL_VERSION=your_model_version

# Cohere (Reranking)
COHERE_API_KEY=your_cohere_api_key

# Tavily (Web Search Fallback)
TAVILY_API_KEY=your_tavily_key
```

---

## Running the System

### 1. Launch FastAPI Web Server

```bash
uv run uvicorn api.main:app --reload --port 8000
```
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Launch Streamlit Web Interface

```bash
uv run streamlit run app.py
```
- Web Application UI: [http://localhost:8501](http://localhost:8501)

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | System health check and API status |
| `/ingest` | `POST` | Upload and process a research PDF into Qdrant |
| `/query` | `POST` | Execute a multi-agent RAG query with trace data |
| `/documents` | `GET` | List all ingested documents across collections |
| `/faq` | `GET` | Retrieve stored semantic FAQ cache records |
| `/status/{doc_id}` | `GET` | Query extraction and ingestion status for a document |

---

## Ingestion Safeguards & Pipeline

### Document Ingestion Safeguards (`app/ingestion/document_validator.py`)

Prior to processing or vectorizing any file, `validate_document_for_ingestion` enforces strict technical boundaries:

1. **File Extension Check** — Rejects non-PDF files (`.txt`, `.docx`, etc.).
2. **File Size Boundary** — Rejects files exceeding **10 MB**.
3. **Page Count Boundary** — Rejects PDFs exceeding **100 pages**.
4. **AI/ML Technical Domain Check** — Samples text from the first 3 pages and executes a lightweight LLM domain classification. Files outside AI, Machine Learning, or Deep Learning (e.g., cooking recipes, general finance) are rejected immediately.
5. **Checkpoint Skip Logic** — Skips re-extraction and re-indexing if `doc_id` is already recorded as `completed` in processing status.

### Ingestion Flow

```
PDF File
  │
  ▼
document_validator.py  ──► [Type / Size / Page Count / LLM Domain Check]
  │ (Passed)
  ▼
document_loader.py     ──► Docling PDF parsing -> Page Markdown text
  │
  ├─► image_extractor.py ──► PyMuPDF figure extraction -> Image files
  │       └── image_describer.py ──► VLM figure summary generation
  │
  ├─► table_extractor.py ──► Table extraction to Markdown
  │       └── table_describer.py ──► LLM table description generation
  │
  └─► content_aggregator.py ──► Aggregate all into manifest.json
          │
          ▼
      chunker.py ──► Split document into semantic text/image/table chunks
          │
          ▼
  embedding_service.py ──► Generate BAAI/bge-m3 dense & sparse vectors
          │
          ▼
  qdrant_ingestor.py ──► Upsert to Qdrant vector store in batches
```

---

## Retrieval Pipeline

`app/retriever/retrieval_service.py` provides hybrid search and reranking:

- **Dense + Sparse Hybrid Search**: Queries Qdrant using `BAAI/bge-m3` dense vectors and sparse lexical weights.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse search rankings.
- **Cohere Rerank v3.5**: Reranks top hits for maximum context relevance before generation.

---

## Generation & Multi-Agent Pipeline

### LangGraph Agent Architecture (`app/agents/`)

1. **FAQ Agent (`faq_agent.py`)**: Strictly matches project-related questions about InSightDocs (capabilities, limits, architecture, API) using in-memory vector embeddings of `data/faq.json`.
2. **Planner Agent (`planner_agent.py`)**: Runs structured LLM classification returning `route`, `needs_image`, `needs_web_search`, `is_atomic`, and `domain`.
3. **Support Agent (`support_agent.py`)**: Handles non-RAG, greeting, and out-of-domain queries. Rejects general non-tech queries with standard identity refusal, uses LLM parametric knowledge for technical fallback, and invokes Tavily Web Search when `needs_web_search: true`.
4. **Generation Service (`generation_service.py`)**: Synthesizes comprehensive answers using retrieved chunks with multi-provider failover.
5. **Validation Agent (`validation_agent.py`)**: Evaluates faithfulness and relevance. Escalates to Support Agent if validation score falls below threshold (`0.70`).

---

## Semantic FAQ Cache

- **Project-Specific Knowledge**: Dedicated strictly to answering questions about our system, InSightDocs (capabilities, supported file limits, architecture, API endpoints, and setup).
- **Fast In-Memory Vector Search**: Pre-calculates dense vector embeddings of cached questions at startup for instant cosine similarity matching.
- **Clean Data Storage (`data/faq.json`)**: Stores clean, human-readable Q&A records without raw float vectors on disk.
- **Role Separation**: RAG handles technical paper domain queries, Support Agent handles greetings & fallbacks, and FAQ handles project-specific system questions. Dynamic auto-promotion of research paper/greeting queries has been removed.

---

## Confidence Scoring & Fallback

Configured in `utils/settings.py`:

| Setting | Default | Action |
|---|---|---|
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.50` | Score `< 0.50` -> Escalates immediately to Support Agent |
| `RETRIEVAL_SKIP_VALIDATION_THRESHOLD` | `0.80` | Score `≥ 0.80` -> Fast Path (skips LLM Validation Judge) |
| `VALIDATION_CONFIDENCE_THRESHOLD` | `0.70` | Validation Score `< 0.70` -> Escalates to Support Agent |

---

## Prompts

Located in `prompts/prompts.py`:

- **Generation Prompt**: Instructs LLM on structured multi-chunk synthesis, strict technical adherence, and citation formatting.
- **Image Description Prompt**: 5-section visual breakdown (Type, Purpose, Structure, Key Details, Findings) for VLM figure extraction.
- **Table Description Prompt**: Transforms Markdown tables into natural language preserving exact numerical metrics.
- **Planner & Validation Prompts**: Structured JSON classification schemas for intent routing and faithfulness judging.

---

## Testing & Verification

Comprehensive automated unit and integration tests are available:

```bash
# Run all core safeguards, FAQ, and LangGraph workflow tests
uv run pytest tests/test_domain_and_ingestion_safeguards.py tests/test_faq_agent.py tests/test_langgraph_workflow.py -v

# Run full unit test suite
uv run pytest tests/ -v
```

### Verified Safeguards Status
- Greeting Guardrail: **PASSED ✅**
- Out-of-Domain Denial: **PASSED ✅**
- Tech Fallback Knowledge: **PASSED ✅**
- Planner Agent LLM Router: **PASSED ✅**
- Document Ingestion Safeguards: **PASSED ✅**

---

## Configuration

Maintained in `utils/settings.py`:

| Property | Default Value | Description |
|---|---|---|
| `COLLECTION_NAME` | `insightdocs_demo_collection` | Qdrant default collection |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Vector embedding model |
| `COHERE_EMBEDDING_MODEL` | `rerank-v3.5` | Cohere reranker model |
| `MAX_FILE_SIZE_MB` | `10.0` | Ingestion file size limit |
| `MAX_PAGE_COUNT` | `100` | Ingestion page count limit |
| `VLM_PROVIDER` | `gemini` | Figure description VLM (`gemini` or `openai`) |

---

## Data Directory Layout

```
data/
├── faq.json                      # FAQ cache records
├── uploads/
│   └── <collection>/<file>.pdf   # Uploaded PDFs
└── output/
    ├── pdfs/<collection>/<doc>/  # Extracted manifests, images, and tables
    ├── processing_status/        # Completed & incomplete document trackers
    └── chat_conversations/       # Timestamped interaction & evaluation logs
```

---

## 📖 Detailed Codebase Documentation

For design decisions, performance benchmarks, and detailed technical specifications, refer to:
👉 **[Architecture and Design Documentation](file:///home/tanisha/Downloads/Simform/POC_MULTIMODAL_MULTIAGENT_RAG/Final_code/InSightDocs/docs/architecture_and_design.md)**

---

## 🔮 Future Roadmap & Enhancements

- [x] **Agentic Workflow Engine**: Stateful LangGraph multi-agent orchestration (`app/agents/graph.py`).
- [x] **Semantic FAQ Caching**: High-performance semantic Q&A cache with dynamic promotion (`app/agents/faq_agent.py`).
- [x] **Ingestion & Domain Safeguards**: Technical boundary validation & LLM paper domain checker (`app/ingestion/document_validator.py`).
- [ ] **Asynchronous Task Queue**: Celery/Redis integration for high-concurrency PDF ingestion background processing.
- [ ] **Multi-Collection Querying**: Simultaneous cross-collection search capabilities.
