# InSightDocs: Multi-Agent Multimodal Document Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Vector_DB-red.svg)](https://qdrant.tech/)
[![BGE-M3](https://img.shields.io/badge/BGE--M3-Native_ColBERT_Rerank-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**InSightDocs** is an enterprise-grade AI platform designed to ingest, index, and analyze complex mixed-content research papers and enterprise documentation (containing text, formulas, tables, charts, and embedded figure diagrams). Powered by a collaborating team of **LangGraph AI Agents**, **Qdrant Hybrid Vector Store**, **BGE-M3 Native ColBERT Reranking**, and a **Multi-Tier LLM Failover Pool**.

---

## 📂 Project Directory Structure

```
InSightDocs/
├── api/                        # FastAPI Web Application Gateway
│   ├── routes.py               # REST Endpoints (/health, /query, /ingest, /inventory)
│   ├── schemas.py              # Pydantic Request/Response Models
│   └── services.py             # FastAPI Service Dispatcher & Ingestion Helpers
│
├── app/                        # Core Application Engine
│   ├── agents/                 # Multi-Agent LangGraph Framework
│   │   ├── graph.py            # Stateful LangGraph Workflow Engine
│   │   ├── nodes.py            # Workflow Execution Nodes (Retrieval, Generation, Validation)
│   │   ├── edges.py            # Conditional Edge Routing & Confidence Thresholds
│   │   ├── planner_agent.py    # Intent Classification, Query Rewriting & Typo Fixes
│   │   ├── support_agent.py    # FAQ Routing, User Identity Guardrail & Inventory Listing
│   │   ├── validation_agent.py # Faithfulness, Relevancy & Context Recall Evaluation
│   │   └── web_search_tool.py  # Parallel Internet Web Search (Tavily / Serper API)
│   │
│   ├── ingestion/              # Document Ingestion & Parsing Services
│   │   ├── pdf_extractor.py    # PyMuPDF / Marker Content Extraction (Text, Tables, Figures)
│   │   ├── qdrant_indexing.py  # BM25 Sparse + BGE-M3 Dense Vector Upserting
│   │   └── ingest_markdown_faq.py # Semantic Sub-Chunking & Ingestion for data/faq.md
│   │
│   ├── retriever/              # Hybrid Retrieval Engine
│   │   └── retrieval_service.py # Qdrant Search, ColBERT Reranking & Title Pagination
│   │
│   └── generation/             # Answer Synthesis Engine
│       ├── generation_service.py # Prompt Builder & LLM Failover Invocation
│       └── query_utils.py       # Retrieval Confidence Scoring & Answer Cleanup
│
├── data/                       # Data Stores & Artifacts
│   ├── faq.md                  # Project FAQ & 103+ Research Paper Titles Manifest
│   ├── output/                 # Extracted PDF Assets, Images & Conversions
│   │   ├── pdfs/               # Extracted Text, Tables & Images per Document
│   │   ├── processing_status/  # complete.json Recovery Checkpoints
│   │   └── chat_conversations/ # Chat History Logs
│   └── reports/                # Benchmark Telemetry & Test Evaluation Outputs
│
├── docs/                       # Architectural Specifications
│   └── architecture_and_design.md # Agentic & Cloud Mermaid Diagrams + ADRs
│
├── groundtruth_evaluation/     # Ground Truth Evaluation Benchmark Suite
│   ├── build_datasets.py       # Dataset Generator (5, 100, 1000 query sets)
│   ├── evaluate_retrieval.py   # Retrieval Benchmark (MRR, MAP, Hit Rate @ K)
│   └── evaluate_generation.py  # Generation Benchmark (LLM-as-Judge Faithfulness)
│
├── prompts/                    # Master Prompt Engineering Registry
│   └── prompts.py              # Planner, Generation, Support & Validation Prompts
│
├── tests/                      # Automated Production Pytest Test Suite
│   ├── test_api_routes.py      # FastAPI Endpoint Tests
│   ├── test_planner_agent.py   # Planner Intent & Typo Correction Tests
│   ├── test_ingestion_service.py # Ingestion & Checkpoint Recovery Tests
│   ├── test_retrieval_service.py # Hybrid Qdrant Search & Title Pagination Tests
│   ├── test_generation_service.py # LLM Failover Pool & Answer Grounding Tests
│   ├── test_support_agent.py   # User Identity Guardrail & Paper Inventory Tests
│   ├── test_validator_and_web.py # Faithfulness Scoring & Web Search Tests
│   └── test_multimodal_features.py # Multimodal Image Chunk Lookup Tests
│
├── utils/                      # Infrastructure & Model Utilities
│   ├── logger_config.py        # Central Loguru Logger Setup
│   ├── models_and_clients.py   # Multi-Tier LLM Failover Pool Manager
│   └── settings.py             # Environment Configuration Settings
│
├── extra/                      # Archived Benchmark Artifacts & Reference Guides
├── app.py                      # Interactive Streamlit UI Client
├── main.py                     # FastAPI Application Server Entrypoint
├── pyproject.toml              # Project Dependencies & Metadata
└── README.md                   # Master Documentation Guide
```

---

## 🌟 Core Architecture Services & Agents

### 🧠 3 Core Application Services

1. **Ingestion Service (`app/ingestion/`)**:
   - Extracts text, markdown tables, and visual figure images (`page_N_figure_M.png`).
   - Maintains recovery checkpoints (`data/output/processing_status/complete.json`) to skip already processed papers.
   - Enforces dynamic upload safeguards (e.g. 20 MB max file size, 100 pages max).
   - Sub-chunks `data/faq.md` into semantic FAQ points upserted into Qdrant collection `InsightDocs`.

2. **Retrieval Service (`app/retriever/`)**:
   - Combines dense BGE-M3 embeddings with BM25 sparse keyword vectors stored in Qdrant.
   - Applies **BGE-M3 Native ColBERT reranking** to elevate exact technical phrase and table precision.
   - Uses fast Qdrant payload scrolling to retrieve all **103+ indexed paper titles** in sub-second latency.

3. **Generation Service (`app/generation/`)**:
   - Powered by a **Multi-Tier LLM Failover Pool** (`Azure OpenAI GPT-5.4` ➔ `Azure OpenAI Backup` ➔ `OpenAI GPT-4o-mini` ➔ `Groq Llama-3.3-70B`).
   - Formulates grounded technical responses formatted with clean LaTeX math equations (`\(...\)`, `\[...\]`) and Markdown tables.

---

### 🤖 Primary AI Agents & Nodes

| Agent / Node Name | Module Path | Purpose & Features |
|---|---|---|
| **Planner Agent** | `app/agents/planner_agent.py` | Classifies query route (`rag` vs `support`), scope (`documents` vs `faq`), fixes typos (*"vanderrer"* ➔ *"VANDERER"*), resolves pronouns, detects atomic vs non-atomic intent, and flags `needs_image` / `needs_web_search`. |
| **Support Agent** | `app/agents/support_agent.py` | Enforces user identity guardrails (*"who am i"* ➔ identity refusal + assistant intro), lists 103+ Knowledge Base paper titles, and handles parametric ML fallbacks. |
| **Validation Agent** | `app/agents/nodes.py` | Evaluates answer factuality against retrieved context ($0.40\text{faithfulness} + 0.30\text{relevancy} + 0.30\text{recall}$). Routes low-confidence queries to support fallback. |
| **Web Search Agent** | `app/agents/web_search_tool.py` | Performs parallel real-time internet search via Tavily / Serper API when SOTA verification is flagged by Planner. |

---

## 🛠️ Step-by-Step Execution Guide

### 1. Environment Configuration
Copy `example.env` to `.env` and configure your API keys:

```bash
cp example.env .env
```

Key environment variables:
```ini
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=InsightDocs

AZURE_OPENAI_API_KEY_5_4=your_azure_key
AZURE_OPENAI_ENDPOINT_5_4=https://your-resource.openai.azure.com/

GROQ_API_KEY_1=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

---

### 2. Running System Services

#### A. Start FastAPI Backend API Server
```bash
# Run server via main.py entrypoint
python main.py

# Or directly via Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- **API Documentation**: Open `http://localhost:8000/docs` in your browser.
- **Endpoints**:
  - `GET  /health`: Health status endpoint
  - `GET  /inventory`: List total indexed paper count & document titles
  - `POST /query`: Query API accepting text and optional image upload
  - `POST /ingest`: Upload and ingest PDF document into Qdrant

#### B. Start Streamlit Interactive UI Frontend
```bash
streamlit run app.py
```
- Open `http://localhost:8501` to use the interactive chat dashboard.

#### C. Ingest FAQ & System Knowledge Base
```bash
python app/ingestion/ingest_markdown_faq.py
```

---

### 3. Running Automated Production Tests (`pytest`)

Execute the complete automated test suite covering all services, agents, guardrails, and API routes:

```bash
# Run all tests in parallel/verbose mode
pytest tests/ -v

# Run specific test modules
pytest tests/test_api_routes.py -v
pytest tests/test_planner_agent.py -v
pytest tests/test_ingestion_service.py -v
pytest tests/test_retrieval_service.py -v
pytest tests/test_generation_service.py -v
pytest tests/test_support_agent.py -v
pytest tests/test_validator_and_web.py -v
pytest tests/test_multimodal_features.py -v
```

---

### 4. Running Ground Truth Benchmark Evaluation

To compute retrieval accuracy (MRR, MAP, Hit Rate @ K) and LLM-as-Judge generation quality:

```bash
# 1. Generate ground truth datasets
python groundtruth_evaluation/build_datasets.py

# 2. Evaluate retrieval performance
python groundtruth_evaluation/evaluate_retrieval.py

# 3. Evaluate generation quality (LLM-as-Judge)
python groundtruth_evaluation/evaluate_generation.py
```

---

## ⚡ Special Production Features

1. **Recovery Checkpoints & Extraction/Ingestion Split**:
   - `complete.json` tracks ingested document IDs so ingestion can be resumed safely without redundant re-processing.
2. **User Identity Guardrail (`who am i`)**:
   - When asked *"who am i?"*, the Support Agent explicitly clarifies it does not have access to personal user data while introducing InSightDocs.
3. **Out-of-Domain & Jailbreak Guardrails**:
   - Non-technical queries (e.g., cooking, sports) or system prompt injection attempts are safely caught and guardrailed.
4. **Dynamic Upload Safeguards**:
   - Imposes configurable file size (20 MB) and page count (100 pages) limits during PDF ingestion.
5. **Multimodal Visual Figures**:
   - Automatically extracts figure diagrams (`page_N_figure_M.png`) and links them to answers when figure visual plots are requested.

---

## 📖 Architectural Specifications

For flowcharts, cloud topology, and ADRs, visit:
👉 **[Architecture and Design Specifications](docs/architecture_and_design.md)**
