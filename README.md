# AdaptiveRAG

A Closed-Loop Autonomous Retrieval Optimization Framework (50% Milestone)

## Project Overview

AdaptiveRAG is a master's level research project that explores the concept of a closed-loop, self-optimizing Retrieval-Augmented Generation (RAG) system. Traditional RAG systems use static configurations (e.g., fixed chunk sizes, static Top-K, single retrieval methods) which can fail unpredictably depending on the query. 

This project aims to implement an autonomous loop:
Observe → Evaluate → Diagnose → Generate Hypothesis → Experiment → Measure → Accept/Reject → Remember → Improve

## Current 50% Milestone Implementation

For this 50% milestone, the following foundational components have been implemented:

### IMPLEMENTED Features
1. **Production RAG Baseline**: 
   - Document Ingestion (PDF, DOCX, TXT, MD)
   - Chunking Strategy
   - Hybrid Retrieval (FAISS Dense + BM25 Sparse)
   - Configurable Top-K and weights
   - Answer Generation via LLM (Ollama)

2. **Automated Evaluation Engine**:
   - Rule-based evaluation heuristics (Relevance, Faithfulness, Context Coverage, Completeness).
   - Real-time scoring and failure detection based on thresholds.

3. **Diagnostic Engine**:
   - Classifies query failures into specific categories (e.g., INSUFFICIENT_TOP_K, LOW_RETRIEVAL_RELEVANCE, POSSIBLE_HALLUCINATION).
   - Generates confidence scores and evidence for diagnoses.

4. **Policy Recommendation Engine**:
   - Maps diagnoses to targeted optimization recommendations (e.g., INCREASE_TOP_K, REWRITE_QUERY, ADJUST_HYBRID_WEIGHT).
   - Recommends actions without blindly altering production (important for the research framework).

5. **Traceability & History**:
   - Comprehensive query logging.
   - Storage of chunks, queries, evaluations, and recommendations via SQLAlchemy/SQLite.
   - Clean, professional research UI built with Jinja2/HTML/CSS to observe the pipeline.

### PLANNED Features (Future Work)
- Full Autonomous Experimentation (Phase 5)
- Automated Rollback System
- Multi-Agent Orchestration (e.g., using LangGraph)
- Long-term Persistent Memory Engine for cross-query learning
- Continuous Online Learning
- A/B Benchmarking on real query datasets

## Architecture

```text
adaptive_rag/
├── app/                  # Main FastAPI Application
│   ├── api/              # API endpoints/routes
│   ├── core/             # Configuration & Logging
│   ├── models/           # SQLAlchemy DB Models
│   ├── rag/              # Ingestion, Retrieval, Generation Pipeline
│   ├── evaluation/       # Evaluator logic
│   ├── diagnosis/        # Diagnostic heuristics
│   ├── policy/           # Recommendation mapping
│   ├── templates/        # Jinja2 UI Templates
├── data/                 # Sample Data & Storage
├── scripts/              # Seed scripts
├── requirements.txt      # Dependencies
└── run.py                # App entrypoint
```

## Setup Instructions

1. **Virtual Environment Setup**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ollama Setup (Required for LLM)**:
   - Ensure you have Ollama installed and running on `localhost:11434`.
   - Pull the model: `ollama run llama3.1` (or change `OLLAMA_MODEL` in `.env` if using another).

4. **Environment Configuration**:
   - Copy `.env.example` to `.env` and verify settings.

5. **Seed Demo Data**:
   - Run the script to initialize DB and seed initial sample knowledge base:
   ```bash
   python scripts/seed_demo_data.py
   ```

6. **Run the Application**:
   ```bash
   python run.py
   ```
   - Open `http://localhost:8000` in your browser.

## Demonstration Scenarios Supported

1. **Good Retrieval**: Ask a clear question covered in the sample data (e.g., "What is the authentication mechanism?").
2. **Ambiguous Query**: Ask a vague query to trigger `QUERY_AMBIGUITY` diagnosis.
3. **Missing Context**: Ask something completely unrelated (e.g., "How do I bake a cake?") to trigger `POSSIBLE_HALLUCINATION` or context missing logic.
4. **Insufficient Top-K**: Use a low top-k in `.env` and ask a question spanning multiple documents.
