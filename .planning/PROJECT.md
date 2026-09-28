# RAG Hallucination Detection System

## 1. Project Vision & Core Question
**"Can claim-level evidence verification using Natural Language Inference (NLI) detect hallucinated or unsupported claims in RAG-generated answers?"**

This system evaluates RAG (Retrieval-Augmented Generation) answers by decomposing them into atomic factual claims, retrieving relevant context passages via dense vector search (FAISS + sentence-transformers), and performing granular NLI-based verification (Entailment / Contradiction / Neutral) against the evidence.

## 2. Core Principles
- **No Synthetic Shortcuts**: Evaluated directly against the published **RAGTruth** benchmark (`dataset/response.jsonl` and `dataset/source_info.jsonl`).
- **Data Hygiene**: Preserve existing `train`/`test` splits strictly. Never train and evaluate on overlapping data.
- **Empirical Rigor**: No fabricated numbers or arbitrary metrics. Every score reported in the final README originates from direct pipeline evaluation.
- **Minimal, Explainable Stack**: Clear, modular Python code without unnecessary dependencies (no heavy orchestration frameworks like LangChain, no complex distributed infrastructure).
- **Dual-Level Output**:
  - **Claim-level verdicts**: Supported / Contradicted / Unsupported with retrieved evidence snippets and confidence scores.
  - **Answer-level faithfulness**: Ratio of supported claims and aggregate hallucination classification.

## 3. Dataset Characteristics (RAGTruth)
- `dataset/response.jsonl`: Contains model outputs, prompt IDs, split designation (`train`, `test`), quality rating, and fine-grained annotated hallucination spans (`labels`) categorized into:
  - `Evident Conflict`
  - `Evident Baseless Info`
  - `Subtle Baseless Info`
- `dataset/source_info.jsonl`: Contains `source_id`, `task_type` (`QA`, `Summary`, `Data2txt`), `source`, `source_info` (passages or reference text), and `prompt`.
- Join Key: `source_id`.

## 4. Pipeline Architecture
```mermaid
flowchart TD
    A[Input: Context + Prompt + Generated Answer] --> B[Claim Extraction (claims.py)]
    B --> C[Evidence Retrieval (retrieve.py: FAISS + all-MiniLM-L6-v2)]
    C --> D[NLI Verification (verify.py: DeBERTa-v3 Cross-Encoder)]
    D --> E[Faithfulness Scoring (score.py: Claim Verdicts + Score)]
    E --> F[Evaluation vs RAGTruth (eval.py: Precision / Recall / F1)]
    E --> G[FastAPI Service (app.py: POST /verify)]
    E --> H[Interactive UI (streamlit_app.py)]
```

## 5. Technology Stack
- **Language**: Python 3.13 / 3.11
- **Embeddings & Vector Search**: `sentence-transformers` (`all-MiniLM-L6-v2`), `faiss-cpu`
- **NLI Model**: `cross-encoder/nli-deberta-v3-base` (or equivalent lightweight Hugging Face DeBERTa NLI cross-encoder)
- **Evaluation**: `scikit-learn`, `pandas`, `numpy`
- **API & Serving**: `fastapi`, `uvicorn`, `pydantic`
- **UI & Visualization**: `streamlit`
- **Testing & Containerization**: `pytest`, `docker`
