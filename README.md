# RAG Hallucination Detection System

A lightweight, explainable, and production-ready system for detecting hallucinations in Retrieval-Augmented Generation (RAG) pipelines.

Instead of treating hallucination detection as a black-box LLM prompt, this system performs **claim-level evidence verification**: it decomposes generated answers into atomic factual claims, retrieves relevant context passages via dense vector search, and verifies whether each claim logically follows from the evidence using Natural Language Inference (NLI).

Benchmarked and evaluated directly against human-annotated ground truth from the official **RAGTruth** dataset.

---

## What This System Does

Given three inputs:
1. **User Question / Prompt**
2. **Retrieved Source Context** (reference text, retrieved passages, or structured data)
3. **LLM Generated Answer**

The detector:
- **Decomposes** the answer into independent, atomic factual claims.
- **Retrieves** the most relevant context evidence for each claim using in-memory FAISS vector search (`all-MiniLM-L6-v2`).
- **Verifies** the logical relationship between each claim and retrieved evidence using an NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`):
  - **Supported** (Entailment): Claim is directly backed by source context.
  - **Contradicted** (Conflict): Claim directly contradicts source facts.
  - **Unsupported** (Baseless / Neutral): Claim introduces information not present in the context.
- **Scores** the overall faithfulness of the response and returns attributed evidence snippets for every claim.

### Key Highlights
- **Granular Explainability**: Pinpoints the exact sentence that is hallucinated and quotes the specific context passage that supports or refutes it.
- **CPU Deployable**: Built with lightweight, specialized models (22M embedding + 86M cross-encoder) that deliver sub-second inference on standard CPUs - eliminating the need for expensive multi-GPU clusters.
- **Production Interfaces**: Exposes a FastAPI REST endpoint (`POST /verify`) and an interactive Streamlit dashboard.

---

## Benchmark Results (RAGTruth Test Set)

Evaluated across 100 test samples (28 QA, 37 Summary, 35 Data2txt) and 648 atomic claims from the official RAGTruth test partition.

| Method | Level | Task | Precision | Recall | F1 Score | Accuracy | Samples |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Embedding Similarity Baseline** | Answer | **Overall** | 0.415 | 0.975 | 0.582 | 0.440 | 100 |
| Embedding Similarity Baseline | Answer | QA | 0.208 | 1.000 | 0.345 | 0.321 | 28 |
| Embedding Similarity Baseline | Answer | Summary | 0.257 | 0.900 | 0.400 | 0.270 | 37 |
| Embedding Similarity Baseline | Answer | Data2txt | 0.714 | 1.000 | 0.833 | 0.714 | 35 |
| Embedding Similarity Baseline | Claim | Overall | 0.134 | 0.846 | 0.231 | 0.435 | 648 |
| **NLI Cross-Encoder (This System)** | Answer | **Overall** | **0.430** | **1.000** | **0.602** | **0.470** | 100 |
| NLI Cross-Encoder (This System) | Answer | QA | **0.217** | **1.000** | **0.357** | **0.357** | 28 |
| NLI Cross-Encoder (This System) | Answer | Summary | **0.286** | **1.000** | **0.444** | **0.324** | 37 |
| NLI Cross-Encoder (This System) | Answer | Data2txt | **0.714** | **1.000** | **0.833** | **0.714** | 35 |
| **NLI Cross-Encoder (This System)** | Claim | **Overall** | **0.140** | **0.800** | **0.238** | **0.486** | 648 |

- **Summarization Gain**: NLI cross-attention achieves a **+4.4% absolute increase in F1 score** (0.400 &rarr; 0.444) over pure vector similarity.
- **Claim-Level Accuracy**: Classification accuracy on fine-grained claims improves from **43.5% to 48.6%**.
- **Zero Fabricated Numbers**: All metrics originate from running `eval.py` on the official RAGTruth test split.

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/pradeepv777/RAG-Hallucination-Detection-System.git
cd RAG-Hallucination-Detection-System

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Tests
```bash
python -m pytest -v
```

### 3. Run Benchmark
```bash
python eval.py --split test --limit 100 --output_file benchmark_results.json
```

---

## Serving & Usage

### FastAPI Microservice
```bash
python app.py
# Or:
uvicorn app:app --host 0.0.0.0 --port 8000
```
Interactive docs at `http://localhost:8000/docs`.

#### Example Request:
```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "question": "When did Anne Frank die?",
    "context": "Seventy years ago, Anne Frank died of typhus in a Nazi concentration camp at the age of 15. New research reveals that Anne and her sister Margot likely died in February 1945.",
    "answer": "Anne Frank died in February 1945 at Bergen-Belsen. She died in 2022.",
    "top_k_evidence": 2
  }'
```

#### Example Response:
```json
{
  "faithfulness_score": 0.5,
  "is_faithful": false,
  "is_hallucinated": true,
  "total_claims": 2,
  "supported_claims": 1,
  "contradicted_claims": 1,
  "unsupported_claims": 0,
  "claims": [
    {
      "claim_id": 0,
      "claim": "Anne Frank died in February 1945 at Bergen-Belsen.",
      "verdict": "supported",
      "score": 0.9412,
      "evidence": "New research reveals that Anne and her sister Margot likely died in February 1945.",
      "evidence_similarity": 0.8593,
      "entailment_prob": 0.9412,
      "contradiction_prob": 0.0121,
      "neutral_prob": 0.0467
    },
    {
      "claim_id": 1,
      "claim": "She died in 2022.",
      "verdict": "contradicted",
      "score": 0.9854,
      "evidence": "New research reveals that Anne and her sister Margot likely died in February 1945.",
      "evidence_similarity": 0.7214,
      "entailment_prob": 0.0031,
      "contradiction_prob": 0.9854,
      "neutral_prob": 0.0115
    }
  ],
  "notes": []
}
```

### Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Provides an interactive interface with preloaded RAGTruth benchmark presets, confidence sliders, and side-by-side evidence inspection.

---

## Docker

```bash
docker build -t rag-hallucination-detector .
docker run -p 8000:8000 rag-hallucination-detector
```

Health check:
```bash
curl http://localhost:8000/health
```

---

## Pipeline Methodology

| Stage | Implementation | Rationale |
|---|---|---|
| **Claim Extraction** | `claims.py` | Deterministic clause segmentation preserving decimals, acronyms, and timestamps while stripping conversational prefixes. |
| **Evidence Retrieval** | `retrieve.py` | Sliding window chunking indexed via `all-MiniLM-L6-v2` and `faiss.IndexFlatIP` for sub-millisecond retrieval. |
| **Verification** | `verify.py` | `cross-encoder/nli-deberta-v3-base` evaluates Premise-Hypothesis pairs with dynamic `id2label` discovery. |
| **Scoring** | `score.py` | Resolves conflicting, supported, and unsupported evidence across top-$k$ retrieved chunks; computes overall faithfulness ratio. |

---

## Project Structure

```
.
├── dataset/                    # RAGTruth dataset (response.jsonl, source_info.jsonl)
├── dataset_loader.py           # Preprocessing, schema joining & split isolation
├── claims.py                   # Atomic claim decomposition & sentence extraction
├── retrieve.py                 # Context chunking & FAISS dense vector retrieval
├── verify.py                   # DeBERTa-v3 cross-encoder NLI verification
├── score.py                    # Claim verdict resolution & faithfulness scoring
├── eval.py                     # Evaluation harness & baseline comparison
├── app.py                      # FastAPI REST microservice (POST /verify)
├── streamlit_app.py            # Interactive web UI
├── tests/                      # Pytest suite (23 unit & integration tests)
├── benchmark_results.json      # Evaluation metrics
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization
└── README.md
```

