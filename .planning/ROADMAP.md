# Roadmap: RAG Hallucination Detection System

## Phase Overview

| Phase | Description | Key Deliverables | Status |
|---|---|---|---|
| **Phase 1** | Repository & Dataset Inspection | Schema analysis, split validation, exploratory data checks | In Progress |
| **Phase 2** | Dataset Loader & Preprocessing | `dataset_loader.py` with strict split support and label extraction | Planned |
| **Phase 3** | Claim Extraction | `claims.py` (atomic claim decomposition with structured output) | Planned |
| **Phase 4** | Evidence Retrieval | `retrieve.py` (chunking, sentence-transformers, FAISS vector index) | Planned |
| **Phase 5** | NLI Verification | `verify.py` (DeBERTa cross-encoder classification: entailment/neutral/contradiction) | Planned |
| **Phase 6** | Faithfulness Scoring | `score.py` (claim verdicts, overall faithfulness ratio, edge cases) | Planned |
| **Phase 7** | Evaluation Against RAGTruth | `eval.py` (reproducible precision, recall, F1, accuracy, confusion matrix) | Planned |
| **Phase 8** | Baseline Comparison | Embedding similarity baseline vs NLI pipeline benchmark table | Planned |
| **Phase 9** | FastAPI Microservice | `app.py` (`POST /verify` endpoint, Pydantic schemas, validation) | Planned |
| **Phase 10** | Testing & Containerization | `tests/` (unit and integration tests), `Dockerfile`, `requirements.txt` | Planned |
| **Phase 11** | Streamlit Interactive UI | `streamlit_app.py` (interactive hallucination inspector with evidence highlights) | Planned |
| **Phase 12** | Documentation & Portfolio Polish | Comprehensive technical `README.md`, Mermaid architecture diagrams, interview talking points | Planned |

---

## Detailed Phase Breakdown

### Phase 1: Repository & Dataset Inspection
- Inspect `dataset/response.jsonl` and `dataset/source_info.jsonl` schema and record counts.
- Verify split distributions (`train`, `test`) and label distributions (`Evident Conflict`, `Evident Baseless Info`, `Subtle Baseless Info`).
- Inspect environment packages and runtime dependencies.

### Phase 2: RAGTruth Dataset Loader & Preprocessing
- Implement `dataset_loader.py`.
- Support `train`, `dev`, `test` partitions without leaking data.
- Standardize records: context, question, response, ground truth hallucinated spans, and binary hallucination flag.
- Add test script to verify joined schema integrity.

### Phase 3: Claim Extraction
- Implement `claims.py`.
- Formulate deterministic claim extraction using robust sentence boundary and clause parsing.
- Provide clean abstraction for pluggable atomic claim extraction.
- Validate structured output format (`claim_id`, `claim`).

### Phase 4: Context Chunking & FAISS Evidence Retrieval
- Implement `retrieve.py`.
- Implement dynamic chunking with configurable overlap.
- Index chunks with `sentence-transformers/all-MiniLM-L6-v2` and FAISS `IndexFlatIP` (cosine similarity with normalized vectors).
- Return top-$k$ evidence snippets with similarity scores and chunk IDs.

### Phase 5: NLI Verification
- Implement `verify.py`.
- Set up Hugging Face cross-encoder NLI model (e.g. `cross-encoder/nli-deberta-v3-base` or lightweight fallback).
- Map premise (context chunk) and hypothesis (claim) to `supported` (entailment), `contradicted` (contradiction), or `neutral`.
- Ensure verification executes independently from retrieval.

### Phase 6: Faithfulness Scoring
- Implement `score.py`.
- Define claim-level verdict resolution logic (handling top-$k$ retrieved chunks per claim).
- Compute answer-level faithfulness score $S \in [0, 1]$.
- Explicitly handle edge cases: zero claims, completely unsupported responses, contradictory claims, empty inputs.

### Phase 7: Evaluation Against RAGTruth
- Implement `eval.py`.
- Evaluate detector across test set samples.
- Compute Precision, Recall, F1, Accuracy, and Confusion Matrix against RAGTruth ground truth.
- Log performance by task type (`QA`, `Summary`, `Data2txt`).

### Phase 8: Baseline Comparison
- Implement Baseline 1: Cosine similarity threshold (embedding-only retrieval similarity without NLI).
- Compare with Baseline 2: Full NLI verification pipeline.
- Record real, empirically measured metrics into the benchmark comparison table.

### Phase 9: FastAPI Microservice
- Implement `app.py`.
- Define Pydantic request models (`VerificationRequest`) and response models (`VerificationResponse`, `ClaimVerdict`).
- Expose `POST /verify` and healthcheck endpoints.
- Test endpoint via local client requests.

### Phase 10: Testing & Docker
- Create comprehensive pytest suite in `tests/`:
  - `test_dataset_loader.py`
  - `test_claims.py`
  - `test_retrieve.py`
  - `test_verify.py`
  - `test_score.py`
  - `test_api.py`
- Create production-ready `Dockerfile` and finalized `requirements.txt`.

### Phase 11: Streamlit Interactive UI
- Implement `streamlit_app.py`.
- Input fields for Question, Context, and LLM Answer.
- Visual display of overall faithfulness score, claim breakdown badges, and side-by-side evidence inspection.

### Phase 12: Technical Documentation Polish
- Author in-depth `README.md` complete with problem statement, architecture diagrams, benchmark results table, API guide, and interview walkthrough.
