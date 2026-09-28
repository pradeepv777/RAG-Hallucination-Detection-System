# System Requirements: RAG Hallucination Detection System

## 1. Functional Requirements

### Data Ingestion & Preprocessing (`dataset_loader.py`)
- **REQ-01**: Must read and parse `dataset/response.jsonl` and `dataset/source_info.jsonl` without modifying the original files.
- **REQ-02**: Must join records on `source_id` and cleanly extract context (passages for QA, reference for Summary/Data2txt), prompt/question, and generated response.
- **REQ-03**: Must strictly preserve and isolate `train` and `test` splits.
- **REQ-04**: Must derive ground-truth hallucination labels:
  - Answer-level: Binary (`is_hallucinated = True` if any ground truth span labels exist; `False` if labels is empty).
  - Claim/Span-level: Preserve span offsets and categorize labels into `Evident Conflict`, `Evident Baseless Info`, and `Subtle Baseless Info`.

### Claim Extraction (`claims.py`)
- **REQ-05**: Decompose an answer into independent, verifiable atomic factual claims.
- **REQ-06**: Provide a clean, deterministic base extraction (e.g. robust sentence/clause segmentation) and an optional LLM-assisted atomic claim extraction interface.
- **REQ-07**: Return a structured format: `List[Dict[str, Any]]` containing `claim_id` and `claim` text.

### Evidence Retrieval (`retrieve.py`)
- **REQ-08**: Split source context into manageable chunks (e.g. sentence/passage level).
- **REQ-09**: Index context chunks using `sentence-transformers` (configurable, default: `all-MiniLM-L6-v2`) and a local `faiss` index.
- **REQ-10**: For each claim, retrieve top-$k$ evidence chunks with similarity scores and chunk identifiers.

### NLI Verification (`verify.py`)
- **REQ-11**: Feed Claim (Hypothesis) and Retrieved Evidence (Premise) into an NLI Cross-Encoder (`cross-encoder/nli-deberta-v3-base` or equivalent).
- **REQ-12**: Classify relationship into 3 canonical categories:
  - `supported` (Entailment)
  - `contradicted` (Contradiction)
  - `unsupported` / `neutral` (Neutral / Not Enough Information)
- **REQ-13**: Expose verification logic independently from retrieval.

### Faithfulness Scoring (`score.py`)
- **REQ-14**: Assign claim-level verdicts based on verification probabilities and thresholds.
- **REQ-15**: Compute overall answer-level faithfulness score (e.g., $N_{\text{supported}} / N_{\text{total}}$).
- **REQ-16**: Formally handle edge cases: zero extracted claims, contradictory claims, unsupported claims, empty answer, or empty context.

### Evaluation & Baseline Comparison (`eval.py`)
- **REQ-17**: Benchmark predictions against RAGTruth ground truth on the test split.
- **REQ-18**: Compute Precision, Recall, F1, Accuracy, and Confusion Matrix for hallucination detection.
- **REQ-19**: Implement Baseline 1 (Embedding Similarity Threshold alone) and Baseline 2 (NLI Verification Pipeline).
- **REQ-20**: Ensure all metrics printed and recorded are genuinely computed on actual data—zero fabricated numbers.

### Serving & Interface (`app.py` & `streamlit_app.py`)
- **REQ-21**: Expose a FastAPI `POST /verify` endpoint accepting `question`, `context`, `answer` and returning faithfulness score and claim verdicts with evidence.
- **REQ-22**: Validate request/response payloads with Pydantic schemas.
- **REQ-23**: Provide an intuitive Streamlit UI highlighting supported, unsupported, and contradicted claims with supporting evidence.

### Packaging, Testing & Quality (`tests/`, `Dockerfile`, `README.md`)
- **REQ-24**: Comprehensive test suite testing edge cases (empty context, fully supported, contradicted, mixed).
- **REQ-25**: Dockerfile for reproducible container execution.
- **REQ-26**: Rigorous technical `README.md` covering theory, architecture, evaluation results, baseline table, and interview talking points.
