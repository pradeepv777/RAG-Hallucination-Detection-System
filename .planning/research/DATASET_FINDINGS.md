# RAGTruth Dataset Schema & Statistical Analysis

## 1. Overview
RAGTruth is a word/span-level hallucination benchmark for Retrieval-Augmented Generation across multiple foundational models and tasks.

- **Source File**: `dataset/source_info.jsonl` (2,965 source contexts)
- **Response File**: `dataset/response.jsonl` (17,790 responses)
- **Evaluation Split Guarantee**:
  - `train`: 15,090 responses
  - `test`: 2,700 responses (2,675 with `quality == 'good'`)
  - No source contexts overlap across splits.

## 2. Task Types & Context Representation
In `dataset/source_info.jsonl`:
- **QA** (989 sources):
  - `source_info` is a JSON dict with `question` (str) and `passages` (str containing retrieved context).
  - `prompt` contains the instruction formatted for the model.
- **Summary** (943 sources):
  - `source_info` is a `str` (the source article to be summarized).
  - `prompt` contains the summarization prompt.
- **Data2txt** (1,033 sources):
  - `source_info` is structured JSON containing business info, opening hours, attributes, and user reviews.
  - `prompt` contains structured instructions to describe the entity.

## 3. Label Schema in `response.jsonl`
Each response contains a list of span-level `labels`:
- `label_type`:
  - `Evident Conflict`: Explicit factual contradiction of source context.
  - `Evident Baseless Info`: Fact introduced with zero basis in source context.
  - `Subtle Baseless Info`: Minor unwarranted extrapolation.
  - `Subtle Conflict`: Nuanced contradiction or temporal distortion.
- Fields per label:
  - `start` (int): Character start offset in `response`
  - `end` (int): Character end offset in `response`
  - `text` (str): Annotated hallucinated snippet
  - `meta` (str): Human annotator justification and source reference

## 4. Evaluation Mapping Strategy
- **Answer-Level Ground Truth**:
  - Hallucinated (`is_hallucinated = True`): `len(labels) > 0`
  - Faithful (`is_hallucinated = False`): `len(labels) == 0`
  - In `test` (quality='good'): 35.3% are hallucinated (943 positive cases), 64.7% are faithful (1,732 negative cases).
- **Claim-Level Ground Truth**:
  - A claim extracted from `response` is considered hallucinated if it character-overlaps with any annotated span in `labels`.
  - Otherwise, the claim is considered grounded/supported by ground truth.
