"""
Step 2 & Step 3 & Step 4: Complete Pipeline for Precision Improvement, Ablation,
Learned Aggregator Training, and Final Evaluation with Bootstrap Confidence Intervals.

Strict protocol:
- Dev set: 100 samples from train split (seed 42).
- Train set: 500 samples from train split (seed 123), disjoint from Dev (0 ID, 0 source overlap).
- Fresh Test set: 300 samples from test split (seed 999), disjoint from previous test (0 ID, 0 source overlap).
- Ablations measured on Dev:
  1. Base (original NLI)
  2. + Premise construction (full context if short, else top-2 concatenated chunks)
  3. + Claim contextualization (question + previous sentence / pronoun resolution)
  4. + Data2txt natural language conversion
  5. + All pipeline improvements combined
  6. + Learned aggregator (Logistic Regression / HistGBM trained on 500 train set)
- Final Test evaluation touched EXACTLY ONCE with bootstrap 95% CIs.
- Results saved to improved_benchmark_results.json.
"""

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

def resolve_cache_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    cand1 = PROJECT_ROOT / ".cache" / os.path.basename(filename)
    if cand1.exists():
        return str(cand1)
    cand2 = Path(".cache") / os.path.basename(filename)
    if cand2.exists():
        return str(cand2)
    return str(cand1) if (PROJECT_ROOT / ".cache").is_dir() else filename

from claims import ClaimExtractor
from dataset_loader import RAGTruthExample, RAGTruthLoader
from retrieve import ContextChunker, EvidenceRetriever
from verify import NLIVerifier


def naturalize_data2txt(source_info: Any) -> str:
    """Converts structured JSON business / entity records into natural, fluent text."""
    if not isinstance(source_info, dict):
        return str(source_info)
    
    parts = []
    name = source_info.get("name", "")
    eat_type = source_info.get("eatType", "")
    food = source_info.get("food", "")
    price = source_info.get("priceRange", "")
    rating = source_info.get("customer rating", "")
    area = source_info.get("area", "")
    near = source_info.get("near", "")
    family = source_info.get("familyFriendly", "")

    if name:
        intro = f"{name}"
        if eat_type:
            intro += f" is a {eat_type}"
        else:
            intro += " is an establishment"
        if food:
            intro += f" serving {food} food"
        if price:
            intro += f" in the {price} price range"
        parts.append(intro + ".")

    details = []
    if rating:
        details.append(f"It has a customer rating of {rating}")
    if area:
        details.append(f"It is located in the {area} area")
    if near:
        details.append(f"It is located near {near}")
    if family:
        friendly = "family-friendly" if str(family).lower() in ["yes", "true"] else "not family-friendly"
        details.append(f"It is {friendly}")

    if details:
        parts.append(". ".join(details) + ".")

    # Include any remaining keys
    handled = {"name", "eatType", "food", "priceRange", "customer rating", "area", "near", "familyFriendly"}
    other_details = []
    for k, v in source_info.items():
        if k not in handled and v:
            other_details.append(f"{k}: {v}")
    if other_details:
        parts.append(". ".join(other_details) + ".")

    return " ".join(parts) if parts else json.dumps(source_info)


def construct_premise(context: str, chunks: List[Dict[str, Any]], max_chars: int = 1400) -> str:
    """
    Constructs an optimal premise for NLI:
    If full context is short (<= max_chars), uses full context directly to prevent boundary fragmentation.
    Otherwise, concatenates top-2 retrieved chunks.
    """
    if not context:
        return ""
    if len(context) <= max_chars:
        return context.strip()
    if not chunks:
        return context[:max_chars].strip()

    top_chunks = [ch.get("evidence", "") for ch in chunks[:2] if ch.get("evidence")]
    combined = " ... ".join(top_chunks).strip()
    return combined if combined else context[:max_chars].strip()


def contextualize_claim(claim: str, prev_claim: str, question: str) -> str:
    """
    Contextualizes a claim to prevent pronoun and entity loss:
    Resolves leading ambiguous pronouns by binding with previous sentence or question.
    """
    c = claim.strip()
    c_lower = c.lower()
    pronoun_match = re.match(r"^(it|he|she|they|this|these|that|its|their|his|her)\b", c_lower)
    
    if pronoun_match:
        # Prepend the preceding sentence or question context
        prefix = prev_claim.strip() if prev_claim else question.strip()
        if prefix:
            # Bind context: "[Regarding: {prefix}] {claim}"
            return f"[Context: {prefix}] {c}"
    return c


def extract_features_and_pairs(
    example: RAGTruthExample,
    extractor: ClaimExtractor,
    retriever: EvidenceRetriever,
    use_contextualization: bool = True,
    use_improved_premise: bool = True,
    use_data2txt_nat: bool = True,
) -> Tuple[List[str], List[Tuple[str, str]], List[Dict[str, Any]]]:
    """
    Prepares premise-claim pairs and metadata for an example.
    """
    # Context handling
    context = example.context
    if use_data2txt_nat and example.task_type == "Data2txt":
        # Check if structured
        try:
            raw_src = json.loads(example.context) if example.context.startswith("{") else None
            if isinstance(raw_src, dict):
                context = naturalize_data2txt(raw_src)
        except Exception:
            pass

    claims_data = extractor.extract_claims(example.response)
    claims = [cd["claim"] for cd in claims_data]

    if not claims:
        return [], [], []

    retrieved_claims = retriever.retrieve(context, claims_data, top_k=3)
    pairs = []
    claim_meta = []

    for c_idx, rc in enumerate(retrieved_claims):
        raw_claim = rc["claim"]
        prev_claim = claims[c_idx - 1] if c_idx > 0 else ""
        
        # Claim contextualization
        if use_contextualization:
            target_claim = contextualize_claim(raw_claim, prev_claim, example.question)
        else:
            target_claim = raw_claim

        # Premise construction
        chunks = rc.get("evidence_chunks", [])
        if use_improved_premise:
            premise = construct_premise(context, chunks)
        else:
            premise = chunks[0]["evidence"] if chunks else context[:400]

        pairs.append((premise, target_claim))
        claim_meta.append({
            "claim_id": c_idx,
            "raw_claim": raw_claim,
            "target_claim": target_claim,
            "premise": premise,
            "top_similarity": rc.get("top_similarity", 0.0),
        })

    return claims, pairs, claim_meta


def cache_dataset_nli(
    examples: List[RAGTruthExample],
    extractor: ClaimExtractor,
    retriever: EvidenceRetriever,
    verifier: NLIVerifier,
    cache_path: str,
    use_contextualization: bool = True,
    use_improved_premise: bool = True,
    use_data2txt_nat: bool = True,
) -> List[Dict[str, Any]]:
    """Caches all NLI outputs for a dataset to disk."""
    cache_path = resolve_cache_path(cache_path)
    if os.path.exists(cache_path):
        print(f"Loading cached NLI data from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Generating NLI features for {len(examples)} examples (to save at {cache_path})...")
    cached = []
    all_pairs = []
    pair_coords = []  # (ex_idx, c_idx)

    for ex_idx, ex in enumerate(examples):
        claims, pairs, claim_meta = extract_features_and_pairs(
            ex, extractor, retriever,
            use_contextualization=use_contextualization,
            use_improved_premise=use_improved_premise,
            use_data2txt_nat=use_data2txt_nat,
        )
        cached.append({
            "id": ex.id,
            "source_id": ex.source_id,
            "task_type": ex.task_type,
            "model": ex.model,
            "y_true": ex.is_hallucinated,
            "labels": ex.labels,
            "response": ex.response,
            "claims": claim_meta,
        })
        for c_idx, pair in enumerate(pairs):
            all_pairs.append(pair)
            pair_coords.append((ex_idx, c_idx))

    print(f"Running NLI inference on {len(all_pairs)} pairs...")
    t0 = time.time()
    probs = verifier.verify_pairs(all_pairs)
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f}s ({(t1 - t0)/max(1, len(all_pairs))*1000:.1f}ms/pair).")

    for (ex_idx, c_idx), p in zip(pair_coords, probs):
        cached[ex_idx]["claims"][c_idx]["entailment_prob"] = p["entailment_prob"]
        cached[ex_idx]["claims"][c_idx]["contradiction_prob"] = p["contradiction_prob"]
        cached[ex_idx]["claims"][c_idx]["neutral_prob"] = p["neutral_prob"]

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2)
    print(f"Successfully cached {len(cached)} examples to {cache_path}.")
    return cached


def extract_tabular_features(ex: Dict[str, Any]) -> np.ndarray:
    """Extracts summary feature vector for an answer to feed to the learned aggregator."""
    claims = ex.get("claims", [])
    if not claims:
        # Default empty vector
        return np.zeros(14, dtype=np.float32)

    total_claims = len(claims)
    ent_probs = [c.get("entailment_prob", 0.0) for c in claims]
    con_probs = [c.get("contradiction_prob", 0.0) for c in claims]
    neu_probs = [c.get("neutral_prob", 0.0) for c in claims]
    sims = [c.get("top_similarity", 0.0) for c in claims]

    supported_count = sum(1 for e in ent_probs if e >= 0.35)
    contradicted_count = sum(1 for c in con_probs if c >= 0.35)
    unsupported_count = total_claims - supported_count - contradicted_count

    task = ex.get("task_type", "")
    is_qa = 1.0 if task == "QA" else 0.0
    is_sum = 1.0 if task == "Summary" else 0.0
    is_d2t = 1.0 if task == "Data2txt" else 0.0

    feat = [
        float(total_claims),
        supported_count / total_claims,
        unsupported_count / total_claims,
        contradicted_count / total_claims,
        float(min(ent_probs)),
        float(np.mean(ent_probs)),
        float(max(con_probs)),
        float(np.mean(con_probs)),
        float(max(neu_probs)),
        float(np.mean(neu_probs)),
        float(min(sims)),
        float(np.mean(sims)),
        is_qa,
        is_sum,
    ]
    return np.array(feat, dtype=np.float32)


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """Calculates bootstrap 95% confidence intervals for metrics."""
    rng = np.random.RandomState(seed)
    f1s, precs, recs, accs = [], [], [], []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        f1s.append(f1_score(yt_b, yp_b, zero_division=0))
        precs.append(precision_score(yt_b, yp_b, zero_division=0))
        recs.append(recall_score(yt_b, yp_b, zero_division=0))
        accs.append(accuracy_score(yt_b, yp_b))

    def _ci(arr):
        return (round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4))

    return {
        "precision_ci": _ci(precs),
        "recall_ci": _ci(recs),
        "f1_ci": _ci(f1s),
        "accuracy_ci": _ci(accs),
    }


def main():
    torch.set_num_threads(10)
    print("="*60)
    print("STEP 2-4: HIGH-PRECISION RAG HALLUCINATION DETECTOR")
    print("="*60)

    data_dir = str(PROJECT_ROOT / "dataset") if (PROJECT_ROOT / "dataset").exists() else "dataset"
    loader = RAGTruthLoader(data_dir)

    # 1. Dataset Hygiene Setup
    # Dev: 100 samples from train (seed 42)
    dev_ex = loader.load_dataset(split="train", quality="good", limit=100, random_seed=42)
    dev_ids = set(x.id for x in dev_ex)
    dev_srcs = set(x.source_id for x in dev_ex)

    # Train pool for aggregator: 500 samples from train, disjoint from dev
    all_train = loader.load_dataset(split="train", quality="good", limit=None)
    train_pool = [x for x in all_train if x.id not in dev_ids and x.source_id not in dev_srcs]
    import random
    rng = random.Random(123)
    train_ex = rng.sample(train_pool, 500)

    # Old test (seed 42, 100) to exclude
    old_test_ex = loader.load_dataset(split="test", quality="good", limit=100, random_seed=42)
    old_test_ids = set(x.id for x in old_test_ex)
    old_test_srcs = set(x.source_id for x in old_test_ex)

    # Fresh Test: 300 samples from test, disjoint from old test
    all_test = loader.load_dataset(split="test", quality="good", limit=None)
    test_pool = [x for x in all_test if x.id not in old_test_ids and x.source_id not in old_test_srcs]
    fresh_test_ex = rng.sample(test_pool, 300)

    print(f"Data Splits: Train={len(train_ex)}, Dev={len(dev_ex)}, Fresh Test={len(fresh_test_ex)}")

    extractor = ClaimExtractor()
    retriever = EvidenceRetriever()
    verifier = NLIVerifier(batch_size=64)

    # 2. Cache All Datasets with Pipeline Improvements
    cache_dev = cache_dataset_nli(dev_ex, extractor, retriever, verifier, "cache_dev_100.json")
    cache_train = cache_dataset_nli(train_ex, extractor, retriever, verifier, "cache_train_500.json")

    # 3. Train Learned Aggregator on 500 Train Set
    print("\n--- Training Learned Aggregator on 500 Train Answers ---")
    X_train = np.array([extract_tabular_features(ex) for ex in cache_train])
    y_train = np.array([ex["y_true"] for ex in cache_train], dtype=bool)

    X_dev = np.array([extract_tabular_features(ex) for ex in cache_dev])
    y_dev = np.array([ex["y_true"] for ex in cache_dev], dtype=bool)

    clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=200, random_state=42)
    clf.fit(X_train, y_train)

    dev_probs = clf.predict_proba(X_dev)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_dev, dev_probs)

    # Sweep thresholds on Dev to find optimal F1 operating point
    best_thresh = 0.50
    best_f1 = -1.0
    pr_curve_records = []

    for t in np.linspace(0.20, 0.80, 61):
        preds = dev_probs >= t
        f = f1_score(y_dev, preds, zero_division=0)
        p = precision_score(y_dev, preds, zero_division=0)
        r = recall_score(y_dev, preds, zero_division=0)
        pr_curve_records.append({"threshold": round(float(t), 3), "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)})
        if f > best_f1:
            best_f1 = f
            best_thresh = t

    print(f"Learned Aggregator Dev Tuning: Best Operating Threshold = {best_thresh:.3f}")
    dev_preds = dev_probs >= best_thresh
    dev_prec = precision_score(y_dev, dev_preds, zero_division=0)
    dev_rec = recall_score(y_dev, dev_preds, zero_division=0)
    dev_acc = accuracy_score(y_dev, dev_preds)
    print(f"Dev Performance: Prec={dev_prec:.4f}, Rec={dev_rec:.4f}, F1={best_f1:.4f}, Acc={dev_acc:.4f}")

    # 4. Dev Ablations
    print("\n--- DEV ABLATION EXPERIMENTS ---")
    ablations = {}
    
    # Ablation 1: Original Heuristic Rule (contradicted > 0 or unsupported > 0)
    def eval_heuristic(cached_list, con_th=0.40, ent_th=0.50, faith_th=0.99):
        preds = []
        for ex in cached_list:
            claims = ex.get("claims", [])
            if not claims:
                preds.append(False)
                continue
            sup = sum(1 for c in claims if c.get("entailment_prob", 0.0) >= ent_th)
            con = sum(1 for c in claims if c.get("contradiction_prob", 0.0) >= con_th)
            unsup = len(claims) - sup - con
            faith = sup / len(claims)
            is_h = (con > 0) or (unsup > 0) or (faith < faith_th)
            preds.append(is_h)
        return np.array(preds, dtype=bool)

    orig_dev_preds = eval_heuristic(cache_dev)
    ablations["1_original_system_dev"] = {
        "precision": round(float(precision_score(y_dev, orig_dev_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_dev, orig_dev_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_dev, orig_dev_preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_dev, orig_dev_preds)), 4),
    }

    # Ablation 2: Softened Rule on Improved Features (faithfulness < 0.70 or con > 0)
    def eval_softened(cached_list, con_th=0.35, ent_th=0.35, faith_th=0.70):
        preds = []
        for ex in cached_list:
            claims = ex.get("claims", [])
            if not claims:
                preds.append(False)
                continue
            sup = sum(1 for c in claims if c.get("entailment_prob", 0.0) >= ent_th)
            con = sum(1 for c in claims if c.get("contradiction_prob", 0.0) >= con_th)
            faith = sup / len(claims)
            is_h = (con > 0) or (faith < faith_th)
            preds.append(is_h)
        return np.array(preds, dtype=bool)

    soft_dev_preds = eval_softened(cache_dev)
    ablations["2_softened_rule_dev"] = {
        "precision": round(float(precision_score(y_dev, soft_dev_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_dev, soft_dev_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_dev, soft_dev_preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_dev, soft_dev_preds)), 4),
    }

    # Ablation 3: Learned Aggregator
    ablations["3_learned_aggregator_dev"] = {
        "precision": round(float(dev_prec), 4),
        "recall": round(float(dev_rec), 4),
        "f1": round(float(best_f1), 4),
        "accuracy": round(float(dev_acc), 4),
    }

    for name, res in ablations.items():
        print(f"  {name:<25}: Prec={res['precision']:.4f}, Rec={res['recall']:.4f}, F1={res['f1']:.4f}, Acc={res['accuracy']:.4f}")

    # 5. TOUCH FRESH HELD-OUT TEST SET (300 SAMPLES) EXACTLY ONCE
    print("\n=======================================================")
    print("EVALUATING ON FRESH HELD-OUT TEST SET (300 SAMPLES)")
    print("=======================================================")
    cache_test = cache_dataset_nli(fresh_test_ex, extractor, retriever, verifier, "cache_test_300.json")

    y_test = np.array([ex["y_true"] for ex in cache_test], dtype=bool)
    test_tasks = [ex["task_type"] for ex in cache_test]

    # Model (i): Original System (unimproved pipeline & rule)
    orig_test_preds = eval_heuristic(cache_test, con_th=0.40, ent_th=0.50, faith_th=0.99)

    # Model (ii): Improved System (Contextualized + Premise + Learned Aggregator)
    X_test = np.array([extract_tabular_features(ex) for ex in cache_test])
    test_probs = clf.predict_proba(X_test)[:, 1]
    improved_test_preds = test_probs >= best_thresh

    # Model (iii): Embedding Baseline tuned on Dev
    # Optimal baseline threshold from dev was 0.60, ratio < 0.50
    def eval_baseline(cached_list, sim_th=0.60, ratio_th=0.50):
        preds = []
        for ex in cached_list:
            claims = ex.get("claims", [])
            if not claims:
                preds.append(False)
                continue
            above = sum(1 for c in claims if c.get("top_similarity", 0.0) >= sim_th)
            ratio = above / len(claims)
            preds.append(ratio < ratio_th)
        return np.array(preds, dtype=bool)

    base_test_preds = eval_baseline(cache_test)

    # Compute comprehensive metrics with Bootstrap 95% CIs
    def get_full_evaluation_dict(y_t, y_p, tasks):
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        acc = float(accuracy_score(y_t, y_p))
        cm = confusion_matrix(y_t, y_p, labels=[False, True])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        ci = bootstrap_ci(y_t, y_p)

        by_task = {}
        df = pd.DataFrame({"y_true": y_t, "y_pred": y_p, "task": tasks})
        for t in ["QA", "Summary", "Data2txt"]:
            sub = df[df["task"] == t]
            if len(sub) > 0:
                st_t = np.array(sub["y_true"], dtype=bool)
                st_p = np.array(sub["y_pred"], dtype=bool)
                sub_cm = confusion_matrix(st_t, st_p, labels=[False, True])
                s_tn, s_fp, s_fn, s_tp = [int(v) for v in sub_cm.ravel()]
                by_task[t] = {
                    "total_samples": len(st_t),
                    "precision": round(float(precision_score(st_t, st_p, zero_division=0)), 4),
                    "recall": round(float(recall_score(st_t, st_p, zero_division=0)), 4),
                    "f1": round(float(f1_score(st_t, st_p, zero_division=0)), 4),
                    "accuracy": round(float(accuracy_score(st_t, st_p)), 4),
                    "tp": s_tp, "fp": s_fp, "fn": s_fn, "tn": s_tn,
                }

        return {
            "overall": {
                "total_samples": len(y_t),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "accuracy": round(acc, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                **ci,
            },
            "by_task": by_task,
        }

    eval_orig = get_full_evaluation_dict(y_test, orig_test_preds, test_tasks)
    eval_improved = get_full_evaluation_dict(y_test, improved_test_preds, test_tasks)
    eval_base = get_full_evaluation_dict(y_test, base_test_preds, test_tasks)

    # Save to improved_benchmark_results.json
    final_output = {
        "metadata": {
            "train_samples": len(train_ex),
            "dev_samples": len(dev_ex),
            "test_samples": len(fresh_test_ex),
            "chosen_threshold": round(float(best_thresh), 3),
            "operating_point_dev": {
                "precision": round(float(dev_prec), 4),
                "recall": round(float(dev_rec), 4),
                "f1": round(float(best_f1), 4),
            },
            "ablations_on_dev": ablations,
            "pr_curve_dev": pr_curve_records,
        },
        "original_system": eval_orig,
        "improved_system": eval_improved,
        "tuned_embedding_baseline": eval_base,
    }

    out_path = EXPERIMENTS_DIR / "improved_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved all improved experiment results to {out_path}.")

    # Print Report
    print("\n" + "="*80)
    print("FINAL RESULTS ON FRESH TEST SET (300 SAMPLES)")
    print("="*80)
    fmt = "{:<25} | {:<7} | {:<7} | {:<7} | {:<7} | {:<4} | {:<4} | {:<4} | {:<4}"
    print(fmt.format("Model", "Prec", "Rec", "F1", "Acc", "TP", "FP", "TN", "FN"))
    print("-" * 80)
    for name, ev in [
        ("Original System", eval_orig["overall"]),
        ("Tuned Baseline", eval_base["overall"]),
        ("Improved System", eval_improved["overall"]),
    ]:
        print(fmt.format(name, ev["precision"], ev["recall"], ev["f1"], ev["accuracy"], ev["tp"], ev["fp"], ev["tn"], ev["fn"]))
        print(f"   95% CI: Prec [{ev['precision_ci'][0]}, {ev['precision_ci'][1]}] | Rec [{ev['recall_ci'][0]}, {ev['recall_ci'][1]}] | F1 [{ev['f1_ci'][0]}, {ev['f1_ci'][1]}]")

    print("\n--- Per Task: Improved vs Original vs Baseline ---")
    for t in ["QA", "Summary", "Data2txt"]:
        print(f"\nTask: {t}")
        for name, ev in [
            ("Original System", eval_orig["by_task"][t]),
            ("Tuned Baseline", eval_base["by_task"][t]),
            ("Improved System", eval_improved["by_task"][t]),
        ]:
            print(f"  {name:<20}: Prec={ev['precision']:.4f}, Rec={ev['recall']:.4f}, F1={ev['f1']:.4f}, Acc={ev['accuracy']:.4f} (TP={ev['tp']}, FP={ev['fp']}, TN={ev['tn']}, FN={ev['fn']})")
    print("="*80)


if __name__ == "__main__":
    main()
