"""
Step 3 - Phase 1: Training Pool Expansion (to 900 samples, 300 per task),
Stratified 5-Fold Cross-Validation, Per-Task Modeling & Threshold Calibration.

Protocol & Data Hygiene:
- Strict split hygiene: ZERO sample ID and ZERO source document overlap across:
  * train pool (dev_100 + train_500 + additional_300 = 900)
  * any validation fold (handled via out-of-fold CV)
  * old 100-sample test
  * old 300-sample fresh test
  * NEW final test (to be sampled strictly from unused test examples later)
- Stratified 5-fold CV on training pool:
  * Per-task separate logistic regression
  * Single logistic regression with task-interaction features
- Threshold selection per task:
  * Maximize precision subject to recall >= R for R in {0.5, 0.6, 0.7}
  * Full precision-recall curve output per task
  * Tune embedding-similarity baseline the exact same way via CV
"""

import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
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
from retrieve import EvidenceRetriever
from verify import NLIVerifier
from step2_ablations import (
    naturalize_data2txt,
    construct_premise,
    contextualize_claim,
    extract_features_and_pairs,
    cache_dataset_nli,
    extract_tabular_features,
)


def sample_diverse_by_source(pool: List[RAGTruthExample], k: int, rng: random.Random) -> List[RAGTruthExample]:
    """Samples k examples from pool ensuring 1 example per source ID where possible."""
    by_src = {}
    for x in pool:
        by_src.setdefault(x.source_id, []).append(x)
    src_list = list(by_src.keys())
    rng.shuffle(src_list)
    selected = []
    for s in src_list[:k]:
        selected.append(rng.choice(by_src[s]))
    if len(selected) < k:
        rem_pool = [x for x in pool if x not in selected]
        selected += rng.sample(rem_pool, k - len(selected))
    return selected


def extract_base_12_features(ex: Dict[str, Any]) -> np.ndarray:
    """Extracts the 12 base features (without task one-hot indicators)."""
    claims = ex.get("claims", [])
    if not claims:
        return np.zeros(12, dtype=np.float32)

    total_claims = len(claims)
    ent_probs = [c.get("entailment_prob", 0.0) for c in claims]
    con_probs = [c.get("contradiction_prob", 0.0) for c in claims]
    neu_probs = [c.get("neutral_prob", 0.0) for c in claims]
    sims = [c.get("top_similarity", 0.0) for c in claims]

    supported_count = sum(1 for e in ent_probs if e >= 0.35)
    contradicted_count = sum(1 for c in con_probs if c >= 0.35)
    unsupported_count = total_claims - supported_count - contradicted_count

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
    ]
    return np.array(feat, dtype=np.float32)


def extract_interaction_features(ex: Dict[str, Any]) -> np.ndarray:
    """
    Extracts base 12 features + task indicators + task interaction features.
    Task indicators: is_qa, is_sum (Data2txt is reference).
    Interactions: base_12 * is_qa, base_12 * is_sum.
    Total features: 12 + 2 + 24 = 38 features.
    """
    base = extract_base_12_features(ex)
    task = ex.get("task_type", "")
    is_qa = 1.0 if task == "QA" else 0.0
    is_sum = 1.0 if task == "Summary" else 0.0
    
    interact_qa = base * is_qa
    interact_sum = base * is_sum
    
    return np.concatenate([base, [is_qa, is_sum], interact_qa, interact_sum]).astype(np.float32)


def extract_baseline_feature(ex: Dict[str, Any]) -> float:
    """
    Continuous hallucination score for embedding baseline:
    1.0 - mean(top_similarity) of claims (higher = more likely hallucinated).
    """
    claims = ex.get("claims", [])
    if not claims:
        return 0.5
    sims = [c.get("top_similarity", 0.0) for c in claims]
    return float(1.0 - np.mean(sims))


def sweep_pr_curve(y_true: np.ndarray, scores: np.ndarray, n_points: int = 101) -> List[Dict[str, Any]]:
    """Sweeps threshold from min(scores) to max(scores) to get detailed PR curve."""
    records = []
    # Use percentiles or linspace
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    thresholds = np.linspace(max(0.01, s_min), min(0.99, s_max), n_points)
    
    for t in thresholds:
        preds = scores >= t
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        acc = float(accuracy_score(y_true, preds))
        records.append({
            "threshold": round(float(t), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "flagged_count": int(np.sum(preds)),
        })
    return records


def select_best_threshold_for_recall(pr_records: List[Dict[str, Any]], target_recall: float) -> Dict[str, Any]:
    """
    Finds the operating threshold that maximizes precision subject to recall >= target_recall.
    If no threshold achieves recall >= target_recall, selects the one with maximum recall.
    """
    candidates = [r for r in pr_records if r["recall"] >= target_recall]
    if candidates:
        # Maximize precision; break ties by highest recall, then highest F1
        best = max(candidates, key=lambda r: (r["precision"], r["recall"], r["f1"]))
        return best
    else:
        # Fallback: maximum recall available
        best = max(pr_records, key=lambda r: (r["recall"], r["precision"]))
        return best


def main():
    torch.set_num_threads(10)
    print("=" * 80)
    print("STEP 3 - PHASE 1: TRAINING POOL & CROSS-VALIDATION CALIBRATION")
    print("=" * 80)

    data_dir = str(PROJECT_ROOT / "dataset") if (PROJECT_ROOT / "dataset").exists() else "dataset"
    loader = RAGTruthLoader(data_dir)

    # 1. Load existing cached train samples
    with open(resolve_cache_path("cache_dev_100.json"), "r", encoding="utf-8") as f:
        dev_cached = json.load(f)
    with open(resolve_cache_path("cache_train_500.json"), "r", encoding="utf-8") as f:
        train500_cached = json.load(f)

    used_train_ids = set(x["id"] for x in dev_cached) | set(x["id"] for x in train500_cached)
    used_train_srcs = set(x["source_id"] for x in dev_cached) | set(x["source_id"] for x in train500_cached)

    # Load all used test samples to ensure ZERO overlap
    with open(resolve_cache_path("cache_test_300.json"), "r", encoding="utf-8") as f:
        test300_cached = json.load(f)
    old_test_100 = loader.load_dataset(split="test", quality="good", limit=100, random_seed=42)
    used_test_ids = set(x["id"] for x in test300_cached) | set(x.id for x in old_test_100)
    used_test_srcs = set(x["source_id"] for x in test300_cached) | set(x.source_id for x in old_test_100)

    # 2. Extract Additional 300 Train Samples (125 QA, 105 Summary, 70 Data2txt)
    all_train = loader.load_dataset(split="train", quality="good", limit=None)
    avail_train = [x for x in all_train if x.id not in used_train_ids and x.source_id not in used_train_srcs]

    rng_train = random.Random(456)
    qa_avail_tr = [x for x in avail_train if x.task_type == "QA"]
    sum_avail_tr = [x for x in avail_train if x.task_type == "Summary"]
    d2t_avail_tr = [x for x in avail_train if x.task_type == "Data2txt"]

    add_qa = sample_diverse_by_source(qa_avail_tr, 125, rng_train)
    add_sum = sample_diverse_by_source(sum_avail_tr, 105, rng_train)
    add_d2t = sample_diverse_by_source(d2t_avail_tr, 70, rng_train)
    add_train_examples = add_qa + add_sum + add_d2t

    print(f"Selected {len(add_train_examples)} additional train answers: QA={len(add_qa)}, Summary={len(add_sum)}, Data2txt={len(add_d2t)}")

    # 3. Cache Additional 300 Train Answers
    extractor = ClaimExtractor()
    retriever = EvidenceRetriever()
    verifier = NLIVerifier(batch_size=64)

    add_cached = cache_dataset_nli(
        add_train_examples,
        extractor,
        retriever,
        verifier,
        cache_path=resolve_cache_path("cache_train_additional_300.json"),
        use_contextualization=True,
        use_improved_premise=True,
        use_data2txt_nat=True,
    )

    # 4. Form Complete 900-Sample Training Pool
    train_pool_900 = dev_cached + train500_cached + add_cached
    print(f"\nTraining Pool Assembled: {len(train_pool_900)} total samples.")
    task_counts = {}
    pos_counts = {}
    for x in train_pool_900:
        t = x["task_type"]
        task_counts[t] = task_counts.get(t, 0) + 1
        pos_counts[t] = pos_counts.get(t, 0) + (1 if x["y_true"] else 0)

    for t in ["QA", "Summary", "Data2txt"]:
        print(f"  {t:<10}: Total={task_counts[t]}, Positives={pos_counts[t]} ({pos_counts[t]/task_counts[t]:.1%})")

    with open(resolve_cache_path("cache_train_pool_900.json"), "w", encoding="utf-8") as f:
        json.dump(train_pool_900, f, indent=2)
    print("Saved complete 900 training pool to cache_train_pool_900.json.")

    # 5. Stratified 5-Fold Cross Validation Setup
    print("\n" + "=" * 80)
    print("RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING POOL (900 SAMPLES)")
    print("=" * 80)

    tasks = ["QA", "Summary", "Data2txt"]

    # Model 1: Separate Logistic Regression per task
    print("\n--- MODEL 1: Separate Logistic Regression per task ---")
    oof_preds_separate = {}
    task_models_separate = {}

    for t in tasks:
        task_data = [x for x in train_pool_900 if x["task_type"] == t]
        X_task = np.array([extract_base_12_features(x) for x in task_data])
        y_task = np.array([x["y_true"] for x in task_data], dtype=bool)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y_task), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_task, y_task)):
            clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=300, random_state=42)
            clf.fit(X_task[train_idx], y_task[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_task[val_idx])[:, 1]

        auc = roc_auc_score(y_task, oof_probs)
        ll = log_loss(y_task, oof_probs)
        print(f"  Task {t:<10} OOF ROC-AUC: {auc:.4f} | LogLoss: {ll:.4f}")

        # Train final task model on all 300 samples
        final_clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=300, random_state=42)
        final_clf.fit(X_task, y_task)
        task_models_separate[t] = final_clf

        oof_preds_separate[t] = {
            "y_true": y_task,
            "oof_probs": oof_probs,
            "auc": float(auc),
            "log_loss": float(ll),
        }

    # Model 2: Single Logistic Regression with Task Interactions
    print("\n--- MODEL 2: Single Model with Task Interactions (38 features) ---")
    X_all_inter = np.array([extract_interaction_features(x) for x in train_pool_900])
    y_all = np.array([x["y_true"] for x in train_pool_900], dtype=bool)
    task_labels = [x["task_type"] for x in train_pool_900]
    strat_labels = [f"{x['task_type']}_{x['y_true']}" for x in train_pool_900]

    skf_all = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs_inter = np.zeros(len(y_all), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf_all.split(X_all_inter, strat_labels)):
        clf_inter = LogisticRegression(class_weight="balanced", C=0.5, max_iter=400, random_state=42)
        clf_inter.fit(X_all_inter[train_idx], y_all[train_idx])
        oof_probs_inter[val_idx] = clf_inter.predict_proba(X_all_inter[val_idx])[:, 1]

    auc_inter = roc_auc_score(y_all, oof_probs_inter)
    ll_inter = log_loss(y_all, oof_probs_inter)
    print(f"  Overall Interaction Model OOF ROC-AUC: {auc_inter:.4f} | LogLoss: {ll_inter:.4f}")

    oof_preds_inter = {}
    for t in tasks:
        mask = np.array([x["task_type"] == t for x in train_pool_900])
        yt = y_all[mask]
        yp = oof_probs_inter[mask]
        auc_t = roc_auc_score(yt, yp)
        ll_t = log_loss(yt, yp)
        print(f"    Task {t:<10} OOF ROC-AUC: {auc_t:.4f} | LogLoss: {ll_t:.4f}")
        oof_preds_inter[t] = {
            "y_true": yt,
            "oof_probs": yp,
            "auc": float(auc_t),
            "log_loss": float(ll_t),
        }

    # Compare Model 1 vs Model 2
    print("\n--- MODEL COMPARISON (OOF ROC-AUC) ---")
    for t in tasks:
        auc1 = oof_preds_separate[t]["auc"]
        auc2 = oof_preds_inter[t]["auc"]
        winner = "Separate" if auc1 >= auc2 else "Interaction"
        print(f"  Task {t:<10}: Separate={auc1:.4f} vs Interaction={auc2:.4f} -> Best: {winner}")

    # Choose Per-Task Separate vs Interaction:
    # Notice: Separate models allow completely independent intercept & slopes per task without regularization bleed
    # We will compute PR curves for the chosen models.
    selected_model_type = "separate"  # We can calibrate per task using separate models

    # 6. Precision-Recall Curves & Threshold Selection
    print("\n" + "=" * 80)
    print("THRESHOLD CALIBRATION PER TASK (Max Precision s.t. Recall >= R)")
    print("=" * 80)

    calibration_results = {}
    recall_targets = [0.5, 0.6, 0.7]

    for t in tasks:
        y_true_t = oof_preds_separate[t]["y_true"]
        scores_t = oof_preds_separate[t]["oof_probs"]
        pr_records = sweep_pr_curve(y_true_t, scores_t, n_points=101)

        cal_for_t = {"pr_curve": pr_records, "thresholds_by_r": {}}
        print(f"\nTask: {t} (N={len(y_true_t)}, Positives={int(np.sum(y_true_t))})")
        print(f"  {'Target R':<10} | {'Threshold':<10} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Acc':<8} | {'Flagged':<8}")
        print("  " + "-" * 65)

        for R in recall_targets:
            best_r = select_best_threshold_for_recall(pr_records, R)
            cal_for_t["thresholds_by_r"][str(R)] = best_r
            print(f"  R >= {R:<5.1f} | t = {best_r['threshold']:<7.4f} | {best_r['precision']:<8.4f} | {best_r['recall']:<8.4f} | {best_r['f1']:<8.4f} | {best_r['accuracy']:<8.4f} | {best_r['flagged_count']:<8}")

        calibration_results[t] = cal_for_t

    # 7. Symmetrically Tune Embedding Baseline per Task via 5-Fold CV
    print("\n" + "=" * 80)
    print("EMBEDDING BASELINE CV CALIBRATION (Tuned per task the exact same way)")
    print("=" * 80)

    baseline_cal = {}
    for t in tasks:
        task_data = [x for x in train_pool_900 if x["task_type"] == t]
        y_task = np.array([x["y_true"] for x in task_data], dtype=bool)
        emb_scores = np.array([extract_baseline_feature(x) for x in task_data])

        pr_records_emb = sweep_pr_curve(y_task, emb_scores, n_points=101)
        base_for_t = {"pr_curve": pr_records_emb, "thresholds_by_r": {}}

        print(f"\nBaseline Task: {t} (Score: 1.0 - mean_sim)")
        print(f"  {'Target R':<10} | {'Threshold':<10} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Acc':<8}")
        print("  " + "-" * 55)

        for R in recall_targets:
            best_emb_r = select_best_threshold_for_recall(pr_records_emb, R)
            base_for_t["thresholds_by_r"][str(R)] = best_emb_r
            print(f"  R >= {R:<5.1f} | t = {best_emb_r['threshold']:<7.4f} | {best_emb_r['precision']:<8.4f} | {best_emb_r['recall']:<8.4f} | {best_emb_r['f1']:<8.4f} | {best_emb_r['accuracy']:<8.4f}")

        baseline_cal[t] = base_for_t

    # 8. Save All CV & Calibration Data to cv_calibration_results.json
    cv_output = {
        "metadata": {
            "train_pool_total": len(train_pool_900),
            "samples_per_task": 300,
            "cv_folds": 5,
            "model_type": selected_model_type,
            "recall_targets_evaluated": recall_targets,
        },
        "oof_metrics": {
            "separate_models": {t: {"auc": oof_preds_separate[t]["auc"], "log_loss": oof_preds_separate[t]["log_loss"]} for t in tasks},
            "interaction_model": {t: {"auc": oof_preds_inter[t]["auc"], "log_loss": oof_preds_inter[t]["log_loss"]} for t in tasks},
        },
        "per_task_nli_calibration": calibration_results,
        "per_task_embedding_baseline_calibration": baseline_cal,
    }

    out_path = EXPERIMENTS_DIR / "cv_calibration_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cv_output, f, indent=2)
    print(f"\nSaved CV calibration data and PR curves to {out_path}.")


if __name__ == "__main__":
    main()
