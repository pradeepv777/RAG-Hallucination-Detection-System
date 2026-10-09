"""
Canonical Feature Engineering Module for RAG Hallucination Detection.

Consolidates all feature extraction logic across the repository into ONE
modular, extensible, and independently selectable architecture.

Feature Groups:
  - BASE: The 14 standard baseline features (counts, ent/con/neu stats, sim stats, task indicators)
  - RETRIEVAL: Advanced similarity distributions, top-3 spread, and chunk quality metrics
  - NLI: Non-linear probability gaps, ratios, confidence percentiles, and contradiction signals
  - CONSISTENCY: Deterministic extraction & verification of numbers, percentages, currencies, dates, years
  - CLAIM: Claim-level surface features (lengths, position, entity/num presence)
  - ANSWER: Response-level density, aggregate scores, and full task representations

Usage:
  extractor = FeatureExtractor(groups=["BASE", "RETRIEVAL", "NLI"])
  X = extractor.extract_batch(examples)
"""

from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Deterministic Numeric & Date Extraction Utilities
# ---------------------------------------------------------------------------

_CURRENCY_PATTERN = re.compile(r"\$\s*[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s*(?:dollars?|usd|eur|gbp)\b", re.IGNORECASE)
_PERCENT_PATTERN = re.compile(r"[\d,]+(?:\.\d+)?\s*%", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"\b(?:1[789]\d{2}|20\d{2})\b")
_DATE_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    re.IGNORECASE,
)
_GENERIC_NUM_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")


def extract_numbers_and_dates(text: str) -> Dict[str, List[str]]:
    """Extracts typed numeric and temporal entities from text."""
    if not text:
        return {"currencies": [], "percentages": [], "years": [], "dates": [], "numbers": []}

    currencies = _CURRENCY_PATTERN.findall(text)
    percentages = _PERCENT_PATTERN.findall(text)
    years = _YEAR_PATTERN.findall(text)
    dates = _DATE_PATTERN.findall(text)
    all_nums = _GENERIC_NUM_PATTERN.findall(text)

    # Normalize extracted tokens
    def clean(s: str) -> str:
        return s.strip().lower().replace(",", "")

    return {
        "currencies": [clean(c) for c in currencies],
        "percentages": [clean(p) for p in percentages],
        "years": [clean(y) for y in years],
        "dates": [clean(d) for d in dates],
        "numbers": [clean(n) for n in all_nums],
    }


def compute_token_overlap_ratio(source_tokens: List[str], target_text: str) -> Tuple[float, float]:
    """
    Computes (match_ratio, mismatch_ratio) for a list of tokens against target text.
    Returns (1.0, 0.0) if source_tokens is empty (vacuous consistency).
    """
    if not source_tokens:
        return 1.0, 0.0

    target_clean = target_text.lower().replace(",", "")
    matched = sum(1 for tok in source_tokens if tok in target_clean)
    total = len(source_tokens)
    ratio = matched / total
    return ratio, 1.0 - ratio


# ---------------------------------------------------------------------------
# Canonical Feature Extractor
# ---------------------------------------------------------------------------

class FeatureGroup(str, Enum):
    BASE = "BASE"
    RETRIEVAL = "RETRIEVAL"
    NLI = "NLI"
    CONSISTENCY = "CONSISTENCY"
    CLAIM = "CLAIM"
    ANSWER = "ANSWER"


class FeatureExtractor:
    """
    Canonical, modular feature extraction system for RAG hallucination detection.
    Supports modular selection of feature groups with zero duplication.
    """

    AVAILABLE_GROUPS = [g.value for g in FeatureGroup]

    def __init__(self, groups: Optional[List[Union[str, FeatureGroup]]] = None):
        if groups is None:
            self.active_groups = list(self.AVAILABLE_GROUPS)
        else:
            self.active_groups = [g.value if isinstance(g, FeatureGroup) else str(g).upper() for g in groups]
            for g in self.active_groups:
                if g not in self.AVAILABLE_GROUPS:
                    raise ValueError(f"Unknown feature group: {g}. Available: {self.AVAILABLE_GROUPS}")

        self._feature_names = self._build_feature_names()

    @property
    def feature_dim(self) -> int:
        return len(self._feature_names)

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names)

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []

        if FeatureGroup.BASE.value in self.active_groups:
            names.extend([
                "base_total_claims",
                "base_supported_fraction",
                "base_unsupported_fraction",
                "base_contradicted_fraction",
                "base_min_entailment",
                "base_mean_entailment",
                "base_max_contradiction",
                "base_mean_contradiction",
                "base_max_neutral",
                "base_mean_neutral",
                "base_min_similarity",
                "base_mean_similarity",
                "base_is_qa",
                "base_is_summary",
            ])

        if FeatureGroup.RETRIEVAL.value in self.active_groups:
            names.extend([
                "ret_max_similarity",
                "ret_std_similarity",
                "ret_similarity_spread",       # max - min
                "ret_high_sim_fraction",       # claims with sim >= 0.70
                "ret_low_sim_fraction",        # claims with sim < 0.40
                "ret_mean_top3_similarity",
            ])

        if FeatureGroup.NLI.value in self.active_groups:
            names.extend([
                "nli_max_entailment",
                "nli_std_entailment",
                "nli_min_ent_minus_con",
                "nli_mean_ent_minus_con",
                "nli_max_ent_minus_con",
                "nli_mean_ent_ratio",          # ent / (ent + con + eps)
                "nli_min_ent_ratio",
                "nli_strongly_entailed_fraction",  # ent >= 0.60
                "nli_uncertain_fraction",          # neu >= 0.50
                "nli_has_any_contradiction",       # con >= 0.35 on any claim
            ])

        if FeatureGroup.CONSISTENCY.value in self.active_groups:
            names.extend([
                "cons_has_numbers",
                "cons_number_match_ratio",
                "cons_number_mismatch_ratio",
                "cons_has_dates_or_years",
                "cons_date_match_ratio",
                "cons_date_mismatch_ratio",
                "cons_has_any_mismatch",
            ])

        if FeatureGroup.CLAIM.value in self.active_groups:
            names.extend([
                "claim_mean_word_length",
                "claim_max_word_length",
                "claim_mean_char_length",
                "claim_has_number_fraction",
                "claim_has_date_fraction",
                "claim_has_percent_fraction",
                "claim_first_claim_entailment",
                "claim_last_claim_entailment",
            ])

        if FeatureGroup.ANSWER.value in self.active_groups:
            names.extend([
                "ans_char_length",
                "ans_word_length",
                "ans_claims_per_100_words",
                "ans_mean_claim_score",        # mean(ent - con)
                "ans_min_claim_score",         # min(ent - con)
                "ans_max_claim_score",         # max(ent - con)
                "ans_is_data2txt",
            ])

        return names

    def extract(self, ex: Dict[str, Any]) -> np.ndarray:
        """Extracts a feature vector for a single example dictionary."""
        claims = ex.get("claims", [])
        task = ex.get("task_type", "")
        response = ex.get("response", "")

        n_claims = len(claims)
        feature_parts: List[np.ndarray] = []

        # Precompute arrays for safety
        if n_claims > 0:
            ents = np.array([float(c.get("entailment_prob", 0.0)) for c in claims], dtype=np.float32)
            cons = np.array([float(c.get("contradiction_prob", 0.0)) for c in claims], dtype=np.float32)
            neus = np.array([float(c.get("neutral_prob", 0.0)) for c in claims], dtype=np.float32)
            sims = np.array([float(c.get("top_similarity", 0.0)) for c in claims], dtype=np.float32)
        else:
            ents = np.array([0.0], dtype=np.float32)
            cons = np.array([0.0], dtype=np.float32)
            neus = np.array([1.0], dtype=np.float32)
            sims = np.array([0.0], dtype=np.float32)

        # -------------------------------------------------------------
        # 1. BASE FEATURES (14 dims)
        # -------------------------------------------------------------
        if FeatureGroup.BASE.value in self.active_groups:
            if n_claims > 0:
                sup_count = float(np.sum(ents >= 0.35))
                con_count = float(np.sum(cons >= 0.35))
                unsup_count = float(n_claims - sup_count - con_count)

                base_vec = [
                    float(n_claims),
                    sup_count / n_claims,
                    unsup_count / n_claims,
                    con_count / n_claims,
                    float(np.min(ents)),
                    float(np.mean(ents)),
                    float(np.max(cons)),
                    float(np.mean(cons)),
                    float(np.max(neus)),
                    float(np.mean(neus)),
                    float(np.min(sims)),
                    float(np.mean(sims)),
                    1.0 if task == "QA" else 0.0,
                    1.0 if task == "Summary" else 0.0,
                ]
            else:
                base_vec = [0.0] * 12 + [1.0 if task == "QA" else 0.0, 1.0 if task == "Summary" else 0.0]
            feature_parts.append(np.array(base_vec, dtype=np.float32))

        # -------------------------------------------------------------
        # 2. RETRIEVAL FEATURES (6 dims)
        # -------------------------------------------------------------
        if FeatureGroup.RETRIEVAL.value in self.active_groups:
            if n_claims > 0:
                max_s = float(np.max(sims))
                min_s = float(np.min(sims))
                std_s = float(np.std(sims))
                spread_s = max_s - min_s
                high_sim = float(np.mean(sims >= 0.70))
                low_sim = float(np.mean(sims < 0.40))
                mean_top3 = float(np.mean(sorted(sims, reverse=True)[:3]))
                ret_vec = [max_s, std_s, spread_s, high_sim, low_sim, mean_top3]
            else:
                ret_vec = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            feature_parts.append(np.array(ret_vec, dtype=np.float32))

        # -------------------------------------------------------------
        # 3. NLI FEATURES (10 dims)
        # -------------------------------------------------------------
        if FeatureGroup.NLI.value in self.active_groups:
            if n_claims > 0:
                max_e = float(np.max(ents))
                std_e = float(np.std(ents))
                diff = ents - cons
                min_diff = float(np.min(diff))
                mean_diff = float(np.mean(diff))
                max_diff = float(np.max(diff))

                eps = 1e-5
                ratio = ents / (ents + cons + eps)
                mean_ratio = float(np.mean(ratio))
                min_ratio = float(np.min(ratio))

                strong_ent = float(np.mean(ents >= 0.60))
                uncertain = float(np.mean(neus >= 0.50))
                has_any_con = 1.0 if np.any(cons >= 0.35) else 0.0

                nli_vec = [
                    max_e,
                    std_e,
                    min_diff,
                    mean_diff,
                    max_diff,
                    mean_ratio,
                    min_ratio,
                    strong_ent,
                    uncertain,
                    has_any_con,
                ]
            else:
                nli_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            feature_parts.append(np.array(nli_vec, dtype=np.float32))

        # -------------------------------------------------------------
        # 4. CONSISTENCY FEATURES (7 dims)
        # -------------------------------------------------------------
        if FeatureGroup.CONSISTENCY.value in self.active_groups:
            if n_claims > 0:
                num_matches = []
                num_mismatches = []
                date_matches = []
                date_mismatches = []
                has_num_any = 0.0
                has_date_any = 0.0

                for c in claims:
                    c_text = c.get("raw_claim", c.get("target_claim", ""))
                    premise = c.get("premise", "")
                    parsed = extract_numbers_and_dates(c_text)

                    # Numbers (plain numbers, currencies, percentages)
                    all_claim_nums = parsed["numbers"] + parsed["currencies"] + parsed["percentages"]
                    if all_claim_nums:
                        has_num_any = 1.0
                        m_ratio, mis_ratio = compute_token_overlap_ratio(all_claim_nums, premise)
                        num_matches.append(m_ratio)
                        num_mismatches.append(mis_ratio)

                    # Dates and years
                    all_claim_dates = parsed["dates"] + parsed["years"]
                    if all_claim_dates:
                        has_date_any = 1.0
                        dm_ratio, dmis_ratio = compute_token_overlap_ratio(all_claim_dates, premise)
                        date_matches.append(dm_ratio)
                        date_mismatches.append(dmis_ratio)

                mean_num_match = float(np.mean(num_matches)) if num_matches else 1.0
                mean_num_mismatch = float(np.mean(num_mismatches)) if num_mismatches else 0.0
                mean_date_match = float(np.mean(date_matches)) if date_matches else 1.0
                mean_date_mismatch = float(np.mean(date_mismatches)) if date_mismatches else 0.0
                has_any_mis = 1.0 if (mean_num_mismatch > 0.0 or mean_date_mismatch > 0.0) else 0.0

                cons_vec = [
                    has_num_any,
                    mean_num_match,
                    mean_num_mismatch,
                    has_date_any,
                    mean_date_match,
                    mean_date_mismatch,
                    has_any_mis,
                ]
            else:
                cons_vec = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            feature_parts.append(np.array(cons_vec, dtype=np.float32))

        # -------------------------------------------------------------
        # 5. CLAIM FEATURES (8 dims)
        # -------------------------------------------------------------
        if FeatureGroup.CLAIM.value in self.active_groups:
            if n_claims > 0:
                word_lens = []
                char_lens = []
                has_num_flags = []
                has_date_flags = []
                has_pct_flags = []

                for c in claims:
                    txt = c.get("raw_claim", c.get("target_claim", ""))
                    words = txt.split()
                    word_lens.append(len(words))
                    char_lens.append(len(txt))

                    parsed = extract_numbers_and_dates(txt)
                    has_num_flags.append(1.0 if (parsed["numbers"] or parsed["currencies"]) else 0.0)
                    has_date_flags.append(1.0 if (parsed["dates"] or parsed["years"]) else 0.0)
                    has_pct_flags.append(1.0 if parsed["percentages"] else 0.0)

                first_ent = float(ents[0])
                last_ent = float(ents[-1])

                claim_vec = [
                    float(np.mean(word_lens)),
                    float(np.max(word_lens)),
                    float(np.mean(char_lens)),
                    float(np.mean(has_num_flags)),
                    float(np.mean(has_date_flags)),
                    float(np.mean(has_pct_flags)),
                    first_ent,
                    last_ent,
                ]
            else:
                claim_vec = [0.0] * 8
            feature_parts.append(np.array(claim_vec, dtype=np.float32))

        # -------------------------------------------------------------
        # 6. ANSWER FEATURES (7 dims)
        # -------------------------------------------------------------
        if FeatureGroup.ANSWER.value in self.active_groups:
            resp_words = len(response.split()) if response else 0
            resp_chars = len(response) if response else 0
            claims_per_100 = (float(n_claims) / max(1, resp_words)) * 100.0

            if n_claims > 0:
                claim_scores = ents - cons
                mean_score = float(np.mean(claim_scores))
                min_score = float(np.min(claim_scores))
                max_score = float(np.max(claim_scores))
            else:
                mean_score, min_score, max_score = 0.0, 0.0, 0.0

            ans_vec = [
                float(resp_chars),
                float(resp_words),
                claims_per_100,
                mean_score,
                min_score,
                max_score,
                1.0 if task == "Data2txt" else 0.0,
            ]
            feature_parts.append(np.array(ans_vec, dtype=np.float32))

        return np.concatenate(feature_parts).astype(np.float32)

    def extract_batch(self, examples: List[Dict[str, Any]]) -> np.ndarray:
        """Extracts feature matrix (N, D) for a list of examples."""
        if not examples:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        rows = [self.extract(ex) for ex in examples]
        return np.vstack(rows).astype(np.float32)


# Helper for backward compatibility
def extract_canonical_features(ex: Dict[str, Any], groups: Optional[List[str]] = None) -> np.ndarray:
    extractor = FeatureExtractor(groups=groups)
    return extractor.extract(ex)
