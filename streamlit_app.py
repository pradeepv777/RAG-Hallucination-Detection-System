"""
Streamlit Interactive Dashboard for RAG Hallucination Detection.

Allows users to inspect RAG answers against context, view atomic claims,
inspect retrieved evidence from FAISS vector search, and explore NLI claim-level verdicts.
"""

import time
import streamlit as st

from claims import ClaimExtractor
from retrieve import EvidenceRetriever
from score import FaithfulnessReport, FaithfulnessScorer
from verify import NLIVerifier

# Page Configuration
st.set_page_config(
    page_title="RAG Hallucination Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .verdict-supported {
        color: #16A34A;
        font-weight: 600;
        background-color: #DCFCE7;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .verdict-contradicted {
        color: #DC2626;
        font-weight: 600;
        background-color: #FEE2E2;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .verdict-unsupported {
        color: #D97706;
        font-weight: 600;
        background-color: #FEF3C7;
        padding: 4px 8px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    """Caches pipeline models in memory."""
    extractor = ClaimExtractor()
    retriever = EvidenceRetriever()
    verifier = NLIVerifier()
    scorer = FaithfulnessScorer()
    return extractor, retriever, verifier, scorer


# Preset Examples from RAGTruth
PRESETS = {
    "Select an example...": {
        "question": "",
        "context": "",
        "answer": "",
    },
    "Example 1: Anne Frank (Contradicted + Baseless Hallucinations)": {
        "question": "When did Anne Frank die and what were the findings?",
        "context": (
            "Seventy years ago, Anne Frank died of typhus in a Nazi concentration camp at the age of 15. "
            "Just two weeks after her supposed death on March 31, 1945, the Bergen-Belsen camp was liberated. "
            "New research by the Anne Frank House reveals that Anne and her sister Margot likely died at least a month earlier. "
            "Witness testimonies and Red Cross archives indicate both had typhus symptoms before February 7, 1945. "
            "Dutch authorities previously recorded their deaths as March 31, 1945."
        ),
        "answer": (
            "Anne Frank and her sister Margot died in Bergen-Belsen. "
            "Recent investigations revealed they died on February 7, 2022. "
            "The Anne Frank House issued an official global retraction statement regarding her diary."
        ),
    },
    "Example 2: Transistor (Faithful Grounded QA)": {
        "question": "What is a transistor and how does it work?",
        "context": (
            "The design of a transistor allows it to function as an amplifier or a switch. "
            "It is a miniature electronic component made of semiconductor material. "
            "A small electric current flowing through one part can control a much larger current flowing through another part."
        ),
        "answer": (
            "A transistor is a miniature electronic component made of semiconductor material. "
            "It functions as an amplifier or an electrical switch. "
            "A small electric current through one part controls a much larger current in another part."
        ),
    },
    "Example 3: Restaurant Overview (Baseless Extrapolation)": {
        "question": "Summarize Panera Bread in Santa Barbara.",
        "context": (
            "name: Panera Bread\n"
            "city: Santa Barbara\n"
            "business_stars: 2.5\n"
            "review_count: 3\n"
            "RestaurantsReservations: False\n"
            "WiFi: free\n"
            "OutdoorSeating: True"
        ),
        "answer": (
            "Panera Bread in Santa Barbara has an average rating of 2.5 stars. "
            "It offers free WiFi and outdoor seating for guests. "
            "The restaurant has an atmosphere rating of 4 out of 5 stars and accepts table reservations."
        ),
    },
}


def main():
    st.markdown('<div class="main-header">RAG Hallucination Detection System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluating RAG answer faithfulness via atomic claim decomposition, dense FAISS retrieval, and DeBERTa NLI cross-encoder verification.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Model Config & Example Presets
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_preset = st.selectbox("Load RAGTruth Benchmark Preset:", list(PRESETS.keys()))

        top_k = st.slider("Evidence Chunks per Claim (Top-k)", min_value=1, max_value=5, value=3)
        entail_thresh = st.slider("Entailment Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        contra_thresh = st.slider("Contradiction Threshold", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

        st.markdown("---")
        st.markdown("**Architecture Components:**")
        st.markdown("- **Decomposition**: Deterministic Claim Extractor")
        st.markdown("- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`")
        st.markdown("- **Retrieval**: FAISS Cosine Index (`IndexFlatIP`)")
        st.markdown("- **NLI Model**: `cross-encoder/nli-deberta-v3-base`")

    preset_data = PRESETS[selected_preset]

    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        question_input = st.text_input(
            "User Question / Prompt:",
            value=preset_data["question"],
            placeholder="e.g. When did Anne Frank die?",
        )
        context_input = st.text_area(
            "Retrieved Context / Source Passages:",
            value=preset_data["context"],
            height=280,
            placeholder="Paste reference text or retrieved knowledge context here...",
        )

    with col_input2:
        answer_input = st.text_area(
            "RAG Generated Answer to Verify:",
            value=preset_data["answer"],
            height=345,
            placeholder="Paste the LLM generated answer here...",
        )

    verify_btn = st.button("🔍 Verify Answer Faithfulness", type="primary", use_container_width=True)

    if verify_btn:
        if not context_input.strip() or not answer_input.strip():
            st.error("Please provide both context and an answer to verify.")
            return

        with st.spinner("Analyzing answer claims against context..."):
            start_time = time.time()
            extractor, retriever, verifier, default_scorer = load_pipeline()

            # Dynamic scorer thresholds from sidebar
            scorer = FaithfulnessScorer(
                entailment_threshold=entail_thresh,
                contradiction_threshold=contra_thresh,
            )

            # 1. Claims
            claims = extractor.extract_claims(answer_input)
            if not claims:
                st.warning("No factual claims could be extracted from the answer.")
                return

            # 2. Retrieval
            retrieved = retriever.retrieve(context_input, claims, top_k=top_k)

            # 3. Verification
            verified = verifier.verify_claims(retrieved)

            # 4. Scoring
            report: FaithfulnessReport = scorer.score(
                verified, raw_answer=answer_input, raw_context=context_input
            )
            elapsed = time.time() - start_time

        st.markdown("---")
        st.subheader("📊 Faithfulness & Hallucination Assessment")

        # Top summary metric row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Faithfulness Score", f"{report.faithfulness_score * 100:.1f}%")
        with m2:
            status_text = "✅ Faithful" if report.is_faithful else "⚠️ Hallucinated"
            st.metric("Aggregate Status", status_text)
        with m3:
            st.metric("Supported Claims", f"{report.supported_claims}/{report.total_claims}")
        with m4:
            st.metric("Contradicted Claims", f"{report.contradicted_claims}")
        with m5:
            st.metric("Latency", f"{elapsed:.2f}s")

        st.markdown("### 📝 Claim-by-Claim Breakdown")

        for c in report.claims:
            with st.expander(f"Claim #{c.claim_id + 1}: {c.claim}", expanded=(c.verdict != "supported")):
                badge_class = f"verdict-{c.verdict}"
                st.markdown(
                    f"**Verdict:** <span class='{badge_class}'>{c.verdict.upper()}</span> "
                    f"&nbsp;&nbsp;|&nbsp;&nbsp; **Confidence:** `{c.score:.2%}`",
                    unsafe_allow_html=True,
                )

                st.markdown("**NLI Probabilities:**")
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                prob_col1.progress(c.entailment_prob, text=f"Entailment: {c.entailment_prob:.1%}")
                prob_col2.progress(c.contradiction_prob, text=f"Contradiction: {c.contradiction_prob:.1%}")
                prob_col3.progress(c.neutral_prob, text=f"Neutral: {c.neutral_prob:.1%}")

                st.markdown("**Best Retrieved Context Evidence:**")
                if c.evidence:
                    st.info(f"\"{c.evidence}\"\n\n*(Vector Cosine Similarity: `{c.evidence_similarity:.4f}`)*")
                else:
                    st.warning("No supporting evidence found in context.")


if __name__ == "__main__":
    main()
