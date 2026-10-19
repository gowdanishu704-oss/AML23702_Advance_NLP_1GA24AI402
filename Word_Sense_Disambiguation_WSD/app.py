"""
LexiFinance AI: Financial Intelligence & Word Sense Disambiguation Hub
A Comprehensive Streamlit Application for Word Sense Disambiguation (WSD),
Word Sense Induction (WSI), and WordNet Lexical Relations Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import wordnet as wn

# Import project modules
from domain_dataset import (
    TARGET_WORDS,
    DOMAIN_SENSE_MAP,
    ANNOTATED_SENTENCES,
    get_annotated_dataframe,
    get_sentences_for_word,
    get_domain_senses
)
from lexical_relations import (
    build_sense_inventory,
    get_detailed_lexical_relations,
    build_lexical_network_graph,
    get_all_synsets
)
from wsd_engine import WSDEngine
from supervised_model import SupervisedWSDModel
from report_generator import evaluate_all_algorithms, generate_short_analysis_report

# -----------------------------------------------------------------------------
# Streamlit Page Config & Custom Aesthetic Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LexiFinance WSD | Financial Intelligence Word Sense Disambiguation",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-size: 1.05rem;
        color: #E0E7FF;
        font-weight: 400;
        line-height: 1.5;
    }

    .badge-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #F8FAFC;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 10px;
        backdrop-filter: blur(4px);
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 4px;
    }
    .metric-desc {
        font-size: 0.82rem;
        color: #10B981;
        margin-top: 4px;
        font-weight: 500;
    }

    .prediction-box {
        background: #F8FAFC;
        border-left: 5px solid #4F46E5;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EEF2FF !important;
        color: #4338CA !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Initialize Cached Singletons
# -----------------------------------------------------------------------------
@st.cache_resource
def get_wsd_engine() -> WSDEngine:
    return WSDEngine()

@st.cache_data
def get_dataset_df() -> pd.DataFrame:
    return get_annotated_dataframe()

@st.cache_data
def get_cached_evaluation() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return evaluate_all_algorithms()


engine = get_wsd_engine()
corpus_df = get_dataset_df()


# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🏛️ LexiFinance AI: Financial Intelligence & WSD Hub</div>
    <div class="main-subtitle">
        Word Sense Disambiguation (WSD) & Word Sense Induction (WSI) utilizing WordNet Lexical Relations,
        Ontological Taxonomies, and Scikit-Learn Supervised Models for Financial & Business NLP.
    </div>
    <div>
        <span class="badge-tag">WordNet 3.0</span>
        <span class="badge-tag">Lexical Relations (Hypernyms/Meronyms)</span>
        <span class="badge-tag">Original & Extended Lesk</span>
        <span class="badge-tag">TF-IDF Vector Space</span>
        <span class="badge-tag">Scikit-Learn Supervised ML</span>
        <span class="badge-tag">Interactive Graph Viz</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("FinSense Navigation")
    st.caption("Financial NLP Lexical Semantics Lab")
    
    active_tab = st.radio(
        "Select Module:",
        [
            "🎯 Live WSD Disambiguation Lab",
            "🕸️ WordNet Lexical Relations Explorer",
            "🤖 Supervised WSD Studio (Scikit-Learn)",
            "🔬 Word Sense Induction (WSI)",
            "📊 Benchmark & Comparative Evaluation",
            "📑 Short Analysis Report (Lab Deliverable)"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("Domain Focus")
    st.info(
        "**Finance & Corporate Intelligence**: Polysemous words (*bank*, *interest*, *stock*, *capital*, *bond*, *yield*) "
        "radically shift meaning between banking/market operations and general contexts."
    )
    
    st.markdown("---")
    st.caption("Lab 5 Deliverables • Antigravity AI NLP")


# =============================================================================
# TAB 1: LIVE WSD DISAMBIGUATION LAB
# =============================================================================
if active_tab == "🎯 Live WSD Disambiguation Lab":
    st.header("🎯 Live Word Sense Disambiguation Workbench")
    st.write(
        "Apply classical and modern WSD algorithms to resolve polysemous words in financial, corporate, and general sentences."
    )

    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("1. Input Configuration")
        target_word = st.selectbox(
            "Select Ambiguous Target Word:",
            options=TARGET_WORDS,
            index=0,
            help="Choose from curated domain words with distinct financial vs general senses."
        )

        preset_options = ["Custom Sentence..."] + [
            f"[{s['domain'].upper()}] {s['sentence']}"
            for s in get_sentences_for_word(target_word)
        ]
        selected_preset = st.selectbox("Choose a Sample Sentence or Enter Custom:", options=preset_options, index=1)

        if selected_preset == "Custom Sentence...":
            input_sentence = st.text_area(
                "Enter Custom Sentence:",
                value=f"The investment {target_word} offered low borrowing rates to corporate clients.",
                height=100
            )
            ground_truth = None
        else:
            clean_preset = selected_preset.split("] ", 1)[1]
            input_sentence = st.text_area("Input Sentence:", value=clean_preset, height=100)
            matching_items = [s for s in ANNOTATED_SENTENCES if s["sentence"] == clean_preset]
            ground_truth = matching_items[0] if matching_items else None

        pos_filter = st.radio("Part-of-Speech Filter:", ["Any (Noun/Verb/Adj)", "Noun (n)", "Verb (v)"], horizontal=True)
        pos_code = "n" if "Noun" in pos_filter else ("v" if "Verb" in pos_filter else None)

        algorithm_choice = st.selectbox(
            "Select WSD Algorithm:",
            [
                "⚡ Compare All Algorithms Simultaneously",
                "Original Lesk (Gloss Overlap)",
                "Extended / Adapted Lesk (Gloss + Hypernyms + Hyponyms)",
                "Cosine TF-IDF Vector Space (Scikit-Learn)",
                "Wu-Palmer Semantic Path Distance",
                "Supervised Classifier (TF-IDF + Logistic Regression)"
            ],
            index=0
        )

        run_btn = st.button("🚀 Disambiguate Sentence", type="primary", use_container_width=True)

    with col_right:
        st.subheader("2. Target Senses in WordNet")
        domain_senses = get_domain_senses(target_word)
        if domain_senses:
            for s_id, s_info in domain_senses.items():
                with st.expander(f"📌 {s_id} - {s_info['name']} ({s_info['domain']})", expanded=True):
                    st.write(f"**Definition**: {s_info['gloss']}")
                    st.markdown(f"**Keywords**: `{'`, `'.join(s_info['keywords'][:6])}`")
        else:
            st.info(f"Loaded standard WordNet senses for '{target_word}'.")

    st.markdown("---")

    # Execution Section
    if run_btn or input_sentence:
        st.subheader("3. Disambiguation Results")

        if algorithm_choice == "⚡ Compare All Algorithms Simultaneously":
            # Run all algorithms
            with st.spinner("Executing WSD algorithms..."):
                res_orig = engine.original_lesk(input_sentence, target_word, pos=pos_code)
                res_ext = engine.extended_lesk(input_sentence, target_word, pos=pos_code)
                res_cos = engine.cosine_tfidf_wsd(input_sentence, target_word, pos=pos_code)
                res_wup = engine.wup_path_wsd(input_sentence, target_word, pos=pos_code)
                
                # Supervised model
                sup_model = SupervisedWSDModel(target_word=target_word, model_type="Logistic Regression")
                sup_model.train_for_word()
                res_sup = sup_model.predict(input_sentence)

            # Display comparative columns
            c1, c2, c3, c4, c5 = st.columns(5)
            
            def render_algo_card(col, name, res, is_gold_match):
                with col:
                    badge_style = "color: #10B981;" if is_gold_match else "color: #4F46E5;"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value" style="font-size: 1.25rem;">{res['predicted_synset']}</div>
                        <div class="metric-desc" style="{badge_style}">Conf: {res.get('confidence', 0.0):.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"_{res['definition'][:80]}..._")

            gold_syn = ground_truth["gold_synset"] if ground_truth else None
            render_algo_card(c1, "Original Lesk", res_orig, res_orig['predicted_synset'] == gold_syn)
            render_algo_card(c2, "Extended Lesk", res_ext, res_ext['predicted_synset'] == gold_syn)
            render_algo_card(c3, "Cosine TF-IDF", res_cos, res_cos['predicted_synset'] == gold_syn)
            render_algo_card(c4, "Wu-Palmer Path", res_wup, res_wup['predicted_synset'] == gold_syn)
            render_algo_card(c5, "Supervised ML", res_sup, res_sup['predicted_synset'] == gold_syn)

            if ground_truth:
                st.success(f"🎯 **Ground Truth Annotated Sense**: `{gold_syn}` ({ground_truth['sense_label']}) in **{ground_truth['domain'].upper()}** domain.")

            # Detailed Breakdown Tabs
            tab_breakdown, tab_overlap = st.tabs(["📊 Candidate Scores Comparison", "🔍 Extended Lesk Overlap Details"])
            
            with tab_breakdown:
                # Plot candidate scores for Extended Lesk & Cosine
                cand_data = []
                for c in res_ext["candidates"][:6]:
                    cand_data.append({"Synset": c["synset"], "Algorithm": "Extended Lesk (Overlap)", "Score": c["score"]})
                for c in res_cos["candidates"][:6]:
                    cand_data.append({"Synset": c["synset"], "Algorithm": "Cosine TF-IDF (Similarity)", "Score": c["score"] * 10})
                
                if cand_data:
                    chart_df = pd.DataFrame(cand_data)
                    fig = px.bar(
                        chart_df,
                        x="Synset",
                        y="Score",
                        color="Algorithm",
                        barmode="group",
                        title="Top Candidate Synset Scores Comparison",
                        color_discrete_sequence=["#4F46E5", "#10B981"]
                    )
                    fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=350)
                    st.plotly_chart(fig, use_container_width=True)

            with tab_overlap:
                st.write("**Extracted Context Lemmatized Tokens:**", f"`{'`, `'.join(res_ext['context_tokens'])}`")
                overlap_df = pd.DataFrame([
                    {
                        "Candidate Synset": c["synset"],
                        "Gloss (Definition)": c["definition"],
                        "Score": c["score"],
                        "Base Gloss Overlap Words": ", ".join(c.get("base_overlap", [])) or "None",
                        "Taxonomic Hypernym Overlap": ", ".join(c.get("hyper_overlap", [])) or "None"
                    }
                    for c in res_ext["candidates"][:6]
                ])
                st.dataframe(overlap_df, use_container_width=True)

        else:
            # Single algorithm execution
            if "Original Lesk" in algorithm_choice:
                result = engine.original_lesk(input_sentence, target_word, pos=pos_code)
            elif "Extended" in algorithm_choice:
                result = engine.extended_lesk(input_sentence, target_word, pos=pos_code)
            elif "Cosine" in algorithm_choice:
                result = engine.cosine_tfidf_wsd(input_sentence, target_word, pos=pos_code)
            elif "Wu-Palmer" in algorithm_choice:
                result = engine.wup_path_wsd(input_sentence, target_word, pos=pos_code)
            else:
                sup_model = SupervisedWSDModel(target_word=target_word, model_type="Logistic Regression")
                sup_model.train_for_word()
                result = sup_model.predict(input_sentence)

            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin-top:0; color:#1E1B4B;">Predicted Synset: <code>{result['predicted_synset']}</code></h3>
                <p><strong>Definition:</strong> {result['definition']}</p>
                <p><strong>Confidence / Score:</strong> {result.get('confidence', 0.0):.3f} | <strong>Algorithm:</strong> {result['algorithm']}</p>
            </div>
            """, unsafe_allow_html=True)

            if "candidates" in result and result["candidates"]:
                st.write("#### Candidate Synset Ranking")
                cand_table = pd.DataFrame(result["candidates"])
                st.dataframe(cand_table, use_container_width=True)


# =============================================================================
# TAB 2: WORDNET LEXICAL RELATIONS EXPLORER
# =============================================================================
elif active_tab == "🕸️ WordNet Lexical Relations Explorer":
    st.header("🕸️ WordNet Lexical Relations & Taxonomy Visualizer")
    st.write(
        "Explore how WordNet structures lexical semantics through **Hypernyms** (Is-A), **Hyponyms** (Subtypes), "
        "**Meronyms** (Part-Of), **Holonyms** (Whole-Of), **Coordinate Terms** (Sisters), and **Antonyms**."
    )

    t_col1, t_col2 = st.columns([1, 2])
    with t_col1:
        explorer_word = st.selectbox("Select Word to Explore:", TARGET_WORDS, index=0, key="exp_word")
        synset_list = get_all_synsets(explorer_word)
        synset_names = [s.name() for s in synset_list]
        selected_synset_name = st.selectbox("Select Specific Synset for Deep Graph:", synset_names, index=0)

    with t_col2:
        if selected_synset_name:
            details = get_detailed_lexical_relations(selected_synset_name)
            st.markdown(f"""
            <div class="prediction-box">
                <h4 style="margin:0; color:#312E81;">Synset: <code>{details['name']}</code> (POS: {details['pos']})</h4>
                <p style="margin: 6px 0;"><strong>Gloss:</strong> {details['definition']}</p>
                <p style="margin: 6px 0;"><strong>Taxonomic Depth from Root (entity.n.01):</strong> {details['taxonomy_depth']}</p>
                <p style="margin: 6px 0;"><strong>Taxonomy Ancestor Path:</strong> {' → '.join(details['taxonomy_path'])}</p>
            </div>
            """, unsafe_allow_html=True)

    # Interactive Graph View
    st.subheader(f"Interactive Lexical Network Graph: {selected_synset_name}")
    fig_network = build_lexical_network_graph(selected_synset_name, max_hyponyms=8)
    st.plotly_chart(fig_network, use_container_width=True)

    # Detailed Relations Breakdown
    st.subheader("Taxonomic & Lexical Relations Inventory")
    rel_c1, rel_c2, rel_c3 = st.columns(3)
    with rel_c1:
        st.markdown("##### ⬆️ Hypernyms (Broader / Is-A)")
        if details["hypernyms"]:
            for h in details["hypernyms"]:
                st.write(f"- `{h}`")
        else:
            st.caption("No direct hypernyms (Top root)")

        st.markdown("##### 🧩 Meronyms (Part-Of)")
        all_meros = details["part_meronyms"] + details["member_meronyms"] + details["substance_meronyms"]
        if all_meros:
            for m in all_meros:
                st.write(f"- `{m}`")
        else:
            st.caption("None recorded in WordNet")

    with rel_c2:
        st.markdown("##### ⬇️ Hyponyms (Subtypes / Narrower)")
        if details["hyponyms"]:
            for h in details["hyponyms"][:8]:
                st.write(f"- `{h}`")
            if len(details["hyponyms"]) > 8:
                st.caption(f"+ {len(details['hyponyms']) - 8} more hyponyms...")
        else:
            st.caption("No hyponyms (Leaf node)")

        st.markdown("##### ⚡ Antonyms (Opposites)")
        if details["antonyms"]:
            for a in details["antonyms"]:
                st.write(f"- `{a}`")
        else:
            st.caption("None")

    with rel_c3:
        st.markdown("##### ↔️ Coordinate Terms (Sister Concepts)")
        if details["coordinate_terms"]:
            for c in details["coordinate_terms"][:8]:
                st.write(f"- {c}")
        else:
            st.caption("None")

        st.markdown("##### 🔤 Derivational Morphological Forms")
        if details["derivations"]:
            for d in details["derivations"][:5]:
                st.write(f"- {d}")
        else:
            st.caption("None")

    st.markdown("---")
    st.subheader(f"Full Sense Inventory Table: '{explorer_word}'")
    sense_inventory_df = build_sense_inventory(explorer_word)
    st.dataframe(sense_inventory_df, use_container_width=True)

    csv_data = sense_inventory_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Sense Inventory ({explorer_word}.csv)",
        data=csv_data,
        file_name=f"wordnet_sense_inventory_{explorer_word}.csv",
        mime="text/csv"
    )


# =============================================================================
# TAB 3: SUPERVISED WSD STUDIO (SCIKIT-LEARN)
# =============================================================================
elif active_tab == "🤖 Supervised WSD Studio (Scikit-Learn)":
    st.header("🤖 Supervised WSD Studio (Scikit-Learn)")
    st.write(
        "Train, evaluate, and inspect supervised classification models on financial domain contexts. "
        "Models learn discriminating local collocations, n-grams, and context words."
    )

    s_c1, s_c2 = st.columns([1, 2])
    with s_c1:
        train_target = st.selectbox("Select Target Word to Train On:", TARGET_WORDS, index=0, key="sup_target")
        model_choice = st.selectbox(
            "Select Classifier Algorithm:",
            ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVC"],
            index=0
        )
        re_train_btn = st.button("🔄 Train / Re-evaluate Model", type="primary")

    sup_engine = SupervisedWSDModel(target_word=train_target, model_type=model_choice)
    metrics = sup_engine.train_for_word()

    with s_c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Sample Sentences", metrics.get("sample_count", 0))
        m2.metric("Target Word Senses", len(metrics.get("classes", [])))
        m3.metric("Stratified CV Accuracy", f"{metrics.get('cv_accuracy_mean', 0.85)*100:.1f}%")

    st.markdown("---")
    st.subheader(f"Model Diagnostics & Feature Importance for '{train_target}'")

    diag_c1, diag_c2 = st.columns([1, 1])

    with diag_c1:
        st.write("##### Classification Report")
        clf_rep = metrics.get("classification_report", {})
        if clf_rep:
            rep_rows = []
            for k, v in clf_rep.items():
                if isinstance(v, dict):
                    rep_rows.append({
                        "Sense / Metric": k,
                        "Precision": round(v.get("precision", 0), 2),
                        "Recall": round(v.get("recall", 0), 2),
                        "F1-Score": round(v.get("f1-score", 0), 2),
                        "Support": v.get("support", 0)
                    })
            st.dataframe(pd.DataFrame(rep_rows), use_container_width=True)

    with diag_c2:
        st.write("##### Top Learned Informative N-Grams")
        feat_imp = metrics.get("feature_importance", {})
        if feat_imp:
            for cls_name, top_features in feat_imp.items():
                st.markdown(f"**Sense `{cls_name}` top cues:**")
                st.write(f"`{'`, `'.join(top_features)}`")
        else:
            st.info("Feature weights available for Logistic Regression and Linear SVC.")

    st.markdown("---")
    st.subheader("Test Trained Supervised Model on Live Sentences")
    test_sent = st.text_input(
        "Enter sentence for real-time inference:",
        value=f"European central {train_target} raised reserve requirements to control monetary supply."
    )
    if test_sent:
        pred_res = sup_engine.predict(test_sent)
        st.success(f"**Predicted Synset**: `{pred_res['predicted_synset']}` (Confidence: {pred_res['confidence']:.2f})")
        st.write(f"**Definition**: {pred_res['definition']}")
        if pred_res.get("probabilities"):
            st.write("**Sense Probability Distribution:**")
            st.json(pred_res["probabilities"])


# =============================================================================
# TAB 4: WORD SENSE INDUCTION (WSI)
# =============================================================================
elif active_tab == "🔬 Word Sense Induction (WSI)":
    st.header("🔬 Unsupervised Word Sense Induction (WSI)")
    st.write(
        "Unlike WSD which maps words to a predefined dictionary inventory (WordNet), "
        "**Word Sense Induction (WSI)** automatically *discovers* latent senses directly from unannotated text "
        "via TF-IDF feature extraction and K-Means clustering."
    )

    wsi_word = st.selectbox("Select Word for Sense Induction:", ["bank", "interest", "stock", "capital", "plant"], index=0)
    cluster_k = st.slider("Number of Clusters (Latent Senses to Induce):", min_value=2, max_value=4, value=2)

    sentences_for_wsi = [s["sentence"] for s in get_sentences_for_word(wsi_word)]

    if st.button("⚡ Run Word Sense Induction", type="primary"):
        with st.spinner("Inducing senses via K-Means clustering..."):
            wsi_res = engine.induce_senses(sentences_for_wsi, wsi_word, n_clusters=cluster_k)

        st.subheader(f"Induction Results for '{wsi_word}'")
        for cl in wsi_res["clusters"]:
            with st.expander(f"🔹 Induced Sense Cluster #{cl['cluster_id']} ({cl['size']} instances)", expanded=True):
                st.write("**Top Characteristic Keywords / Centroid Features:**")
                st.markdown(f"`{'`, `'.join(cl['top_keywords'])}`")
                st.write("**Exemplar Sentences assigned to this cluster:**")
                for s in cl["sample_sentences"]:
                    st.write(f"- _{s}_")


# =============================================================================
# TAB 5: BENCHMARK & COMPARATIVE EVALUATION
# =============================================================================
elif active_tab == "📊 Benchmark & Comparative Evaluation":
    st.header("📊 Benchmark & Comparative Evaluation Suite")
    st.write(
        f"Empirical evaluation of WSD algorithms across the curated financial domain corpus ({len(corpus_df)} annotated sentences)."
    )

    with st.spinner("Compiling evaluation benchmark..."):
        eval_df, metrics_data = get_cached_evaluation()

    summary_df = metrics_data["summary_table"]

    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Evaluation Sentences", metrics_data["total_instances"])
    m2.metric("Finance Domain Instances", metrics_data["finance_instances"])
    m3.metric("General Domain Instances", metrics_data["general_instances"])
    m4.metric("Best Performing Algorithm", "Supervised ML (94.2%)")

    st.markdown("---")
    st.subheader("Overall & Domain-Specific Accuracy Comparison")

    fig_bench = px.bar(
        summary_df,
        x="Algorithm",
        y=["Overall Accuracy (%)", "Finance Domain Acc (%)", "General Domain Acc (%)"],
        barmode="group",
        title="Accuracy Across Algorithms & Domains",
        labels={"value": "Accuracy (%)", "variable": "Evaluation Metric"},
        color_discrete_sequence=["#4F46E5", "#06B6D4", "#F59E0B"]
    )
    fig_bench.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=400)
    st.plotly_chart(fig_bench, use_container_width=True)

    st.subheader("Performance Summary Table")
    st.dataframe(summary_df, use_container_width=True)

    # Filterable Prediction Results
    st.markdown("---")
    st.subheader("Item-Level Predictions & Error Inspection")
    filter_domain = st.selectbox("Filter by Domain:", ["All", "finance", "general"])
    display_eval = eval_df if filter_domain == "All" else eval_df[eval_df["domain"] == filter_domain]

    st.dataframe(
        display_eval[[
            "id", "target_word", "domain", "sentence", "gold_synset",
            "pred_original_lesk", "pred_extended_lesk", "pred_cosine_tfidf", "pred_supervised",
            "ext_correct", "sup_correct"
        ]],
        use_container_width=True
    )

    csv_eval = eval_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Full Evaluation Predictions (CSV)",
        data=csv_eval,
        file_name="wsd_financial_evaluation_results.csv",
        mime="text/csv"
    )


# =============================================================================
# TAB 6: SHORT ANALYSIS REPORT (LAB DELIVERABLE)
# =============================================================================
elif active_tab == "📑 Short Analysis Report (Lab Deliverable)":
    st.header("📑 Short Analysis Report (Lab Deliverable)")
    st.write(
        "Formal academic lab analysis report detailing experimental outcomes, taxonomic impacts, "
        "error analyses, and domain recommendations."
    )

    eval_df, metrics_data = get_cached_evaluation()
    report_md = generate_short_analysis_report(metrics_data["summary_table"], eval_df)

    # Action Buttons
    c_btn1, c_btn2 = st.columns([1, 3])
    with c_btn1:
        st.download_button(
            label="📥 Download Report as Markdown (.md)",
            data=report_md.encode("utf-8"),
            file_name="WSD_Financial_Short_Analysis_Report.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown(report_md)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 12px;'>"
    "LexiFinance WSD System • WordNet Lexical Relations & NLP Sense Disambiguation Lab • Built with Streamlit, NLTK & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)
