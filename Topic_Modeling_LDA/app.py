import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessor import TextPreprocessor
from src.lda_engine import LDAEngine
import src.visualizations as viz
from src.report_generator import generate_topic_report

# Page configuration
st.set_page_config(
    page_title="TopicScope | LDA Topic Modeling Studio",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    
    .badge-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #E0E7FF;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-right: 8px;
        margin-bottom: 8px;
        backdrop-filter: blur(4px);
    }
    
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    .metric-title {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .metric-value {
        color: #0F172A;
        font-size: 26px;
        font-weight: 700;
    }
    
    .topic-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE MANAGEMENT -----------------
if "corpus_loaded" not in st.session_state:
    st.session_state.corpus_loaded = False
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "lda_engine" not in st.session_state:
    st.session_state.lda_engine = LDAEngine()
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()

# ----------------- HEADER -----------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 32px; font-weight: 700; color: white;">TopicScope Studio</h1>
            <p style="margin: 6px 0 12px 0; font-size: 15px; color: #C7D2FE;">
                Probabilistic Topic Discovery & Latent Dirichlet Allocation (LDA) for News Media & Journalism
            </p>
            <div>
                <span class="badge-tag">Unsupervised NLP</span>
                <span class="badge-tag">Variational Bayes</span>
                <span class="badge-tag">Dirichlet Priors</span>
                <span class="badge-tag">Topic-Word Distributions</span>
                <span class="badge-tag">News Media Domain</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### ⚙️ Pipeline Configuration")

# 1. Corpus Selector
data_source = st.sidebar.selectbox(
    "Corpus Source",
    ["Curated Benchmark News Corpus", "Upload Custom CSV / TXT", "Scikit-Learn 20 Newsgroups (Subset)"],
    index=0,
    help="Select the document collection to analyze."
)

raw_texts = []
categories = []

if data_source == "Curated Benchmark News Corpus":
    corpus_path = "data/news_corpus.csv"
    if os.path.exists(corpus_path):
        df_raw = pd.read_csv(corpus_path)
        raw_texts = df_raw["text"].tolist()
        categories = df_raw["category"].tolist() if "category" in df_raw.columns else None
        st.sidebar.success(f"✓ Loaded {len(raw_texts)} curated news articles across 5 domains.")
    else:
        st.sidebar.error("Corpus file not found. Generating default...")
        os.system("python generate_corpus.py")
        df_raw = pd.read_csv(corpus_path)
        raw_texts = df_raw["text"].tolist()
        categories = df_raw["category"].tolist()
elif data_source == "Upload Custom CSV / TXT":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or TXT File", type=["csv", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
            text_col = st.sidebar.selectbox("Select Text Column", df_upload.columns)
            raw_texts = df_upload[text_col].dropna().astype(str).tolist()
            categories = None
            st.sidebar.success(f"Loaded {len(raw_texts)} uploaded documents.")
        else:
            content = uploaded_file.read().decode("utf-8")
            raw_texts = [line.strip() for line in content.split("\n") if len(line.strip()) > 30]
            categories = None
            st.sidebar.success(f"Loaded {len(raw_texts)} paragraphs from TXT.")
elif data_source == "Scikit-Learn 20 Newsgroups (Subset)":
    st.sidebar.info("Loading 20 Newsgroups subset...")
    try:
        from sklearn.datasets import fetch_20newsgroups
        selected_cats = ['sci.space', 'comp.graphics', 'talk.politics.mideast', 'rec.sport.hockey', 'sci.med']
        ns_data = fetch_20newsgroups(subset='train', categories=selected_cats, remove=('headers', 'footers', 'quotes'))
        raw_texts = [d for d in ns_data.data if len(d.strip()) > 50][:300]
        categories = [selected_cats[t] for t in ns_data.target[:len(raw_texts)]]
        st.sidebar.success(f"Loaded {len(raw_texts)} newsgroup articles.")
    except Exception as e:
        st.sidebar.warning(f"Could not load 20 Newsgroups online ({e}). Falling back to curated corpus.")
        df_raw = pd.read_csv("data/news_corpus.csv")
        raw_texts = df_raw["text"].tolist()
        categories = df_raw["category"].tolist()

# 2. Preprocessing Options
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧹 Preprocessing Hyperparameters")

use_lemmatization = st.sidebar.checkbox("Apply WordNet Lemmatization", value=True)
min_token_len = st.sidebar.slider("Minimum Token Length", 2, 5, 3)

custom_stops_input = st.sidebar.text_input(
    "Additional Custom Stopwords (comma-separated)",
    value="also, said, reported, would, could, one, time"
)
custom_stops = [w.strip() for w in custom_stops_input.split(",") if w.strip()]

ngram_choice = st.sidebar.selectbox("N-gram Range", ["Unigrams (1, 1)", "Unigrams + Bigrams (1, 2)"], index=0)
ngram_range = (1, 2) if "Bigrams" in ngram_choice else (1, 1)

min_df = st.sidebar.slider("Min Document Frequency (min_df)", 1, 5, 2, help="Ignore terms appearing in fewer than this many documents")
max_df = st.sidebar.slider("Max Document Frequency (max_df)", 0.5, 1.0, 0.85, step=0.05, help="Ignore terms appearing in more than this fraction of documents")
max_features = st.sidebar.slider("Max Vocabulary Size", 200, 3000, 1000, step=100)

# 3. LDA Model Hyperparameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 LDA Model Settings")

n_topics = st.sidebar.slider("Number of Topics (K)", min_value=2, max_value=10, value=5, help="Number of latent topic mixtures to discover")
learning_method = st.sidebar.selectbox("Learning Method", ["online", "batch"], index=0, help="Online Variational Bayes is fast and memory efficient")
max_iter = st.sidebar.slider("Max Iterations", 10, 100, 25, step=5)

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    alpha_prior = st.number_input("Doc-Topic Prior (α)", value=0.1, min_value=0.01, max_value=1.0, step=0.05, help="Controls document topic sparsity")
with col_p2:
    beta_prior = st.number_input("Topic-Word Prior (β)", value=0.01, min_value=0.001, max_value=0.5, step=0.01, help="Controls topic word vocabulary sparsity")

random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Execution Button
train_btn = st.sidebar.button("🚀 Run Preprocessing & Train LDA", type="primary", use_container_width=True)

# ----------------- PIPELINE EXECUTION -----------------
if train_btn or not st.session_state.model_trained:
    with st.spinner("Executing Text Preprocessing & Training Latent Dirichlet Allocation..."):
        # 1. Preprocessing
        preprocessor = TextPreprocessor(
            custom_stopwords=custom_stops,
            use_lemmatizer=use_lemmatization,
            min_word_len=min_token_len
        )
        clean_texts, tokenized_docs = preprocessor.preprocess_corpus(raw_texts)
        
        # 2. DTM
        lda_engine = LDAEngine()
        dtm, vectorizer, corpus_stats = lda_engine.build_dtm(
            clean_texts=clean_texts,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            ngram_range=ngram_range
        )
        
        # 3. Train LDA
        model_results = lda_engine.train(
            n_components=n_topics,
            learning_method=learning_method,
            max_iter=max_iter,
            doc_topic_prior=alpha_prior,
            topic_word_prior=beta_prior,
            random_state=int(random_seed)
        )
        
        # 4. Extract topics & document assignments
        topics_data = lda_engine.get_topic_keywords(top_n=15)
        df_docs = lda_engine.get_document_assignments(raw_texts=raw_texts, categories=categories)
        
        # Save preprocessed corpus & model outputs
        df_preprocessed = pd.DataFrame({
            "doc_id": range(len(raw_texts)),
            "raw_text": raw_texts,
            "cleaned_text": clean_texts,
            "dominant_topic": df_docs["dominant_topic"],
            "topic_weight": df_docs["topic_weight"]
        })
        df_preprocessed.to_csv("data/preprocessed_corpus.csv", index=False)
        
        # Save in session state
        st.session_state.preprocessor = preprocessor
        st.session_state.lda_engine = lda_engine
        st.session_state.corpus_stats = corpus_stats
        st.session_state.model_results = model_results
        st.session_state.topics_data = topics_data
        st.session_state.df_docs = df_docs
        st.session_state.df_preprocessed = df_preprocessed
        st.session_state.clean_texts = clean_texts
        st.session_state.raw_texts = raw_texts
        st.session_state.model_trained = True

# Retrieve cached objects
lda_engine = st.session_state.lda_engine
preprocessor = st.session_state.preprocessor
corpus_stats = st.session_state.corpus_stats
model_results = st.session_state.model_results
topics_data = st.session_state.topics_data
df_docs = st.session_state.df_docs
df_preprocessed = st.session_state.df_preprocessed
raw_texts = st.session_state.raw_texts

# ----------------- SUMMARY METRIC CARDS -----------------
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">Documents</div><div class="metric-value">{corpus_stats['num_docs']:,}</div></div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">Vocabulary Size</div><div class="metric-value">{corpus_stats['vocab_size']:,}</div></div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">Matrix Sparsity</div><div class="metric-value">{corpus_stats['sparsity_pct']}%</div></div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">LDA Perplexity</div><div class="metric-value">{model_results['perplexity']}</div></div>""", unsafe_allow_html=True)
with m5:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">U_mass Coherence</div><div class="metric-value">{model_results['mean_coherence']}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- TABS INTERFACE -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📄 Corpus & Preprocessing",
    "⚙️ Model Diagnostics & Optimal K",
    "🏷️ Topic-Word Distributions",
    "📊 Document Mixtures & Distance Map",
    "🎯 Live Inference Playground",
    "📝 Topic Interpretation Report"
])

# ==================== TAB 1: CORPUS & PREPROCESSING ====================
with tab1:
    st.markdown("### 📄 Text Preprocessing & Document-Term Representation")
    st.markdown("""
    Text preprocessing transforms unstructured raw news documents into a structured **Document-Term Matrix (DTM)**:
    1. **Normalization:** Lowercasing, removing symbols, punctuation, and non-alphabetic noise.
    2. **Stopwords Elimination:** Removing non-informative functional words.
    3. **Morphological Reduction:** Applying WordNet Lemmatization (e.g., *negotiations* $\\rightarrow$ *negotiation*, *played* $\\rightarrow$ *play*).
    4. **Sparsity Constraints:** Pruning rare words with `min_df` and ultra-frequent words with `max_df`.
    """)
    
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("#### Sample Cleaned Documents")
        st.dataframe(
            df_preprocessed[["doc_id", "raw_text", "cleaned_text"]].head(8),
            use_container_width=True,
            height=320
        )
        
        # Download Preprocessed Corpus Button
        csv_data = df_preprocessed.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Preprocessed Corpus (CSV)",
            data=csv_data,
            file_name="preprocessed_news_corpus.csv",
            mime="text/csv",
            help="Download the full preprocessed corpus with clean tokens and metadata"
        )

    with col_t2:
        st.markdown("#### Top Overall Vocabulary Terms")
        # Term frequencies across whole DTM
        word_counts = np.asarray(lda_engine.doc_term_matrix.sum(axis=0)).flatten()
        top_words_idx = np.argsort(word_counts)[::-1][:15]
        top_vocab_df = pd.DataFrame({
            "term": lda_engine.feature_names[top_words_idx],
            "frequency": word_counts[top_words_idx]
        })
        fig_vocab = px.bar(
            top_vocab_df,
            x="frequency",
            y="term",
            orientation="h",
            color="frequency",
            color_continuous_scale="Purples",
            title="Most Frequent Terms in Document-Term Matrix"
        )
        fig_vocab.update_layout(yaxis=dict(autorange="reversed"), height=320, template="plotly_white", margin=dict(l=40, r=20, t=40, b=30))
        st.plotly_chart(fig_vocab, use_container_width=True)

# ==================== TAB 2: MODEL DIAGNOSTICS & OPTIMAL K ====================
with tab2:
    st.markdown("### ⚙️ LDA Model Diagnostics & Hyperparameter Tuning")
    
    diag_c1, diag_c2 = st.columns([1, 1])
    with diag_c1:
        st.markdown("#### Probabilistic Model Parameters")
        st.markdown(f"""
        - **Variational Lower Bound (Log-Likelihood):** `{model_results['log_likelihood']}`
        - **Perplexity Score:** `{model_results['perplexity']}` *(lower perplexity indicates better predictive likelihood)*
        - **Intrinsic $U_{{mass}}$ Topic Coherence:** `{model_results['mean_coherence']}` *(higher/less negative indicates better semantic cohesion)*
        - **Document-Topic Prior ($\\alpha$):** `{alpha_prior}` *(Low values favor documents dominated by few topics)*
        - **Topic-Word Prior ($\\beta$):** `{beta_prior}` *(Low values favor topics dominated by few distinct words)*
        - **Optimization Algorithm:** Online Variational Bayes (Hoffman et al., 2010)
        """)
        
        with st.expander("📖 Mathematical Overview of Latent Dirichlet Allocation"):
            st.latex(r"P(\boldsymbol{W}, \boldsymbol{Z}, \boldsymbol{\theta}, \boldsymbol{\beta}) = \prod_{k=1}^K P(\boldsymbol{\beta}_k | \boldsymbol{\eta}) \prod_{d=1}^D P(\boldsymbol{\theta}_d | \boldsymbol{\alpha}) \prod_{n=1}^{N_d} P(z_{d,n} | \boldsymbol{\theta}_d) P(w_{d,n} | \boldsymbol{\beta}_{z_{d,n}})")
            st.markdown("""
            - $\\boldsymbol{\\alpha}$: Hyperparameter governing the Dirichlet prior on document-topic mixtures $\\boldsymbol{\\theta}_d$.
            - $\\boldsymbol{\\eta}$ ($\\beta$): Hyperparameter governing the Dirichlet prior on topic-word mixtures $\\boldsymbol{\\beta}_k$.
            - $z_{d,n}$: The latent topic assignment for the $n$-th word in document $d$.
            - $w_{d,n}$: The observed word token.
            """)

    with diag_c2:
        st.markdown("#### Optimal Topics Selection ($K$ Coherence Sweep)")
        st.caption("Evaluate how Topic Coherence ($U_{mass}$) varies across different topic counts $K$:")
        
        if st.button("🔍 Run Quick $K$ Coherence Sweep (K=2 to 8)"):
            with st.spinner("Sweeping topic counts $K$ and computing coherence..."):
                k_range = list(range(2, 9))
                coherences = []
                perplexities = []
                for k in k_range:
                    temp_lda = LDAEngine()
                    temp_lda.doc_term_matrix = lda_engine.doc_term_matrix
                    temp_lda.feature_names = lda_engine.feature_names
                    res = temp_lda.train(n_components=k, learning_method='online', max_iter=15, random_state=42)
                    coherences.append(res["mean_coherence"])
                    perplexities.append(res["perplexity"])
                
                sweep_df = pd.DataFrame({"K": k_range, "U_mass Coherence": coherences, "Perplexity": perplexities})
                fig_sweep = px.line(
                    sweep_df, x="K", y="U_mass Coherence", markers=True,
                    title="Topic Coherence vs Number of Topics (K)",
                    color_discrete_sequence=["#4F46E5"]
                )
                fig_sweep.update_layout(template="plotly_white", height=300, margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(fig_sweep, use_container_width=True)
                best_k = sweep_df.loc[sweep_df["U_mass Coherence"].idxmax(), "K"]
                st.success(f"Optimal topic count based on peak coherence: **K = {best_k}**")
        else:
            st.info("Click the button above to run a multi-K coherence comparison curve.")

# ==================== TAB 3: TOPIC-WORD DISTRIBUTIONS ====================
with tab3:
    st.markdown("### 🏷️ Topic-Word Probability Distributions: $P(w \\mid z_k)$")
    st.markdown("""
    Each latent topic is defined by a probability vector $\\boldsymbol{\\beta}_k$ over the entire vocabulary.
    Words with the highest probability values $P(w \\mid z_k)$ characterize the semantic theme of the topic.
    """)
    
    top_n_words = st.slider("Select Number of Top Words to Display per Topic", 5, 20, 10)
    
    # Render interactive Plotly subplots
    fig_topics = viz.plot_top_words_grid(topics_data, max_words=top_n_words)
    st.plotly_chart(fig_topics, use_container_width=True)
    
    # Topic Summary Table
    st.markdown("#### Structured Topic-Word Summary Table")
    topic_summary_rows = []
    for t in topics_data:
        topic_summary_rows.append({
            "Topic ID": t["topic_id"],
            "Suggested Label": t["label"],
            "Top Keywords (Descending Probability)": ", ".join(t["words"][:top_n_words]),
            "Max Word Probability": round(max(t["weights"]), 4)
        })
    df_topic_summary = pd.DataFrame(topic_summary_rows)
    st.dataframe(df_topic_summary, use_container_width=True)
    
    # Download Topic-Word Distributions CSV
    # Construct complete word probability DataFrame
    all_topic_words_list = []
    for t in topics_data:
        for w, weight in zip(t["words"][:top_n_words], t["weights"][:top_n_words]):
            all_topic_words_list.append({
                "topic_id": t["topic_id"],
                "topic_label": t["label"],
                "word": w,
                "probability_p_w_z": round(weight, 5)
            })
    df_all_tw = pd.DataFrame(all_topic_words_list)
    csv_tw = df_all_tw.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Topic-Word Distributions (CSV)",
        data=csv_tw,
        file_name="topic_word_distributions.csv",
        mime="text/csv"
    )

# ==================== TAB 4: DOCUMENT MIXTURES & DISTANCE MAP ====================
with tab4:
    st.markdown("### 📊 Document-Topic Distributions & Inter-Topic Semantic Space")
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        # Dominant Topic Pie / Donut
        fig_donut = viz.plot_topic_proportions(df_docs, topics_data)
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_d2:
        # 2D PCA Inter-topic Distance Map
        fig_intertopic = viz.plot_intertopic_map(
            lda_engine.topic_word_dist,
            lda_engine.doc_topic_dist,
            topics_data
        )
        if fig_intertopic:
            st.plotly_chart(fig_intertopic, use_container_width=True)
        else:
            st.info("Need at least 2 topics for inter-topic projection.")

    # Heatmap of document-topic distributions
    st.markdown("#### Document-Topic Mixture Matrix $\\boldsymbol{\\theta}_{d, k}$")
    sample_heatmap_size = st.slider("Number of Sample Documents to Display in Heatmap", 10, 80, 35)
    fig_heatmap = viz.plot_doc_topic_heatmap(lda_engine.doc_topic_dist, max_docs=sample_heatmap_size)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Document Explorer
    st.markdown("#### 🔎 Interactive Document Topic Explorer")
    selected_topic_filter = st.selectbox(
        "Filter Documents by Dominant Topic",
        ["All Topics"] + [t["label"] for t in topics_data]
    )
    
    filtered_df = df_docs.copy()
    if selected_topic_filter != "All Topics":
        topic_idx = int(selected_topic_filter.split(":")[0].replace("Topic", "").strip())
        filtered_df = filtered_df[filtered_df["dominant_topic"] == topic_idx]

    st.dataframe(
        filtered_df[["doc_id", "dominant_topic", "topic_weight", "snippet"]].head(25),
        use_container_width=True,
        height=280
    )
    
    # Download Document Assignments
    csv_doc_topics = df_docs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Document Topic Assignments (CSV)",
        data=csv_doc_topics,
        file_name="document_topic_assignments.csv",
        mime="text/csv"
    )

# ==================== TAB 5: LIVE INFERENCE PLAYGROUND ====================
with tab5:
    st.markdown("### 🎯 Real-Time News Article Topic Inference")
    st.markdown("""
    Test the trained Latent Dirichlet Allocation model on an unseen headline, excerpt, or custom news text.
    The model transforms the text into the learned vocabulary space and computes the posterior probability distribution $P(z \\mid \\text{new doc})$.
    """)
    
    # Sample news prompts
    sample_presets = {
        "🚀 Space & Astronomy": "Astronomers using space telescopes discovered orbital signatures of distant exoplanets and atmospheric water vapor around red dwarf stars.",
        "🏛️ Politics & Diplomacy": "Ambassadors met at the United Nations summit to draft ceasefire agreements and establish bilateral sanctions against rogue regimes.",
        "🏒 Sports & Athletics": "The star forward scored a hat-trick in the third period as the hockey team defended its championship title in sudden-death overtime.",
        "🧬 Healthcare & Medicine": "Clinical oncology trials confirmed that targeted monoclonal antibody therapy significantly reduced tumor proliferation and inflammatory biomarkers.",
        "💻 Technology & Computing": "Cloud architects deployed high-performance semiconductor chips with liquid cooling systems to accelerate neural network compiler benchmarks."
    }
    
    st.markdown("**Quick Preset Examples:**")
    preset_cols = st.columns(len(sample_presets))
    selected_preset_text = None
    for idx, (p_name, p_text) in enumerate(sample_presets.items()):
        if preset_cols[idx].button(p_name, use_container_width=True):
            st.session_state.custom_test_text = p_text

    default_test = st.session_state.get(
        "custom_test_text",
        "The space agency launched a commercial satellite mission equipped with infrared spectroscopy instruments."
    )
    
    user_test_input = st.text_area(
        "Enter or Paste News Article Excerpt:",
        value=default_test,
        height=120
    )
    
    if st.button("🔮 Infer Topic Mixture", type="primary"):
        if user_test_input.strip():
            topic_mixture, dom_topic, conf = lda_engine.predict_new_text(user_test_input, preprocessor)
            dom_label = topics_data[dom_topic]["label"]
            
            inf_col1, inf_col2 = st.columns([1, 2])
            with inf_col1:
                st.markdown(f"""
                <div style="background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 12px; padding: 20px; text-align: center; margin-top: 10px;">
                    <div style="color: #4F46E5; font-size: 13px; font-weight: 700; text-transform: uppercase;">Dominant Topic Assignment</div>
                    <div style="color: #1E1B4B; font-size: 20px; font-weight: 700; margin: 8px 0;">{dom_label}</div>
                    <div style="color: #059669; font-size: 28px; font-weight: 800;">{conf*100:.1f}%</div>
                    <div style="color: #6B7280; font-size: 12px;">Posterior Confidence Weight</div>
                </div>
                """, unsafe_allow_html=True)
                
            with inf_col2:
                fig_inf = viz.plot_live_inference(topic_mixture, topics_data)
                st.plotly_chart(fig_inf, use_container_width=True)
        else:
            st.warning("Please enter text to infer.")

# ==================== TAB 6: TOPIC INTERPRETATION REPORT ====================
with tab6:
    st.markdown("### 📝 Topic Interpretation & Technical Deliverable Report")
    
    # Generate full markdown report
    full_report_md = generate_topic_report(
        corpus_stats=corpus_stats,
        model_stats=model_results,
        topics_data=topics_data,
        df_docs=df_docs
    )
    
    # Download Button
    st.download_button(
        label="📥 Download Full Report (Markdown .md)",
        data=full_report_md.encode('utf-8'),
        file_name="TopicScope_LDA_Interpretation_Report.md",
        mime="text/markdown",
        help="Download this comprehensive technical report in GitHub-flavored Markdown"
    )
    
    st.markdown("---")
    st.markdown(full_report_md)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 13px; padding: 12px;">
    TopicScope Studio • Latent Dirichlet Allocation (LDA) for News Media & Journalism • Built with Streamlit, Scikit-Learn & Plotly
</div>
""", unsafe_allow_html=True)
