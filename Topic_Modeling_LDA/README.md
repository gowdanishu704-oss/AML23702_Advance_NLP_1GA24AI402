# TopicScope: Latent Dirichlet Allocation (LDA) Topic Modeling Studio

An interactive, end-to-end **Latent Dirichlet Allocation (LDA)** machine learning and natural language processing application built with **Streamlit**, **Scikit-Learn**, **NLTK**, and **Plotly**.

Applied to the **News Media & Journalism** domain to discover latent semantic editorial themes across document collections using probabilistic Variational Bayes and sampling-based topic distributions.

---

## 🚀 Key Features & Expected Outcomes

1. **Text Preprocessing Pipeline:**
   - Case normalization and regex stripping of HTML/URLs, emails, and non-alphabetic noise.
   - Domain-specific news boilerplate and standard English stopword filtering.
   - WordNet Lemmatization for root word extraction.
   - Document-Term Matrix (DTM) construction with vocabulary pruning (`min_df`, `max_df`, `max_features`, and N-grams).

2. **Probabilistic Latent Dirichlet Allocation (LDA):**
   - Implements online Variational Bayes optimization approximating the Dirichlet-Multinomial generative process.
   - Configurable hyperparameter priors: $\alpha$ (document-topic sparsity) and $\beta$ / $\eta$ (topic-word sparsity).
   - Intrinsic $U_{\text{mass}}$ topic coherence scoring and model perplexity tracking across different $K$ topic counts.

3. **Probabilistic Topic-Word Distributions:**
   - Extraction of $\boldsymbol{\beta}_k = P(w \mid z_k)$ vectors.
   - Interactive horizontal probability bar charts and ranked top-word lists per topic.
   - Automatic semantic title tagging based on dominant lexical weights.

4. **Document Mixture Analysis & Visualizations:**
   - Inter-topic distance map via 2D Principal Component Analysis (PCA) projection.
   - Document-Topic mixture heatmap ($\boldsymbol{\theta}_{d, k}$).
   - Dominant topic distribution across the corpus.

5. **Real-Time Live Article Inference Playground:**
   - Paste any new news snippet or headline to instantly infer its posterior topic mixture $\boldsymbol{\theta}_{\text{new}}$ with confidence scores and animated visual breakdowns.

6. **Automated Technical Deliverables:**
   - Preprocessed corpus (`data/preprocessed_corpus.csv`).
   - Topic-word probability distributions (`outputs/topic_word_distributions.csv`).
   - Serialized trained LDA model and vectorizer (`models/lda_model.pkl`, `models/vectorizer.pkl`).
   - Comprehensive **Topic Interpretation Report** in Markdown (`Topic_Interpretation_Report.md`).

---

## 📁 Repository Structure

```
Latent/
├── app.py                         # Interactive Streamlit Web Application
├── generate_corpus.py             # Generates benchmark multi-domain news corpus
├── run_pipeline.py                # Headless execution script to produce all deliverables
├── run_app.bat                    # One-click Windows launcher for Streamlit
├── Topic_Interpretation_Report.md  # Detailed qualitative and quantitative topic report
├── data/
│   ├── news_corpus.csv            # Benchmark news document collection
│   └── preprocessed_corpus.csv    # Cleaned, tokenized, and lemmatized corpus
├── models/
│   ├── lda_model.pkl              # Saved Scikit-Learn LDA model
│   └── vectorizer.pkl             # Saved CountVectorizer vocabulary representation
├── outputs/
│   └── topic_word_distributions.csv# Discovered topics with word probabilities
└── src/
    ├── __init__.py
    ├── preprocessor.py            # Text cleaning, tokenization, lemmatization
    ├── lda_engine.py              # DTM builder, LDA trainer, coherence, inference
    ├── visualizations.py          # Plotly subplots, heatmaps, inter-topic maps
    └── report_generator.py        # Automated Markdown report synthesis
```

---

## 🛠️ How to Run

### 1. Launch the Streamlit Web Application
Double-click `run_app.bat` or run:
```bash
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 2. Run the Headless Pipeline (Generate All Deliverables)
```bash
python run_pipeline.py
```
