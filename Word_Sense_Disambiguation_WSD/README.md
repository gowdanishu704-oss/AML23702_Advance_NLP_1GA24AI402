# 🏛️ LexiFinance AI: Word Sense Disambiguation & Induction System

An advanced Word Sense Disambiguation (WSD) and Word Sense Induction (WSI) system built in **Streamlit** using **WordNet**, **Lexical Relations**, and **Scikit-Learn**, tailored for **Financial Intelligence & Business News NLP**.

---

## 📌 Problem & Domain Motivation
Financial and corporate text is filled with polysemous keywords that mean drastically different things depending on context:
- **"bank"**: *Financial Institution* (`bank.n.01`) vs. *River Bank* (`bank.n.02`)
- **"interest"**: *Borrowing Charge / Rate* (`interest.n.03`) vs. *Curiosity / Concern* (`interest.n.01`) vs. *Equity Stake* (`interest.n.05`)
- **"stock"**: *Capital Stock / Shares* (`stock.n.01`) vs. *Inventory Supply* (`stock.n.04`) vs. *Broth / Soup* (`stock.n.05`)
- **"capital"**: *Financial Wealth / Assets* (`capital.n.01`) vs. *Government Seat / City* (`capital.n.03`)
- **"bond"**: *Debt Certificate* (`bond.n.01`) vs. *Emotional Tie* (`bond.n.02`)
- **"yield"**: *Investment Return / Dividend* (`yield.n.01`) vs. *Surrender / Give Way* (`yield.v.01`)
- **"plant"**: *Manufacturing Facility* (`plant.n.02`) vs. *Botanical Flora* (`plant.n.01`)
- **"balance"**: *Account Remainder* (`balance.n.05`) vs. *Physical Equilibrium* (`balance.n.01`)
- **"credit"**: *Financial Loan / Limit* (`credit.n.02`) vs. *Praise / Acknowledgment* (`credit.n.01`)

Failure to disambiguate these words leads to severe errors in automated algorithmic trading, monetary policy sentiment tracking, and risk assessment pipelines.

---

## 🚀 Key Features

### 1. 🎯 Live WSD Disambiguation Lab
- Real-time sentence disambiguation across single or multiple algorithms simultaneously.
- Supported Algorithms:
  - **Original Lesk Algorithm** (Gloss overlap)
  - **Extended / Adapted Lesk** (Gloss + Hypernyms + Hyponyms + Meronyms)
  - **Cosine TF-IDF Vector Space** (Scikit-Learn TF-IDF representation)
  - **Wu-Palmer Semantic Path Distance** (Taxonomic depth)
  - **Supervised Machine Learning** (Scikit-Learn TF-IDF + Logistic Regression/NB/SVC)
- Visual breakdown: Extracted context tokens, overlap words with WordNet definitions, candidate ranking bar charts.

### 2. 🕸️ WordNet Lexical Relations & Taxonomy Visualizer
- Deep-dive lexical relation extraction:
  - **Hypernyms** (Is-A general concepts)
  - **Hyponyms** (Subtypes & specific categories)
  - **Meronyms & Holonyms** (Part-of and Whole-of structures)
  - **Coordinate Terms** (Sister senses sharing parents)
  - **Antonyms** (Semantic opposites)
  - **Derivational Forms** (Morphological connections, e.g. bank -> banker)
- **Interactive 2D Plotly Network Graph** visualizing synsets and relational links.
- Exportable full Sense Inventory as CSV/JSON.

### 3. 🤖 Supervised WSD Studio (Scikit-Learn)
- Local context window extraction (±5 tokens around target word).
- TF-IDF unigram and bigram feature representations.
- Model training & Cross-Validation (Logistic Regression, Multinomial Naive Bayes, Linear SVC).
- Confusion matrix, precision, recall, and **top learned informative features/n-grams** per sense.
- Live testing on user-entered sentences.

### 4. 🔬 Unsupervised Word Sense Induction (WSI)
- Discovers latent senses without predefined dictionaries using K-Means clustering over contextual TF-IDF vectors.
- Automatically clusters unlabeled domain text and inspects cluster centroids and distinctive keywords.

### 5. 📊 Benchmark Suite & Short Analysis Report
- Automated benchmark across 69 annotated domain sentences.
- Multi-domain performance breakdown (Finance vs. General).
- Downloadable formal academic report in Markdown and CSV exports.

---

## 📁 Repository Structure
```
├── app.py                             # Main Streamlit web application
├── domain_dataset.py                  # Curated financial domain dataset & sense annotations
├── lexical_relations.py               # WordNet taxonomy explorer & Plotly graph builder
├── wsd_engine.py                      # WSD algorithms (Lesk, Extended Lesk, Cosine, Wu-Palmer, WSI)
├── supervised_model.py                # Scikit-Learn supervised WSD training & inference pipeline
├── report_generator.py                # Benchmark evaluator & Short Analysis Report compiler
├── sense_inventory.csv                # Exported WordNet sense inventory deliverable
├── wsd_evaluation_results.csv         # Exported sentence-level WSD predictions deliverable
├── WSD_Financial_Short_Analysis_Report.md # Exported academic lab analysis report
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation
```

---

## 🛠️ How to Run the Project

1. Open a terminal in the project directory:
   ```bash
   streamlit run app.py
   ```
2. The application will open automatically in your browser at `http://localhost:8501`.
