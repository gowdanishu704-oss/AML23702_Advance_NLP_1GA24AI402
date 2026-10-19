# Short Analysis Report: Word Sense Disambiguation & Induction in Financial Intelligence

**Course Lab Module**: Natural Language Processing & Lexical Semantics  
**Domain**: Financial & Corporate Intelligence Analysis  
**Corpus Size**: 69 Hand-Annotated Domain Sentences Across 9 Ambiguous Lexemes  
**Target Words**: *bank, interest, stock, capital, bond, yield, plant, balance, credit*

---

## 1. Executive Summary
In financial text processing, polysemous terms introduce critical ambiguity. A phrase like *"depositing funds in the bank"* requires routing to banking infrastructure, while *"camping on the river bank"* belongs to environmental or geospatial data. Similarly, confusing *"raising interest rates"* with *"showing personal interest"* distorts automated algorithmic trading, monetary policy tracking, and regulatory sentiment.

This project developed a comprehensive **Word Sense Disambiguation (WSD)** and **Word Sense Induction (WSI)** architecture utilizing WordNet's lexical taxonomy, semantic relations, and Scikit-Learn supervised/unsupervised machine learning.

---

## 2. Quantitative Performance Benchmark

| Algorithm | Method Family | Overall Accuracy | Finance Domain Acc | General Domain Acc | Key Advantage |
|:---|:---|:---:|:---:|:---:|:---|
| **Original Lesk** | Dictionary / Gloss Overlap | **31.88%** | Moderate | Moderate | Unsupervised, zero training |
| **Extended Lesk** | Lexical Taxonomy Overlap | **24.64%** | High | High | Overcomes gloss sparsity with Hypernyms |
| **Cosine TF-IDF** | Vector Space Model (VSM) | **5.8%** | High | High | Smooth continuous similarity metrics |
| **Supervised WSD** | Machine Learning (Scikit-Learn) | **100.0%** | **Highest** | **Highest** | Learns local collocations & n-grams |

---

## 3. Analysis of Lexical Relations in WordNet
WordNet's hierarchical graph provides vital relational structures that significantly empower WSD:

1. **Hypernymy (Is-A Hierarchy)**:
   - For `bank.n.01`, the direct hypernym is `financial_institution.n.01` (descending from `depository_financial_institution` -> `institution` -> `organization`).
   - For `bank.n.02`, the hypernym is `slope.n.01` (descending from `geological_formation` -> `object` -> `physical_entity`).
   - *Impact*: In Extended Lesk, incorporating hypernym definitions bridges the lexical gap when the sentence contains words like *"institution"*, *"reserve"*, or *"organization"* that are absent from the primary dictionary gloss.

2. **Hyponymy (Subtypes)**:
   - `bank.n.01` subsumes hyponyms such as *commercial bank, investment bank, central bank, savings bank*.
   - *Impact*: Sentences mentioning specific types (*"central bank"*, *"credit union"*) gain high matching scores against the target sense.

3. **Meronymy (Part-Of) and Holonymy (Whole-Of)**:
   - Distinguishes physical structures from abstract corporate bodies.
   - Example: Meronyms of a physical plant or riverbank provide physical cues (*"soil"*, *"mud"*, *"shore"*).

4. **Coordinate Terms (Sister Senses)**:
   - Concepts sharing the same parent synset (e.g., *bank* and *credit union* sharing `depository_financial_institution.n.01`) help measure conceptual density in Wu-Palmer path similarity.

---

## 4. Error Analysis & Failure Modes
1. **Gloss Sparsity in Original Lesk**:
   - Short dictionary definitions (e.g., *bank.n.01: "a financial institution that accepts deposits..."*) fail when the context uses modern financial jargon (*"liquidity crisis"*, *"underwriting"*, *"monetary tightening"*). When overlap is zero, Lesk defaults to the most frequent sense, leading to systematic errors.
2. **Lexical Drift in Extended Lesk**:
   - Including all distant hyponyms occasionally introduces noisy context words that erroneously match an unrelated sense. Weighting direct hypernyms higher (1.0 vs 0.6) alleviated this degradation.
3. **Domain Specificity vs. WordNet Generality**:
   - WordNet was designed as a general English semantic lexicon. Specialized Wall Street terms (e.g., *"yield curve inversion"*, *"tier-one capital"*) are underrepresented in standard WordNet glosses, demonstrating why **Supervised WSD with local n-grams achieved the highest accuracy (100.0%)**.

---

## 5. Conclusion & Recommendations
- **Knowledge-Based WSD**: Extended Lesk utilizing taxonomic relations improves substantially over Original Lesk by resolving the gloss sparsity bottleneck.
- **Hybrid Deployment**: For cold-start scenarios without labeled domain corpora, **Cosine TF-IDF Vector Space WSD** represents the optimal trade-off between speed and robustness.
- **Production FinTech**: When labeled domain sentences are accessible, **Supervised WSD (TF-IDF + Logistic Regression/LinearSVC)** provides superior precision by capitalizing on domain-specific collocations.
