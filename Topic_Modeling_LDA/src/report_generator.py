import datetime
import numpy as np
import pandas as pd

def generate_topic_report(
    corpus_stats: dict,
    model_stats: dict,
    topics_data: list[dict],
    df_docs: pd.DataFrame
) -> str:
    """
    Generates a structured, professional Topic Interpretation Report
    in Markdown format suitable for download and presentation.
    """
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    n_topics = model_stats.get("n_components", len(topics_data))
    num_docs = corpus_stats.get("num_docs", len(df_docs))
    vocab_size = corpus_stats.get("vocab_size", 0)
    sparsity = corpus_stats.get("sparsity_pct", 0)
    perplexity = model_stats.get("perplexity", 0)
    log_likelihood = model_stats.get("log_likelihood", 0)
    coherence = model_stats.get("mean_coherence", 0)

    # Document proportions per topic
    dominant_counts = df_docs["dominant_topic"].value_counts()
    
    report_lines = []
    report_lines.append("# TopicScope: Latent Dirichlet Allocation (LDA) Interpretation Report")
    report_lines.append(f"**Domain:** News Media & Journalism | **Date:** {date_str}  \n")
    report_lines.append("---")
    
    # 1. Executive Summary
    report_lines.append("## 1. Executive Summary")
    report_lines.append(
        f"This report presents the results of unsupervised topic modeling using **Latent Dirichlet Allocation (LDA)** "
        f"applied to a collection of **{num_docs:,} news documents**. Utilizing an online Variational Bayes optimization method, "
        f"the model successfully discovered **{n_topics} latent semantic topics** from a clean vocabulary of **{vocab_size:,} terms** "
        f"(matrix sparsity: {sparsity}%). The model achieved a perplexity score of **{perplexity}** and an average $U_{{mass}}$ "
        f"topic coherence score of **{coherence}**, indicating sharp thematic separation and high interpretability."
    )
    report_lines.append("")

    # 2. Mathematical Framework
    report_lines.append("## 2. Probabilistic Formulation")
    report_lines.append(
        "Latent Dirichlet Allocation represents each document $d \\in \\mathcal{D}$ as a finite mixture over an underlying "
        "set of $K$ latent topics, where each topic $k$ is characterized by a multinomial distribution over the vocabulary $\\mathcal{V}$:\n\n"
        "- **Document-Topic Distribution:** $\\boldsymbol{\\theta}_d \\sim \\text{Dirichlet}(\\boldsymbol{\\alpha})$  \n"
        "- **Topic-Word Distribution:** $\\boldsymbol{\\beta}_k \\sim \\text{Dirichlet}(\\boldsymbol{\\eta})$  \n"
        "- **Per-Word Topic Assignment:** $z_{d,n} \\sim \\text{Multinomial}(\\boldsymbol{\\theta}_d)$  \n"
        "- **Observed Word Generation:** $w_{d,n} \\sim \\text{Multinomial}(\\boldsymbol{\\beta}_{z_{d,n}})$  \n\n"
        "Posterior inference is executed via **Online Variational Bayes**, approximating intractable true posterior distributions "
        "by maximizing the Evidence Lower Bound (ELBO)."
    )
    report_lines.append("")

    # 3. Discovered Topics Overview Table
    report_lines.append("## 3. Discovered Latent Topics")
    report_lines.append("| Topic ID | Semantic Label | Top Keywords (Descending P(w|z)) | Corpus Prevalence |")
    report_lines.append("|:---:|:---|:---|:---:|")

    for t in topics_data:
        k = t["topic_id"]
        label = t["label"]
        words_preview = ", ".join(t["words"][:7])
        doc_count = dominant_counts.get(k, 0)
        pct = (doc_count / num_docs) * 100.0 if num_docs > 0 else 0.0
        report_lines.append(f"| Topic {k} | **{label}** | `{words_preview}` | {doc_count} docs ({pct:.1f}%) |")
    report_lines.append("")

    # 4. Detailed Qualitative Analysis
    report_lines.append("## 4. In-Depth Topic Breakdown & Qualitative Analysis")
    for t in topics_data:
        k = t["topic_id"]
        label = t["label"]
        report_lines.append(f"### {label}")
        report_lines.append("Top terms and their estimated conditional probabilities $P(w \\mid z_k)$:")
        report_lines.append("")
        for w, weight in zip(t["words"][:8], t["weights"][:8]):
            report_lines.append(f"- **{w}**: {weight:.4f}")
        report_lines.append("")
        report_lines.append(
            f"*Interpretation:* The co-occurrence patterns in Topic {k} distinctly capture discourse centered on "
            f"**{', '.join(t['words'][:4])}**. Articles aligned with this topic exhibit a strong focus on this domain's "
            "specialized vocabulary and domain-specific terminology."
        )
        report_lines.append("")

    # 5. Diagnostic Evaluation
    report_lines.append("## 5. Model Diagnostics & Validation")
    report_lines.append(
        f"- **Perplexity:** {perplexity} (Lower perplexity indicates higher predictive probability on the document collection).  \n"
        f"- **Log-Likelihood:** {log_likelihood} (Overall fit to observed document-term frequencies).  \n"
        f"- **Mean U_mass Coherence:** {coherence} (Negative log-ratio metric measuring co-occurrence consistency among top terms).  \n"
        f"- **Document-Term Matrix Sparsity:** {sparsity}% (Reflects expected textual sparsity in natural language corpora)."
    )
    report_lines.append("")

    # 6. Practical Applications in Journalism & Media
    report_lines.append("## 6. Strategic & Editorial Applications")
    report_lines.append(
        "1. **Automated Content Tagging & Taxonomy:** Articles can be automatically classified into multi-topic tags without manual editorial overhead.\n"
        "2. **Editorial Balance & Agenda Tracking:** Newsrooms can monitor shifts in topic proportions over time (e.g., detecting if sports or politics is dominating news coverage).\n"
        "3. **Semantic Recommendation Systems:** Readers browsing articles dominated by Topic $k$ can receive personalized suggestions based on document-topic cosine similarity $\\boldsymbol{\\theta}_{d_1} \\cdot \\boldsymbol{\\theta}_{d_2}$.\n"
        "4. **Real-Time Ingestion & Triage:** Incoming wires and press releases can be filtered directly to appropriate desk editors based on posterior probability thresholds."
    )
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Report generated automatically by TopicScope LDA Studio.*")

    return "\n".join(report_lines)
