import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDAEngine:
    """
    Manages Document-Term representation, Latent Dirichlet Allocation (LDA)
    training using Variational Bayes, probabilistic distribution extraction,
    and inference diagnostics.
    """
    def __init__(self):
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
        self.doc_term_matrix = None
        self.topic_word_dist = None
        self.doc_topic_dist = None

    def build_dtm(
        self,
        clean_texts: list[str],
        min_df: int = 2,
        max_df: float = 0.85,
        max_features: int = 1500,
        ngram_range: tuple = (1, 1)
    ):
        """
        Builds Document-Term Matrix (Bag of Words) representation.
        Calculates vocabulary size and matrix sparsity.
        """
        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.doc_term_matrix = self.vectorizer.fit_transform(clean_texts)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Matrix sparsity
        total_elements = self.doc_term_matrix.shape[0] * self.doc_term_matrix.shape[1]
        non_zeros = self.doc_term_matrix.nnz
        sparsity = (1.0 - (non_zeros / total_elements)) * 100.0 if total_elements > 0 else 0.0

        stats = {
            "num_docs": self.doc_term_matrix.shape[0],
            "vocab_size": self.doc_term_matrix.shape[1],
            "total_tokens": int(self.doc_term_matrix.sum()),
            "sparsity_pct": round(sparsity, 2),
            "non_zero_entries": non_zeros
        }
        return self.doc_term_matrix, self.vectorizer, stats

    def train(
        self,
        n_components: int = 5,
        learning_method: str = 'online',
        max_iter: int = 25,
        doc_topic_prior: float = None,  # alpha
        topic_word_prior: float = None,  # beta / eta
        random_state: int = 42
    ):
        """
        Trains Latent Dirichlet Allocation model via Variational Bayes.
        Extracts:
        - beta (Topic-Word distribution: P(word | topic))
        - theta (Document-Topic distribution: P(topic | document))
        - Perplexity & Log-Likelihood
        """
        self.lda_model = LatentDirichletAllocation(
            n_components=n_components,
            learning_method=learning_method,
            max_iter=max_iter,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            random_state=random_state,
            evaluate_every=5
        )
        
        # Fit model and infer document-topic distribution
        self.doc_topic_dist = self.lda_model.fit_transform(self.doc_term_matrix)
        
        # Extract and normalize topic-word distribution (beta)
        # components_[k, w] is pseudo-count of word w in topic k
        components = self.lda_model.components_
        self.topic_word_dist = components / components.sum(axis=1)[:, np.newaxis]
        
        # Model diagnostics
        perplexity = self.lda_model.perplexity(self.doc_term_matrix)
        log_likelihood = self.lda_model.score(self.doc_term_matrix)
        coherence = self.compute_umass_coherence(top_n=10)

        results = {
            "n_components": n_components,
            "perplexity": round(perplexity, 2),
            "log_likelihood": round(log_likelihood, 2),
            "mean_coherence": round(coherence, 3),
            "doc_topic_dist": self.doc_topic_dist,
            "topic_word_dist": self.topic_word_dist
        }
        return results

    def compute_umass_coherence(self, top_n: int = 10) -> float:
        """
        Computes intrinsic U_mass Topic Coherence based on document co-occurrence.
        Higher (less negative) coherence indicates better semantic cohesion.
        """
        if self.doc_term_matrix is None or self.topic_word_dist is None:
            return 0.0
            
        dtm_binary = (self.doc_term_matrix > 0).astype(int)
        num_docs = dtm_binary.shape[0]
        coherence_scores = []
        
        for k in range(self.topic_word_dist.shape[0]):
            top_word_indices = np.argsort(self.topic_word_dist[k])[::-1][:top_n]
            topic_score = 0.0
            pair_count = 0
            
            for m in range(1, len(top_word_indices)):
                w_m = top_word_indices[m]
                # Document frequency of word m
                col_m = dtm_binary[:, w_m]
                d_m = col_m.sum()
                
                for l in range(0, m):
                    w_l = top_word_indices[l]
                    col_l = dtm_binary[:, w_l]
                    # Co-occurrence in same document
                    d_ml = (col_m.multiply(col_l)).sum()
                    
                    # U_mass formula: log((D(w_m, w_l) + 1) / D(w_l))
                    pair_score = np.log((d_ml + 1.0) / (d_l := (col_l.sum() + 1.0)))
                    topic_score += pair_score
                    pair_count += 1
                    
            if pair_count > 0:
                coherence_scores.append(topic_score / pair_count)
                
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0

    def get_topic_keywords(self, top_n: int = 10) -> list[dict]:
        """Returns structured top keywords and probability values per topic."""
        topics = []
        for k, topic_probs in enumerate(self.topic_word_dist):
            top_indices = np.argsort(topic_probs)[::-1][:top_n]
            top_words = self.feature_names[top_indices]
            top_weights = topic_probs[top_indices]
            
            # Suggest a semantic label based on top 3 words
            label_hint = " & ".join([w.title() for w in top_words[:3]])
            topics.append({
                "topic_id": k,
                "label": f"Topic {k}: {label_hint}",
                "words": list(top_words),
                "weights": [float(w) for w in top_weights]
            })
        return topics

    def get_document_assignments(self, raw_texts: list[str], categories: list[str] = None) -> pd.DataFrame:
        """Constructs DataFrame showing dominant topic and distribution per document."""
        dominant_topics = np.argmax(self.doc_topic_dist, axis=1)
        max_probs = np.max(self.doc_topic_dist, axis=1)
        
        df_docs = pd.DataFrame({
            "doc_id": np.arange(len(raw_texts)),
            "dominant_topic": dominant_topics,
            "topic_weight": np.round(max_probs, 4),
            "snippet": [t[:180] + "..." if len(t) > 180 else t for t in raw_texts]
        })
        if categories is not None and len(categories) == len(raw_texts):
            df_docs["original_category"] = categories
            
        # Add probability for each topic column
        for k in range(self.doc_topic_dist.shape[1]):
            df_docs[f"topic_{k}_prob"] = np.round(self.doc_topic_dist[:, k], 4)
            
        return df_docs

    def predict_new_text(self, text: str, preprocessor) -> tuple[np.ndarray, int, float]:
        """
        Calculates posterior topic mixture for an unseen test document:
        P(z | text_new)
        """
        clean_text = preprocessor.preprocess_document(text)
        bow_vector = self.vectorizer.transform([clean_text])
        topic_mixture = self.lda_model.transform(bow_vector)[0]
        dominant_topic = int(np.argmax(topic_mixture))
        confidence = float(np.max(topic_mixture))
        return topic_mixture, dominant_topic, confidence
