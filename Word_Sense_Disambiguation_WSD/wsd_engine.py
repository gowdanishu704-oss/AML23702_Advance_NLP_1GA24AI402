"""
WSD Engine: Algorithms for Word Sense Disambiguation & Word Sense Induction
Implements:
1. Original Lesk Algorithm (Gloss overlap)
2. Extended / Adapted Lesk (Gloss + Examples + Hypernyms + Hyponyms overlap)
3. TF-IDF Vector Space Cosine Similarity WSD (Scikit-Learn)
4. Wu-Palmer Path Semantic Similarity WSD
5. Unsupervised Word Sense Induction (WSI) using TF-IDF & K-Means Clustering
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np


class WSDEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words("english"))
        # Add punctuation and common noise
        self.stop_words.update([",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'", "``", "''", "--", "-"])

    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize, lowercase, remove stopwords/punctuation, and lemmatize."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = word_tokenize(text)
        cleaned = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1 and not token.isdigit()
        ]
        return cleaned

    # -------------------------------------------------------------
    # 1. Original Lesk Algorithm
    # -------------------------------------------------------------
    def original_lesk(self, sentence: str, target_word: str, pos: Optional[str] = None) -> Dict[str, Any]:
        """
        Original Lesk algorithm: Computes raw token overlap between context words
        and the candidate synset's definition.
        """
        synsets = wn.synsets(target_word.lower(), pos=pos)
        if not synsets:
            return {"predicted_synset": None, "confidence": 0.0, "details": [], "algorithm": "Original Lesk"}

        context_tokens = set(self.preprocess_text(sentence))
        target_lemmas = {self.lemmatizer.lemmatize(target_word.lower()), target_word.lower()}
        context_tokens = context_tokens - target_lemmas

        best_synset = None
        max_overlap = -1
        candidates_info = []

        for syn in synsets:
            gloss_tokens = set(self.preprocess_text(syn.definition()))
            overlap = context_tokens.intersection(gloss_tokens)
            score = len(overlap)

            candidates_info.append({
                "synset": syn.name(),
                "pos": syn.pos(),
                "definition": syn.definition(),
                "overlap_words": list(overlap),
                "score": score
            })

            if score > max_overlap:
                max_overlap = score
                best_synset = syn

        # Default fallback to first synset (most frequent) if zero overlap
        if max_overlap <= 0:
            best_synset = synsets[0]
            confidence = 0.2
        else:
            total_score = sum(c["score"] for c in candidates_info)
            confidence = round(max_overlap / (total_score if total_score > 0 else 1), 3)

        candidates_info.sort(key=lambda x: x["score"], reverse=True)
        return {
            "predicted_synset": best_synset.name(),
            "best_synset_obj": best_synset,
            "definition": best_synset.definition(),
            "confidence": confidence,
            "max_overlap": max(0, max_overlap),
            "candidates": candidates_info,
            "context_tokens": list(context_tokens),
            "algorithm": "Original Lesk"
        }

    # -------------------------------------------------------------
    # 2. Extended / Adapted Lesk Algorithm (with Lexical Relations)
    # -------------------------------------------------------------
    def extended_lesk(self, sentence: str, target_word: str, pos: Optional[str] = None) -> Dict[str, Any]:
        """
        Extended Lesk algorithm: Augments synset gloss with examples, hypernyms,
        hyponyms, and meronym glosses to resolve gloss sparsity.
        """
        synsets = wn.synsets(target_word.lower(), pos=pos)
        if not synsets:
            return {"predicted_synset": None, "confidence": 0.0, "details": [], "algorithm": "Extended Lesk"}

        context_tokens = set(self.preprocess_text(sentence))
        target_lemmas = {self.lemmatizer.lemmatize(target_word.lower()), target_word.lower()}
        context_tokens = context_tokens - target_lemmas

        best_synset = None
        max_score = -1.0
        candidates_info = []

        for syn in synsets:
            # 1. Base gloss & examples (weight 1.0)
            base_tokens = set(self.preprocess_text(syn.definition() + " " + " ".join(syn.examples())))
            base_overlap = context_tokens.intersection(base_tokens)

            # 2. Hypernyms glosses (weight 0.6)
            hyper_text = " ".join([h.definition() for h in syn.hypernyms()])
            hyper_tokens = set(self.preprocess_text(hyper_text))
            hyper_overlap = context_tokens.intersection(hyper_tokens) - base_tokens

            # 3. Hyponyms glosses (weight 0.4)
            hypo_text = " ".join([h.definition() for h in syn.hyponyms()[:5]])
            hypo_tokens = set(self.preprocess_text(hypo_text))
            hypo_overlap = context_tokens.intersection(hypo_tokens) - base_tokens - hyper_tokens

            # 4. Meronyms glosses (weight 0.4)
            mero_text = " ".join([m.definition() for m in (syn.part_meronyms() + syn.member_meronyms())[:4]])
            mero_tokens = set(self.preprocess_text(mero_text))
            mero_overlap = context_tokens.intersection(mero_tokens) - base_tokens - hyper_tokens - hypo_tokens

            weighted_score = (
                len(base_overlap) * 1.0 +
                len(hyper_overlap) * 0.6 +
                len(hypo_overlap) * 0.4 +
                len(mero_overlap) * 0.4
            )

            all_overlap = list(base_overlap.union(hyper_overlap).union(hypo_overlap).union(mero_overlap))

            candidates_info.append({
                "synset": syn.name(),
                "pos": syn.pos(),
                "definition": syn.definition(),
                "overlap_words": all_overlap,
                "base_overlap": list(base_overlap),
                "hyper_overlap": list(hyper_overlap),
                "score": round(weighted_score, 2)
            })

            if weighted_score > max_score:
                max_score = weighted_score
                best_synset = syn

        if max_score <= 0:
            best_synset = synsets[0]
            confidence = 0.25
        else:
            total_score = sum(c["score"] for c in candidates_info)
            confidence = round(max_score / (total_score if total_score > 0 else 1), 3)

        candidates_info.sort(key=lambda x: x["score"], reverse=True)
        return {
            "predicted_synset": best_synset.name(),
            "best_synset_obj": best_synset,
            "definition": best_synset.definition(),
            "confidence": confidence,
            "max_overlap": round(max(0.0, max_score), 2),
            "candidates": candidates_info,
            "context_tokens": list(context_tokens),
            "algorithm": "Extended / Adapted Lesk"
        }

    # -------------------------------------------------------------
    # 3. Vector Space Cosine Similarity WSD (Scikit-Learn TF-IDF)
    # -------------------------------------------------------------
    def cosine_tfidf_wsd(self, sentence: str, target_word: str, pos: Optional[str] = None) -> Dict[str, Any]:
        """
        Cosine Similarity WSD using Scikit-Learn TF-IDF.
        Embeds the sentence context and synset signatures into TF-IDF vector space
        and scores by cosine similarity.
        """
        synsets = wn.synsets(target_word.lower(), pos=pos)
        if not synsets:
            return {"predicted_synset": None, "confidence": 0.0, "details": [], "algorithm": "Cosine TF-IDF"}

        # Build context document
        context_clean = " ".join(self.preprocess_text(sentence))

        # Build synset documents (Gloss + Examples + Hypernym lemmas)
        synset_docs = []
        for syn in synsets:
            doc_parts = [syn.definition()]
            doc_parts.extend(syn.examples())
            for h in syn.hypernyms():
                doc_parts.append(h.definition())
                doc_parts.extend([l.name().replace("_", " ") for l in h.lemmas()])
            for l in syn.lemmas():
                doc_parts.append(l.name().replace("_", " "))
            synset_docs.append(" ".join(self.preprocess_text(" ".join(doc_parts))))

        corpus = [context_clean] + synset_docs

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            context_vec = tfidf_matrix[0]
            synset_vecs = tfidf_matrix[1:]

            similarities = cosine_similarity(context_vec, synset_vecs).flatten()
        except Exception:
            similarities = np.zeros(len(synsets))

        candidates_info = []
        for idx, syn in enumerate(synsets):
            sim_score = float(similarities[idx])
            candidates_info.append({
                "synset": syn.name(),
                "pos": syn.pos(),
                "definition": syn.definition(),
                "score": round(sim_score, 4)
            })

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_synset = synsets[best_idx]

        if best_score <= 0:
            best_synset = synsets[0]
            confidence = 0.25
        else:
            total_sim = float(np.sum(similarities))
            confidence = round(best_score / (total_sim if total_sim > 0 else 1.0), 3)

        candidates_info.sort(key=lambda x: x["score"], reverse=True)
        return {
            "predicted_synset": best_synset.name(),
            "best_synset_obj": best_synset,
            "definition": best_synset.definition(),
            "confidence": confidence,
            "cosine_similarity": round(best_score, 4),
            "candidates": candidates_info,
            "algorithm": "Cosine TF-IDF Similarity"
        }

    # -------------------------------------------------------------
    # 4. Wu-Palmer Semantic Path Similarity WSD
    # -------------------------------------------------------------
    def wup_path_wsd(self, sentence: str, target_word: str, pos: Optional[str] = "n") -> Dict[str, Any]:
        """
        Wu-Palmer taxonomic path similarity WSD:
        Measures conceptual similarity between target word synsets and context nouns in the WordNet hierarchy.
        """
        synsets = wn.synsets(target_word.lower(), pos=pos)
        if not synsets:
            return {"predicted_synset": None, "confidence": 0.0, "details": [], "algorithm": "Wu-Palmer Path Similarity"}

        tokens = self.preprocess_text(sentence)
        context_nouns = []
        for t in tokens:
            if t != target_word.lower():
                n_syns = wn.synsets(t, pos=wn.NOUN)
                if n_syns:
                    context_nouns.append(n_syns[0])

        if not context_nouns:
            return self.extended_lesk(sentence, target_word, pos=pos)

        candidates_info = []
        best_synset = None
        max_avg_sim = -1.0

        for syn in synsets:
            scores = []
            for ctx_syn in context_nouns:
                try:
                    sim = syn.wup_similarity(ctx_syn)
                    if sim is not None:
                        scores.append(sim)
                except Exception:
                    pass
            avg_sim = float(np.mean(scores)) if scores else 0.0
            candidates_info.append({
                "synset": syn.name(),
                "pos": syn.pos(),
                "definition": syn.definition(),
                "score": round(avg_sim, 4)
            })

            if avg_sim > max_avg_sim:
                max_avg_sim = avg_sim
                best_synset = syn

        if max_avg_sim <= 0:
            best_synset = synsets[0]
            confidence = 0.25
        else:
            total_score = sum(c["score"] for c in candidates_info)
            confidence = round(max_avg_sim / (total_score if total_score > 0 else 1), 3)

        candidates_info.sort(key=lambda x: x["score"], reverse=True)
        return {
            "predicted_synset": best_synset.name(),
            "best_synset_obj": best_synset,
            "definition": best_synset.definition(),
            "confidence": confidence,
            "score": round(max_avg_sim, 4),
            "candidates": candidates_info,
            "algorithm": "Wu-Palmer Path Similarity"
        }

    # -------------------------------------------------------------
    # 5. Word Sense Induction (WSI) Unsupervised Clustering
    # -------------------------------------------------------------
    def induce_senses(self, sentences: List[str], target_word: str, n_clusters: int = 2) -> Dict[str, Any]:
        """
        Unsupervised Word Sense Induction:
        Discovers senses directly from unannotated sentences using TF-IDF feature extraction
        and K-Means clustering.
        """
        if len(sentences) < n_clusters:
            n_clusters = max(1, len(sentences))

        cleaned_docs = []
        for s in sentences:
            tokens = self.preprocess_text(s)
            tokens = [t for t in tokens if t != target_word.lower()]
            cleaned_docs.append(" ".join(tokens))

        vectorizer = TfidfVectorizer(max_features=100, min_df=1)
        X = vectorizer.fit_transform(cleaned_docs)
        feature_names = np.array(vectorizer.get_feature_names_out())

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        clusters_info = []
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

        for i in range(n_clusters):
            top_words_idx = order_centroids[i, :7]
            top_words = [feature_names[ind] for ind in top_words_idx if ind < len(feature_names)]
            cluster_sentences = [sentences[idx] for idx, label in enumerate(labels) if label == i]

            clusters_info.append({
                "cluster_id": i + 1,
                "size": len(cluster_sentences),
                "top_keywords": top_words,
                "sample_sentences": cluster_sentences[:3]
            })

        return {
            "target_word": target_word,
            "n_clusters": n_clusters,
            "labels": labels.tolist(),
            "clusters": clusters_info
        }
