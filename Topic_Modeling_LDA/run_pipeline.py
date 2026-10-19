import os
import joblib
import pandas as pd
from src.preprocessor import TextPreprocessor
from src.lda_engine import LDAEngine
from src.report_generator import generate_topic_report

def main():
    print("=" * 60)
    print("Running Headless Pipeline for Latent Dirichlet Allocation (LDA)")
    print("=" * 60)

    # 1. Load Data
    data_path = "data/news_corpus.csv"
    if not os.path.exists(data_path):
        import generate_corpus
    
    df = pd.read_csv(data_path)
    raw_texts = df["text"].tolist()
    categories = df["category"].tolist() if "category" in df.columns else None
    print(f"[1/5] Loaded raw corpus: {len(raw_texts)} documents.")

    # 2. Preprocess
    preprocessor = TextPreprocessor()
    clean_texts, _ = preprocessor.preprocess_corpus(raw_texts)
    print(f"[2/5] Text preprocessing completed (WordNet lemmatization & stopword removal).")

    # 3. Build Document-Term Matrix
    engine = LDAEngine()
    dtm, vectorizer, corpus_stats = engine.build_dtm(clean_texts, min_df=2, max_df=0.85, max_features=1000)
    print(f"[3/5] Built Document-Term Matrix: {corpus_stats['num_docs']} docs x {corpus_stats['vocab_size']} terms (Sparsity: {corpus_stats['sparsity_pct']}%).")

    # 4. Train LDA Model
    model_results = engine.train(n_components=5, learning_method='online', max_iter=25, random_state=42)
    print(f"[4/5] Trained LDA Model (K=5). Perplexity: {model_results['perplexity']}, Coherence: {model_results['mean_coherence']}.")

    # 5. Extract Deliverables
    topics_data = engine.get_topic_keywords(top_n=15)
    df_docs = engine.get_document_assignments(raw_texts=raw_texts, categories=categories)

    # Save Preprocessed Corpus
    df_preprocessed = pd.DataFrame({
        "doc_id": range(len(raw_texts)),
        "raw_text": raw_texts,
        "cleaned_text": clean_texts,
        "dominant_topic": df_docs["dominant_topic"],
        "topic_weight": df_docs["topic_weight"]
    })
    os.makedirs("data", exist_ok=True)
    df_preprocessed.to_csv("data/preprocessed_corpus.csv", index=False)
    print("  -> Saved data/preprocessed_corpus.csv")

    # Save Topic Word Distributions
    topic_words_list = []
    for t in topics_data:
        for w, weight in zip(t["words"], t["weights"]):
            topic_words_list.append({
                "topic_id": t["topic_id"],
                "topic_label": t["label"],
                "word": w,
                "probability": round(weight, 5)
            })
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(topic_words_list).to_csv("outputs/topic_word_distributions.csv", index=False)
    print("  -> Saved outputs/topic_word_distributions.csv")

    # Save Serialized Models
    os.makedirs("models", exist_ok=True)
    joblib.dump(engine.lda_model, "models/lda_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("  -> Saved models/lda_model.pkl and models/vectorizer.pkl")

    # Generate and Save Topic Interpretation Report
    report_md = generate_topic_report(corpus_stats, model_results, topics_data, df_docs)
    with open("Topic_Interpretation_Report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("  -> Saved Topic_Interpretation_Report.md")

    print("\n[SUCCESS] All pipeline artifacts generated successfully!")

if __name__ == "__main__":
    main()
