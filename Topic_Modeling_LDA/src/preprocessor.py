import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK data is loaded
try:
    _ = stopwords.words('english')
except Exception:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    """
    Handles robust text preprocessing for Latent Dirichlet Allocation (LDA):
    - Lowercasing
    - Stripping URLs, emails, special symbols, numbers
    - Stopword filtering (standard + journalistic boilerplate)
    - WordNet Lemmatization
    - Short-token pruning
    """
    def __init__(self, custom_stopwords=None, use_lemmatizer=True, min_word_len=3):
        self.use_lemmatizer = use_lemmatizer
        self.min_word_len = min_word_len
        self.lemmatizer = WordNetLemmatizer() if use_lemmatizer else None
        
        # Base stopwords
        base_stops = set(stopwords.words('english'))
        
        # Domain-specific news boilerplate stopwords
        domain_stops = {
            'said', 'according', 'also', 'would', 'could', 'one', 'two',
            'new', 'official', 'reports', 'noted', 'coverage', 'breaking',
            'yesterday', 'today', 'year', 'years', 'time', 'first', 'last',
            'news', 'say', 'state', 'many', 'well', 'may', 'even', 'like',
            'analyst', 'industry', 'report'
        }
        self.stopwords = base_stops.union(domain_stops)
        if custom_stopwords:
            self.stopwords.update([w.strip().lower() for w in custom_stopwords if w.strip()])

    def clean_text(self, text: str) -> str:
        """Removes URLs, emails, punctuation, digits, extra whitespace."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', ' ', text)
        # Keep only alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def tokenize_and_lemmatize(self, text: str) -> list[str]:
        """Cleans, tokenizes, removes stopwords, and lemmatizes tokens."""
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        
        filtered_tokens = []
        for token in tokens:
            if len(token) >= self.min_word_len:
                if self.use_lemmatizer:
                    lemma = self.lemmatizer.lemmatize(token, pos='n')
                    if lemma == token:
                        lemma = self.lemmatizer.lemmatize(token, pos='v')
                else:
                    lemma = token
                if lemma not in self.stopwords and token not in self.stopwords and len(lemma) >= self.min_word_len:
                    filtered_tokens.append(lemma)
                    
        return filtered_tokens

    def preprocess_document(self, text: str) -> str:
        """Returns clean preprocessed text as a single string."""
        tokens = self.tokenize_and_lemmatize(text)
        return " ".join(tokens)

    def preprocess_corpus(self, texts: list[str]) -> tuple[list[str], list[list[str]]]:
        """Processes a list of raw documents. Returns (clean_texts, tokenized_docs)."""
        clean_texts = []
        tokenized_docs = []
        for doc in texts:
            tokens = self.tokenize_and_lemmatize(doc)
            clean_texts.append(" ".join(tokens))
            tokenized_docs.append(tokens)
        return clean_texts, tokenized_docs
