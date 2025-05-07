import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
from typing import List, Dict, Tuple
import spacy

class TextPreprocessor:
    def __init__(self, max_features: int = 1000, min_df: int = 2):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
        
        return ' '.join(tokens)
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]
    
    def get_keywords(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """Extract top keywords using TF-IDF."""
        cleaned_texts = [self.clean_text(text) for text in texts]
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
        
        # Get feature names and their scores
        feature_names = self.vectorizer.get_feature_names_out()
        scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        keyword_scores = list(zip(feature_names, scores))
        return sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    def preprocess(self, texts: List[str]) -> np.ndarray:
        """Preprocess texts and return feature matrix."""
        cleaned_texts = [self.clean_text(text) for text in texts]
        features = self.vectorizer.fit_transform(cleaned_texts)
        features_dense = features.toarray()
        return self.scaler.fit_transform(features_dense) 