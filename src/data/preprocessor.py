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
from typing import List, Dict, Tuple, Optional
import spacy
from .word_lists.word_list_manager import WordListManager
from pathlib import Path
import logging

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.word_list_manager = WordListManager()
        
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
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
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

    def check_content(self, text: str) -> Dict[str, Dict[str, List[Tuple[str, int]]]]:
        """
        Check text for inappropriate content, spam, and quality indicators.
        Returns a dictionary with categories and their matches.
        """
        # Get matches from word lists
        matches = self.word_list_manager.check_text(text)
        
        # Get severity summary
        severity_summary = self.word_list_manager.get_severity_summary(text)
        
        return {
            'matches': matches,
            'severity_summary': severity_summary
        }

    def is_appropriate(self, text: str, max_severity: int = 3) -> Tuple[bool, Dict[str, int]]:
        """
        Check if text is appropriate based on severity thresholds.
        Returns (is_appropriate, severity_summary)
        """
        severity_summary = self.word_list_manager.get_severity_summary(text)
        
        # Check if any category exceeds max severity
        is_appropriate = all(severity <= max_severity 
                           for severity in severity_summary.values())
        
        return is_appropriate, severity_summary

    def get_content_analysis(self, text: str) -> Dict[str, any]:
        """
        Get comprehensive content analysis including:
        - Inappropriate content detection
        - Spam detection
        - Quality indicators
        - Named entities
        - Keywords
        """
        # Basic text cleaning
        cleaned_text = self.clean_text(text)
        
        # Get content checks
        content_checks = self.check_content(text)
        
        # Get named entities
        entities = self.extract_entities(text)
        
        # Get keywords
        keywords = self.get_keywords([text], top_n=10)
        
        return {
            'content_checks': content_checks,
            'entities': entities,
            'keywords': keywords,
            'is_appropriate': self.is_appropriate(text)[0]
        }

    def load_speech_patterns(self) -> list:
        """Load speech patterns from file."""
        patterns = []
        pattern_file = Path(__file__).parent / 'word_lists' / 'speech_patterns.txt'
        
        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pattern, category, severity = line.split(',')
                        patterns.append((pattern.strip(), category.strip(), int(severity.strip())))
        except FileNotFoundError:
            logger.warning(f"Speech patterns file not found at {pattern_file}")
        
        return patterns 