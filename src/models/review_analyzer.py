import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from ..utils.review_visualizer import ReviewVisualizer

class ReviewAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model.to(self.device)
        self.sia = SentimentIntensityAnalyzer()
        self.visualizer = ReviewVisualizer()
        
        # Download required NLTK data
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_review(self, review_text: str) -> Dict:
        """Analyze a single review."""
        # Preprocess text
        cleaned_text = self.preprocess_text(review_text)
        
        # Analyze sentiment
        sentiment_analysis = self.analyze_sentiment(cleaned_text)
        
        # Analyze speech patterns
        speech_analysis = self.analyze_speech_patterns(cleaned_text)
        
        # Combine results
        analysis = {
            'review_text': review_text,
            'cleaned_text': cleaned_text,
            'sentiment_analysis': sentiment_analysis,
            'speech_analysis': speech_analysis,
            'is_anomaly': self.is_anomaly(sentiment_analysis, speech_analysis)
        }
        
        return analysis
    
    def analyze_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze multiple reviews from a DataFrame."""
        results = []
        total_reviews = len(reviews_df)
        
        for idx, row in reviews_df.iterrows():
            if idx % 100 == 0:
                self.logger.info(f"Processing review {idx+1}/{total_reviews}")
            
            analysis = self.analyze_review(row['review_text'])
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def generate_report(self, results_df: pd.DataFrame, save_path: Optional[str] = None) -> Dict:
        """Generate a comprehensive report of the analysis."""
        report = {
            'summary': {
                'total_reviews': len(results_df),
                'anomaly_count': results_df['is_anomaly'].sum(),
                'anomaly_percentage': (results_df['is_anomaly'].sum() / len(results_df)) * 100
            },
            'sentiment_distribution': {
                'positive': len(results_df[results_df['sentiment_analysis'].apply(lambda x: x['textblob_sentiment'] > 0)]),
                'negative': len(results_df[results_df['sentiment_analysis'].apply(lambda x: x['textblob_sentiment'] < 0)]),
                'neutral': len(results_df[results_df['sentiment_analysis'].apply(lambda x: x['textblob_sentiment'] == 0)])
            },
            'speech_patterns': {
                'emotional': results_df['speech_analysis'].apply(lambda x: x['is_emotional']).sum(),
                'sarcastic': results_df['speech_analysis'].apply(lambda x: x['is_sarcastic']).sum(),
                'exaggerated': results_df['speech_analysis'].apply(lambda x: x['is_exaggerated']).sum(),
                'personal': results_df['speech_analysis'].apply(lambda x: x['is_personal']).sum(),
                'recommendation': results_df['speech_analysis'].apply(lambda x: x['is_recommendation']).sum()
            }
        }
        
        # Generate visualizations
        visualizations = self.visualizer.create_all_visualizations(results_df, save_path)
        report['visualizations'] = visualizations
        
        return report
    
    def plot_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """Create visualizations of the analysis results."""
        return self.visualizer.create_all_visualizations(results_df, save_path)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess the review text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using multiple methods."""
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        
        # VADER sentiment
        vader_scores = self.sia.polarity_scores(text)
        
        # Transformer sentiment
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            transformer_scores = torch.softmax(outputs.logits, dim=1)
            transformer_sentiment = transformer_scores[0][1].item()  # Positive class probability
        
        return {
            'textblob_sentiment': textblob_sentiment,
            'vader_scores': vader_scores,
            'transformer_sentiment': transformer_sentiment
        }
    
    def analyze_speech_patterns(self, text: str) -> Dict:
        """Analyze speech patterns in the text."""
        # Emotional intensity
        emotional_intensity = abs(TextBlob(text).sentiment.polarity)
        
        # Sarcasm detection (simple heuristic)
        is_sarcastic = any(word in text.lower() for word in ['not', 'but', 'however', 'though', 'although'])
        
        # Exaggeration detection
        exaggeration_patterns = [
            r'\b(very|extremely|absolutely|incredibly|unbelievably)\s+\w+',
            r'\b(best|worst|amazing|terrible|horrible|fantastic)\b',
            r'!{2,}',
            r'\b(never|always|every|all)\b'
        ]
        is_exaggerated = any(re.search(pattern, text, re.IGNORECASE) for pattern in exaggeration_patterns)
        
        # Personal experience detection
        personal_patterns = [
            r'\b(i|me|my|mine|we|our|us)\b',
            r'\b(think|feel|believe|experience|tried|used)\b'
        ]
        is_personal = any(re.search(pattern, text, re.IGNORECASE) for pattern in personal_patterns)
        
        # Recommendation detection
        recommendation_patterns = [
            r'\b(recommend|suggest|advise|should|must|worth)\b',
            r'\b(good|great|excellent|worth|value)\b'
        ]
        is_recommendation = any(re.search(pattern, text, re.IGNORECASE) for pattern in recommendation_patterns)
        
        return {
            'emotional_intensity': emotional_intensity,
            'is_sarcastic': is_sarcastic,
            'is_exaggerated': is_exaggerated,
            'is_personal': is_personal,
            'is_recommendation': is_recommendation
        }
    
    def is_anomaly(self, sentiment_analysis: Dict, speech_analysis: Dict) -> bool:
        """Determine if a review is anomalous based on content and sentiment analysis."""
        # Check for extreme sentiment
        extreme_sentiment = abs(sentiment_analysis['textblob_sentiment']) > 0.8
        
        # Check for sarcasm
        sarcasm_detected = speech_analysis['is_sarcastic']
        
        # Check for exaggeration
        exaggeration_detected = speech_analysis['is_exaggerated']
        
        # Check for emotional intensity
        high_emotional = speech_analysis['emotional_intensity'] > 0.7
        
        # Review is anomalous if it has extreme sentiment or shows sarcasm/exaggeration
        return extreme_sentiment or (sarcasm_detected and exaggeration_detected) or high_emotional 