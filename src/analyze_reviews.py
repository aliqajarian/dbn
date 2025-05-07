import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from train_amazon_optimized import AmazonBooksLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import plotly.graph_objects as go
import plotly.express as px

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pre-trained sentiment analysis model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model.to(self.device)
        
        # Enhanced offensive categories
        self.offensive_categories = {
            'rude_language': [
                'stupid', 'idiot', 'dumb', 'fool', 'incompetent', 'useless',
                'worthless', 'garbage', 'trash', 'waste', 'horrible', 'terrible',
                'moron', 'imbecile', 'clueless', 'brainless', 'ignorant'
            ],
            'aggressive_tone': [
                'angry', 'furious', 'outraged', 'enraged', 'frustrated',
                'annoyed', 'irritated', 'exasperated', 'disgusted', 'appalled',
                'infuriated', 'livid', 'fuming', 'seething', 'raging'
            ],
            'disrespectful': [
                'rude', 'impolite', 'disrespectful', 'unprofessional',
                'incompetent', 'unhelpful', 'uncooperative', 'dismissive',
                'arrogant', 'condescending', 'patronizing', 'insulting'
            ],
            'threatening': [
                'report', 'complain', 'sue', 'legal', 'lawyer', 'court',
                'action', 'proceedings', 'case', 'warn', 'caution',
                'lawsuit', 'litigation', 'prosecute', 'penalize'
            ],
            'emotional_manipulation': [
                'disappointed', 'heartbroken', 'devastated', 'betrayed',
                'deceived', 'misled', 'cheated', 'scammed', 'robbed'
            ],
            'quality_complaints': [
                'defective', 'broken', 'damaged', 'faulty', 'poor quality',
                'substandard', 'inferior', 'cheap', 'flimsy', 'shoddy'
            ],
            'service_complaints': [
                'unresponsive', 'ignored', 'neglected', 'abandoned',
                'ghosted', 'unanswered', 'unaddressed', 'dismissive'
            ],
            'time_waste': [
                'waste of time', 'endless', 'eternal', 'forever',
                'never-ending', 'interminable', 'protracted', 'drawn-out'
            ]
        }
        
        # Enhanced severity patterns
        self.severity_patterns = {
            'critical': [
                r'\b(scam|fraud|rip-off|useless|worthless|garbage|trash)\b',
                r'\b(never|don\'t)\b.*\b(buy|purchase|recommend|use|trust)\b.*\b(again|ever)\b',
                r'\b(demand|expect|require)\b.*\b(refund|compensation|replacement)\b',
                r'\b(legal|lawyer|court)\b.*\b(action|proceedings|case)\b',
                r'\b(complete|total|absolute)\b.*\b(waste|disaster|failure)\b'
            ],
            'high': [
                r'\b(horrible|terrible|awful)\b.*\b(service|quality|product|experience)\b',
                r'\b(waste|wasting)\b.*\b(time|money|effort)\b',
                r'\b(worst|terrible|horrible)\b.*\b(ever|in history|of all time)\b',
                r'\b(angry|furious|outraged)\b.*\b(about|with|at)\b',
                r'\b(disgusted|appalled|shocked)\b.*\b(by|with|at)\b'
            ],
            'medium': [
                r'\b(poor|disappointing|mediocre)\b.*\b(service|quality|product)\b',
                r'\b(not|don\'t)\b.*\b(recommend|suggest|advise)\b',
                r'\b(could|should)\b.*\b(better|improve|enhance)\b',
                r'\b(frustrated|annoyed|irritated)\b.*\b(with|by|at)\b',
                r'\b(disappointed|let down|failed)\b.*\b(by|with|in)\b'
            ],
            'low': [
                r'\b(okay|fine|average)\b.*\b(but|could be better)\b',
                r'\b(not bad|not great)\b',
                r'\b(acceptable|tolerable)\b',
                r'\b(decent|reasonable)\b.*\b(but|could be better)\b'
            ]
        }
        
        # New analysis categories
        self.analysis_categories = {
            'emotional_intensity': {
                'high': ['furious', 'outraged', 'enraged', 'livid'],
                'medium': ['angry', 'frustrated', 'annoyed'],
                'low': ['disappointed', 'dissatisfied']
            },
            'complaint_type': {
                'product': ['defective', 'broken', 'damaged', 'faulty'],
                'service': ['rude', 'unhelpful', 'unresponsive'],
                'delivery': ['late', 'delayed', 'missing'],
                'price': ['overpriced', 'expensive', 'costly']
            },
            'tone': {
                'aggressive': ['demand', 'require', 'must', 'should'],
                'passive': ['could', 'might', 'maybe', 'perhaps'],
                'neutral': ['think', 'believe', 'feel', 'seem']
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess review text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def detect_offensive_language(self, text: str) -> dict:
        """Detect offensive language in the review."""
        words = word_tokenize(text.lower())
        
        # Count offensive words by category
        category_counts = {category: 0 for category in self.offensive_categories.keys()}
        for word in words:
            for category, words_list in self.offensive_categories.items():
                if word in words_list:
                    category_counts[category] += 1
        
        # Check for severity patterns
        severity_matches = {level: [] for level in self.severity_patterns.keys()}
        for level, patterns in self.severity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    severity_matches[level].append(pattern)
        
        # Calculate severity score
        severity_score = 0
        for level, matches in severity_matches.items():
            if level == 'critical':
                severity_score += len(matches) * 4
            elif level == 'high':
                severity_score += len(matches) * 3
            elif level == 'medium':
                severity_score += len(matches) * 2
            else:
                severity_score += len(matches)
        
        # Determine overall severity
        if severity_score >= 8:
            severity = 'critical'
        elif severity_score >= 5:
            severity = 'high'
        elif severity_score >= 3:
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'category_counts': category_counts,
            'severity_matches': severity_matches,
            'severity_score': severity_score,
            'severity_level': severity
        }
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of the review."""
        # Get TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        
        # Get transformer model sentiment
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            transformer_sentiment = torch.softmax(outputs.logits, dim=1)[0][1].item()
        
        return {
            'textblob_sentiment': textblob_sentiment,
            'transformer_sentiment': transformer_sentiment,
            'is_negative': textblob_sentiment < -0.3 or transformer_sentiment < 0.3
        }
    
    def analyze_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze all reviews in the dataset."""
        results = []
        
        for _, row in df.iterrows():
            review_text = str(row['description'])
            processed_text = self.preprocess_text(review_text)
            
            # Get offensive language analysis
            offensive_analysis = self.detect_offensive_language(processed_text)
            
            # Get sentiment analysis
            sentiment_analysis = self.analyze_sentiment(processed_text)
            
            # Combine results
            results.append({
                'review_text': review_text,
                'category_counts': offensive_analysis['category_counts'],
                'severity_matches': offensive_analysis['severity_matches'],
                'severity_score': offensive_analysis['severity_score'],
                'severity_level': offensive_analysis['severity_level'],
                'textblob_sentiment': sentiment_analysis['textblob_sentiment'],
                'transformer_sentiment': sentiment_analysis['transformer_sentiment'],
                'is_negative': sentiment_analysis['is_negative'],
                'is_anomaly': (offensive_analysis['severity_score'] >= 5 or 
                             sentiment_analysis['is_negative'])
            })
        
        return pd.DataFrame(results)
    
    def generate_enhanced_report(self, results_df: pd.DataFrame) -> str:
        """Generate a more comprehensive and structured report."""
        total_reviews = len(results_df)
        anomalous_reviews = results_df['is_anomaly'].sum()
        
        # Calculate statistics
        category_stats = pd.DataFrame([row['category_counts'] for row in results_df['category_counts']])
        severity_dist = results_df['severity_level'].value_counts()
        
        # Generate HTML report
        report = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #2c3e50; }",
            ".section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }",
            ".metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }",
            ".chart { margin: 20px 0; }",
            ".highlight { color: #e74c3c; font-weight: bold; }",
            ".positive { color: #27ae60; }",
            ".negative { color: #c0392b; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Comprehensive Review Analysis Report</h1>",
            
            "<div class='section'>",
            "<h2>1. Executive Summary</h2>",
            f"<p>Total Reviews Analyzed: <span class='metric'>{total_reviews:,}</span></p>",
            f"<p>Anomalous Reviews: <span class='metric highlight'>{anomalous_reviews:,} ({anomalous_reviews/total_reviews*100:.1f}%)</span></p>",
            f"<p>Average Severity Score: <span class='metric'>{results_df['severity_score'].mean():.2f}</span></p>",
            "</div>",
            
            "<div class='section'>",
            "<h2>2. Severity Analysis</h2>",
            "<div class='chart'>",
            "<h3>Severity Distribution</h3>",
        ]
        
        # Add severity distribution
        for level, count in severity_dist.items():
            percentage = (count / total_reviews) * 100
            report.append(f"<p>{level.capitalize()}: <span class='metric'>{count:,} ({percentage:.1f}%)</span></p>")
        
        report.extend([
            "</div>",
            "</div>",
            
            "<div class='section'>",
            "<h2>3. Category Analysis</h2>",
            "<div class='chart'>",
            "<h3>Offensive Language Categories</h3>",
        ])
        
        # Add category statistics
        for category, count in category_stats.sum().items():
            percentage = (count / total_reviews) * 100
            report.append(f"<p>{category.replace('_', ' ').title()}: <span class='metric'>{count:,} ({percentage:.1f}%)</span></p>")
        
        report.extend([
            "</div>",
            "</div>",
            
            "<div class='section'>",
            "<h2>4. Detailed Findings</h2>",
            "<h3>Top Issues</h3>",
        ])
        
        # Add top issues
        top_issues = category_stats.sum().nlargest(5)
        for issue, count in top_issues.items():
            report.append(f"<p>{issue.replace('_', ' ').title()}: <span class='metric'>{count:,} occurrences</span></p>")
        
        report.extend([
            "<h3>Pattern Analysis</h3>",
        ])
        
        # Add pattern analysis
        pattern_stats = pd.DataFrame([row['severity_matches'] for row in results_df['severity_matches']])
        for level, patterns in pattern_stats.items():
            total_matches = sum(len(matches) for matches in patterns)
            report.append(f"<p>{level.capitalize()} Severity Patterns: <span class='metric'>{total_matches:,} matches</span></p>")
        
        report.extend([
            "</div>",
            
            "<div class='section'>",
            "<h2>5. Recommendations</h2>",
            "<h3>Immediate Actions</h3>",
            "<ul>",
            f"<li>Address {top_issues.index[0].replace('_', ' ').title()} issues</li>",
            f"<li>Monitor {severity_dist.index[0].replace('_', ' ').title()} severity cases</li>",
            "<li>Implement stricter content moderation</li>",
            "</ul>",
            
            "<h3>Long-term Improvements</h3>",
            "<ul>",
            "<li>Enhance customer service response system</li>",
            "<li>Review and update content policies</li>",
            "<li>Implement automated content filtering</li>",
            "</ul>",
            "</div>",
            
            "<div class='section'>",
            "<h2>6. Visualizations</h2>",
            "<p>Detailed visualizations are available in the following files:</p>",
            "<ul>",
            "<li>severity_distribution.png</li>",
            "<li>category_analysis.png</li>",
            "<li>pattern_analysis.png</li>",
            "<li>sentiment_analysis.png</li>",
            "</ul>",
            "</div>",
            
            "</body>",
            "</html>"
        ])
        
        return '\n'.join(report)

    def save_visualizations(self, results_df: pd.DataFrame):
        """Generate and save enhanced visualizations."""
        output_dir = Path("results/review_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Severity Distribution (Interactive Pie Chart)
        severity_dist = results_df['severity_level'].value_counts()
        fig = px.pie(values=severity_dist.values, names=severity_dist.index, 
                    title='Severity Distribution')
        fig.write_html(output_dir / 'severity_distribution.html')
        
        # 2. Category Analysis (Interactive Bar Chart)
        category_stats = pd.DataFrame([row['category_counts'] for row in results_df['category_counts']])
        fig = px.bar(category_stats.sum(), title='Offensive Language Categories')
        fig.write_html(output_dir / 'category_analysis.html')
        
        # 3. Pattern Analysis (Heatmap)
        pattern_stats = pd.DataFrame([row['severity_matches'] for row in results_df['severity_matches']])
        plt.figure(figsize=(12, 8))
        sns.heatmap(pattern_stats.corr(), annot=True, cmap='coolwarm')
        plt.title('Pattern Correlation Analysis')
        plt.savefig(output_dir / 'pattern_analysis.png')
        plt.close()
        
        # 4. Sentiment Analysis (Scatter Plot)
        fig = px.scatter(results_df, x='textblob_sentiment', y='transformer_sentiment',
                        color='severity_level', title='Sentiment Analysis')
        fig.write_html(output_dir / 'sentiment_analysis.html')

def main():
    # Load data
    loader = AmazonBooksLoader()
    df = loader.load_data(max_records=10000)
    
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Initialize analyzer
    analyzer = ReviewAnalyzer()
    
    # Analyze reviews
    results_df = analyzer.analyze_reviews(df)
    
    # Generate enhanced report
    report = analyzer.generate_enhanced_report(results_df)
    with open("results/review_analysis/enhanced_report.html", "w") as f:
        f.write(report)
    
    # Save visualizations
    analyzer.save_visualizations(results_df)
    
    logger.info("Enhanced analysis complete. Results saved in results/review_analysis/")

if __name__ == "__main__":
    main() 