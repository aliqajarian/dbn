# Amazon Book Reviews Anomaly Detection System

A comprehensive system for detecting anomalies and analyzing patterns in Amazon book reviews using advanced NLP and visualization techniques.

## Features

### 1. Advanced Review Analysis
- **Multi-model Sentiment Analysis**
  - TextBlob for general sentiment
  - VADER for social media sentiment
  - DistilBERT transformer for deep learning-based sentiment
- **Speech Pattern Detection**
  - Emotional intensity analysis
  - Sarcasm detection
  - Exaggeration patterns
  - Personal experience indicators
  - Recommendation patterns

### 2. Interactive Visualizations
- **Review Type Analysis**
  - Sunburst charts showing review types and sentiment distribution
  - Hierarchical view of review categories
- **Emotion Flow Analysis**
  - Sankey diagrams showing emotion flow in reviews
  - Connection between emotional states and sentiment
- **Review Network Analysis**
  - Network graphs showing relationships between review types
  - Interactive node-link visualization
- **Review Embedding Space**
  - t-SNE visualization of review features
  - Clustering of similar reviews
- **Word Cloud Analysis**
  - Word clouds for different review types
  - Visual representation of common terms
- **Timeline Analysis**
  - Interactive timeline of review patterns
  - Sentiment trends over time

### 3. Anomaly Detection
- **Multi-factor Analysis**
  - Sentiment extremity detection
  - Sarcasm and exaggeration detection
  - Emotional intensity thresholds
  - Pattern-based anomaly scoring

## Quick Start

### Using Google Colab
1. Open the notebook in Google Colab:
   ```python
   !git clone https://github.com/yourusername/amazon-review-analyzer.git
   %cd amazon-review-analyzer
   ```

2. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```

3. Run the analysis:
   ```python
   from src.models.review_analyzer import ReviewAnalyzer
   import pandas as pd

   # Initialize analyzer
   analyzer = ReviewAnalyzer()

   # Load your data
   reviews_df = pd.read_csv('your_reviews.csv')

   # Analyze reviews
   results_df = analyzer.analyze_reviews(reviews_df)

   # Generate visualizations
   visualizations = analyzer.plot_results(results_df, save_path='visualizations')
   ```

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/amazon-review-analyzer.git
   cd amazon-review-analyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis:
   ```python
   from src.models.review_analyzer import ReviewAnalyzer
   import pandas as pd

   analyzer = ReviewAnalyzer()
   results_df = analyzer.analyze_reviews(your_reviews_df)
   ```

## Project Structure
```
amazon-review-analyzer/
├── src/
│   ├── models/
│   │   └── review_analyzer.py
│   ├── utils/
│   │   └── review_visualizer.py
│   └── data/
│       └── word_lists/
├── notebooks/
│   └── analyze_reviews_colab.ipynb
├── requirements.txt
└── README.md
```

## Dependencies
- torch>=2.0.0
- transformers>=4.30.0
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.2.0
- nltk>=3.8.1
- spacy>=3.5.0
- textblob>=0.17.1
- plotly>=5.14.0
- networkx>=3.1
- wordcloud>=1.9.0
- matplotlib>=3.7.0

## Visualization Examples

### Review Type Analysis
The sunburst chart shows the distribution of different review types and their sentiment:
- Emotional reviews (positive/negative)
- Sarcastic reviews
- Personal experience reviews
- Recommendation reviews

### Emotion Flow
The Sankey diagram visualizes how emotions flow through reviews:
- Connection between emotional states
- Sentiment distribution
- Intensity levels

### Review Network
The network graph shows relationships between different review types:
- Node size indicates frequency
- Edge thickness shows co-occurrence
- Color indicates review category

### Review Embedding
The t-SNE plot shows the distribution of reviews in feature space:
- Clusters of similar reviews
- Outliers and anomalies
- Review type distribution

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
