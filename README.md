# Amazon Book Reviews Anomaly Detection

A deep learning-based anomaly detection system that uses sentiment analysis and content filtering to detect anomalies in Amazon book reviews.

## Features

- **Content Analysis**
  - Inappropriate language detection
  - Spam detection
  - Quality indicators
  - Severity scoring

- **Sentiment Analysis**
  - TextBlob sentiment analysis
  - Transformer-based sentiment analysis
  - Combined sentiment scoring

- **Visualization**
  - Sentiment distribution plots
  - Anomaly distribution charts
  - Interactive Plotly visualizations

## Quick Start with Google Colab

1. Open the [analyze_reviews_colab.ipynb](notebooks/analyze_reviews_colab.ipynb) notebook in Google Colab

2. Run the installation cell to set up the environment:
```python
# Install required packages
!pip install torch pandas numpy scikit-learn nltk spacy textblob transformers plotly
!python -m spacy download en_core_web_sm

# Clone the repository
!git clone https://github.com/aliqajarian/dbn.git
%cd dbn
```

3. Load your data:
```python
import pandas as pd
df = pd.read_csv('your_reviews.csv')
```

4. Initialize and run the analyzer:
```python
from src.models.review_analyzer import ReviewAnalyzer

# Initialize analyzer
analyzer = ReviewAnalyzer()

# Analyze reviews
results_df = analyzer.analyze_reviews(df)

# Generate report
report = analyzer.generate_report(results_df)
print(report)

# Visualize results
fig1, fig2 = analyzer.plot_results(results_df)
fig1.show()
fig2.show()
```

## Example Results

The analyzer provides several types of analysis:

1. **Content Analysis**
   - Inappropriate language detection
   - Spam detection
   - Quality indicators
   - Severity scores

2. **Sentiment Analysis**
   - TextBlob sentiment scores
   - Transformer-based sentiment scores
   - Combined sentiment assessment

3. **Anomaly Detection**
   - Identifies reviews with:
     - Inappropriate content
     - Negative sentiment
     - High severity scores
     - Spam indicators

4. **Visualizations**
   - Sentiment distribution
   - Anomaly distribution
   - Interactive plots

## Project Structure

```
dbn/
├── notebooks/
│   └── analyze_reviews_colab.ipynb
├── src/
│   ├── data/
│   │   ├── word_lists/
│   │   │   ├── inappropriate_words.txt
│   │   │   ├── quality_indicators.txt
│   │   │   └── word_list_manager.py
│   │   └── preprocessor.py
│   └── models/
│       └── review_analyzer.py
└── README.md
```

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- NLTK
- spaCy
- TextBlob
- Plotly
- Pandas
- NumPy

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
