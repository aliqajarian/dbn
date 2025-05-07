# Amazon Book Reviews Anomaly Detection

A deep learning-based system for detecting anomalous reviews in Amazon book reviews.

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── review_analyzer.py    # Main model implementation
│   │   ├── train_model.py        # Training script
│   │   └── test_model.py         # Testing script
│   ├── data/
│   │   ├── amazon_loader.py      # Data loading utilities
│   │   └── download_data.py      # Data download script
│   └── utils/
│       └── review_visualizer.py  # Visualization utilities
├── notebooks/
│   └── analyze_amazon_books.ipynb # Example notebook
├── requirements.txt              # Project dependencies
└── README.md                    # This file
```

## Quick Start Guide

### 1. Setup Environment

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download Amazon Books dataset
python -m src.data.download_data
```

This will download:
- Books metadata: `data/raw/meta_Books.jsonl.gz`
- Reviews data: `data/raw/reviews_Books.jsonl.gz`

### 3. Train Model

```bash
# Train the model
python -m src.models.train_model
```

Training will:
- Create checkpoints in `models/checkpoints/`
- Save training history
- Generate visualizations

### 4. Test Model

```bash
# Test the trained model
python -m src.models.test_model
```

Testing will:
- Load the latest model checkpoint
- Generate performance metrics
- Create visualizations in `models/results/`
- Show misclassified examples

### 5. Interactive Analysis

For interactive analysis, use the Jupyter notebook:
```bash
jupyter notebook notebooks/analyze_amazon_books.ipynb
```

## Key Features

1. **Data Processing**
   - Automatic data download
   - Efficient data loading
   - Text preprocessing

2. **Model Training**
   - Automatic checkpointing
   - Training/validation metrics
   - GPU support
   - Progress tracking

3. **Model Testing**
   - Performance metrics
   - Confusion matrix
   - Error analysis
   - Visualization tools

4. **Visualization**
   - Training progress
   - Test results
   - Error analysis
   - Review patterns

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- pandas
- numpy
- scikit-learn
- transformers
- nltk
- plotly
- seaborn
- matplotlib

## Troubleshooting

1. **Data Download Issues**
   - Check internet connection
   - Verify disk space
   - Ensure write permissions

2. **Training Issues**
   - Check GPU availability
   - Verify batch size fits in memory
   - Monitor disk space for checkpoints

3. **Testing Issues**
   - Ensure model checkpoint exists
   - Verify test data format
   - Check visualization dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
