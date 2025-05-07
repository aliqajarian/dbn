# Anomaly Detection System for Amazon Book Reviews

A deep learning-based anomaly detection system that uses Deep Belief Networks (DBN) and GANs to detect anomalies in Amazon book reviews.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Model Architecture](#model-architecture)
- [Performance Optimization](#performance-optimization)
- [Free Platform Deployment](#free-platform-deployment)
- [API Documentation](#api-documentation)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Advanced Text Processing**
  - NLP-based text preprocessing
  - TF-IDF feature extraction
  - Entity recognition

- **Deep Learning Models**
  - Deep Belief Network (DBN)
  - GAN for synthetic data generation
  - Time series analysis

- **Real-time Detection**
  - Streaming data processing
  - Batch and real-time inference
  - Configurable thresholds

- **Deployment**
  - Docker containerization
  - REST API with FastAPI
  - ONNX model export

## Installation

### Prerequisites
- Python 3.9+
- Docker (for deployment)
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/aliqajarian/dbn.git
cd dbn
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Performance Optimization

### Training Time Estimation

1. **Data Processing**:
- Downloading dataset: ~5-10 minutes
- Data preprocessing: ~15-20 minutes
- Feature extraction: ~10-15 minutes

2. **Model Training**:
- DBN training (50 epochs): ~2-3 hours on CPU
- GAN training (50 epochs): ~1-2 hours on CPU
- Total training time: ~3-5 hours on CPU

3. **Testing and Analysis**:
- Model evaluation: ~15-20 minutes
- Results analysis: ~10-15 minutes

**Total estimated time**: 4-6 hours on CPU

### Optimization Techniques

1. **Data Size Reduction**:
```python
# Use a subset of data for faster training
df = df.sample(n=10000, random_state=42)
```

2. **Model Size Reduction**:
```python
# Use smaller architecture
layer_sizes = [500, 250, 100, 50]  # Smaller than original
```

3. **Epoch Reduction**:
```python
# Train for fewer epochs
epochs = 20  # Instead of 50
```

4. **Mixed Precision Training**:
```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

## Free Platform Deployment

### Google Colab (Recommended)

1. Create a new Colab notebook
2. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Run the following commands:
```python
# Install requirements
!pip install torch pandas numpy scikit-learn nltk spacy matplotlib seaborn tqdm plotly
!python -m spacy download en_core_web_sm

# Clone repository
!git clone https://github.com/aliqajarian/dbn.git
%cd dbn

# Enable GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Run training with checkpointing
!python src/train_amazon_optimized.py
```

Benefits:
- Free GPU access (Tesla T4 or P100)
- 12-15 hours runtime
- 15GB RAM
- Pre-installed Python packages
- Automatic checkpointing and model saving to Google Drive
- Ability to resume training from last checkpoint

### Checkpointing and Model Saving

The system now includes automatic checkpointing and model saving features:

1. **Checkpoint Location**:
   - Models are saved in Google Drive: `/content/drive/MyDrive/dbn_checkpoints/`
   - Checkpoints include:
     - Model state
     - Optimizer state
     - Training metrics
     - Current epoch

2. **Resuming Training**:
   - Training automatically resumes from the last checkpoint
   - No data loss on session disconnection
   - Progress is preserved between Colab sessions

3. **Output Storage**:
   - Training logs
   - Performance metrics
   - Visualization plots
   - Model artifacts

### Kaggle Notebooks

1. Create a new Kaggle notebook
2. Run the following commands:

```python
# Install packages
!pip install torch pandas numpy scikit-learn nltk spacy matplotlib seaborn tqdm plotly

# Download spaCy model
!python -m spacy download en_core_web_sm

# Run training
!python src/train_amazon.py
```

Benefits:
- Free GPU access (Tesla P100)
- 9 hours runtime
- 16GB RAM
- Built-in dataset support

### Hugging Face Spaces

1. Create a new Space
2. Add to requirements.txt:
```text
torch
pandas
numpy
scikit-learn
nltk
spacy
matplotlib
seaborn
tqdm
plotly
```

Benefits:
- Free GPU access
- 16GB RAM
- Easy deployment
- Good for sharing results

## Quick Start

1. Download and prepare data:
```bash
python src/data/download_data.py
```

2. Train the model:
```bash
python src/train_amazon_optimized.py
```

3. Run real-time detection:
```bash
python src/run_detector.py
```

4. Deploy the model:
```bash
python src/deploy.py
```

## Detailed Usage

### Data Preparation

1. Download Amazon review data:
```python
from src.data.amazon_loader import AmazonBooksLoader

loader = AmazonBooksLoader()
df = loader.load_data()
```

2. Run exploratory analysis:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Model Training

1. Configure model parameters in `config/model_config.json`:
```json
{
    "model": {
        "layer_sizes": [1000, 500, 200, 100],
        "sequence_length": 10,
        "learning_rate": 0.001
    }
}
```

2. Train the model:
```bash
python src/train_amazon_optimized.py
```

### Real-time Detection

```python
from src.realtime_detector import RealTimeAnomalyDetector

detector = RealTimeAnomalyDetector(
    model_path="models/anomaly_detector.pth",
    threshold=0.1
)

detector.start()
detector.add_data({
    'text': 'review text',
    'timestamp': '2024-01-01T00:00:00',
    'rating': 4.5
})
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size in config
   - Use smaller model architecture
   - Enable mixed precision training

2. **Training Issues**
   - Check learning rate
   - Verify data preprocessing
   - Monitor loss curves
   - Use early stopping

3. **Deployment Issues**
   - Check Docker logs
   - Verify port availability
   - Check model file paths

### Performance Tips

1. **For CPU Training**:
   - Reduce batch size
   - Use smaller model architecture
   - Enable multiprocessing

2. **For GPU Training**:
   - Enable mixed precision
   - Use larger batch sizes
   - Optimize memory usage

3. **For Free Platforms**:
   - Use optimized code
   - Monitor resource usage
   - Save checkpoints regularly

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Amazon Review Dataset
- PyTorch
- FastAPI
- Docker
- Google Colab
- Kaggle
- Hugging Face

## Contact

For questions and support, please open an issue in the GitHub repository.

---

## Additional Resources

- [Model Documentation](docs/model.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)
