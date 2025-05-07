# Anomaly Detection System for Amazon Book Reviews

A deep learning-based anomaly detection system that uses Deep Belief Networks (DBN) and GANs to detect anomalies in Amazon book reviews.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Model Architecture](#model-architecture)
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
git clone https://github.com/aliqajarian/anomaly-detection.git
cd anomaly-detection
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

## Project Structure

```
anomaly_detection/
├── data/
│   ├── raw/                    # Raw data storage
│   └── processed/              # Processed data storage
├── src/
│   ├── data/
│   │   ├── preprocessor.py     # Text preprocessing
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── dbn.py             # DBN implementation
│   │   ├── gan.py             # GAN implementation
│   │   └── autoencoder.py
│   ├── utils/
│   │   ├── visualization.py
│   │   └── metrics.py
│   └── train.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_evaluation.ipynb
├── tests/
├── config/
└── requirements.txt
```

## Quick Start

1. Download and prepare data:
```bash
python src/data/download_data.py
```

2. Train the model:
```bash
python src/train_model.py
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
from src.data.download_data import download_amazon_reviews

download_amazon_reviews(category="Books")
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
python src/train_model.py
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

### Deployment

1. Build Docker image:
```bash
docker build -t anomaly-detector:latest .
```

2. Run container:
```bash
docker run -p 8000:8000 anomaly-detector:latest
```

## Model Architecture

### Deep Belief Network (DBN)
- Multiple layers of Restricted Boltzmann Machines (RBMs)
- Unsupervised pre-training
- Fine-tuning with backpropagation

### GAN Architecture
- Generator: Transforms noise into synthetic data
- Discriminator: Distinguishes real from synthetic data
- Adversarial training for improved performance

## API Documentation

### Endpoints

#### POST /predict
Detect anomalies in review data.

**Request Body:**
```json
{
    "text": "review text",
    "timestamp": "2024-01-01T00:00:00",
    "rating": 4.5
}
```

**Response:**
```json
{
    "is_anomaly": true,
    "confidence": 0.95,
    "error_score": 0.85
}
```

#### GET /health
Check API health status.

#### GET /metrics
Get model performance metrics.

## Monitoring

### Logging
- Application logs in `logs/`
- Docker logs: `docker logs anomaly-detector`

### Metrics
- Reconstruction error
- Anomaly detection rate
- API response time

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size in config
   - Use smaller model architecture

2. **Training Issues**
   - Check learning rate
   - Verify data preprocessing
   - Monitor loss curves

3. **Deployment Issues**
   - Check Docker logs
   - Verify port availability
   - Check model file paths

### Debugging

1. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Check model state:
```python
model.eval()
print(model.state_dict())
```

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

## Contact

For questions and support, please open an issue in the GitHub repository.

---

## Additional Resources

- [Model Documentation](docs/model.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)
