import gzip
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonBooksLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz"
        
    def download_data(self):
        """Download the Amazon Books dataset."""
        logger.info("Downloading Amazon Books dataset...")
        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        file_path = self.data_dir / "meta_Books.jsonl.gz"
        with open(file_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        logger.info(f"Dataset downloaded to {file_path}")
        return file_path
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load and process the Amazon Books dataset."""
        if file_path is None:
            file_path = self.data_dir / "meta_Books.jsonl.gz"
        
        if not file_path.exists():
            file_path = self.download_data()
        
        logger.info("Loading and processing data...")
        data = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing records"):
                try:
                    record = json.loads(line)
                    # Extract relevant fields
                    processed_record = {
                        'asin': record.get('asin', ''),
                        'title': record.get('title', ''),
                        'description': record.get('description', ''),
                        'price': record.get('price', 0.0),
                        'brand': record.get('brand', ''),
                        'categories': record.get('categories', []),
                        'rating': record.get('rating', 0.0),
                        'review_count': record.get('review_count', 0),
                        'main_category': record.get('main_category', ''),
                        'sub_categories': record.get('sub_categories', [])
                    }
                    data.append(processed_record)
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for anomaly detection."""
        logger.info("Preprocessing data...")
        
        # Clean text fields
        df['description'] = df['description'].fillna('')
        df['title'] = df['title'].fillna('')
        
        # Combine text fields for analysis
        df['text'] = df['title'] + ' ' + df['description']
        
        # Handle missing values
        df['price'] = df['price'].fillna(df['price'].median())
        df['rating'] = df['rating'].fillna(df['rating'].median())
        df['review_count'] = df['review_count'].fillna(0)
        
        # Create feature for category depth
        df['category_depth'] = df['categories'].apply(len)
        
        # Create feature for subcategory count
        df['subcategory_count'] = df['sub_categories'].apply(len)
        
        return df 