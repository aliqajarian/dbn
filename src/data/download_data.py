import pandas as pd
import gzip
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_amazon_reviews(category: str = "Books", 
                          output_path: str = "data/raw/amazon_reviews.csv"):
    """
    Download Amazon review data for the specified category.
    You can download the dataset from: https://jmcauley.ucsd.edu/data/amazon/
    """
    logger.info(f"Processing {category} reviews...")
    
    # Read the gzipped JSON file
    data = []
    with gzip.open(f"data/raw/{category}.json.gz", 'rb') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} reviews to {output_path}")

if __name__ == "__main__":
    download_amazon_reviews() 