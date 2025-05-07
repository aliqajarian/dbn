import gzip
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_meta_books(file_path: str, sample_size: int = 5):
    """
    Analyze the structure and content of meta_Books.jsonl.gz file.
    
    Args:
        file_path (str): Path to the meta_Books.jsonl.gz file
        sample_size (int): Number of sample items to print
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Analyzing {file_path}")
    
    # Read and analyze the file
    items = []
    fields = set()
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Reading file")):
                try:
                    item = json.loads(line.strip())
                    items.append(item)
                    fields.update(item.keys())
                    
                    if i >= sample_size - 1:
                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON at line {i}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise
    
    # Print analysis results
    logger.info("\nFile Analysis Results:")
    logger.info(f"Total fields found: {len(fields)}")
    logger.info("\nFields:")
    for field in sorted(fields):
        logger.info(f"- {field}")
    
    logger.info("\nSample Items:")
    for i, item in enumerate(items):
        logger.info(f"\nItem {i + 1}:")
        for key, value in item.items():
            if isinstance(value, list):
                value = f"List with {len(value)} items"
            elif isinstance(value, dict):
                value = f"Dict with {len(value)} keys"
            elif isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            logger.info(f"{key}: {value}")

def main():
    """Main function to analyze the meta books file."""
    file_path = "data/raw/meta_Books.jsonl.gz"
    analyze_meta_books(file_path)

if __name__ == "__main__":
    main() 