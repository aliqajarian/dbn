import gzip
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import os

class AmazonBooksLoader:
    def __init__(self, data_path: str):
        """
        Initialize the Amazon Books data loader.
        
        Args:
            data_path (str): Path to the meta_Books.jsonl.gz file
        """
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
    
    def load_data(self, max_items: Optional[int] = None) -> pd.DataFrame:
        """
        Load the Amazon Books dataset from the JSONL.GZ file.
        
        Args:
            max_items (Optional[int]): Maximum number of items to load. If None, loads all items.
            
        Returns:
            pd.DataFrame: DataFrame containing the book data
        """
        self.logger.info(f"Loading data from {self.data_path}")
        
        data = []
        with gzip.open(self.data_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading books")):
                if max_items and i >= max_items:
                    break
                    
                try:
                    item = json.loads(line.strip())
                    # Extract relevant fields
                    processed_item = {
                        'asin': item.get('asin', ''),
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'categories': item.get('categories', []),
                        'price': item.get('price', 0.0),
                        'brand': item.get('brand', ''),
                        'main_category': item.get('main_category', ''),
                        'image_url': item.get('imageURLHighRes', [None])[0] if item.get('imageURLHighRes') else None,
                        'features': item.get('features', []),
                        'also_buy': item.get('also_buy', []),
                        'also_view': item.get('also_view', []),
                        'rank': item.get('rank', {}),
                        'rating': item.get('rating', 0.0),
                        'review_count': item.get('review_count', 0)
                    }
                    data.append(processed_item)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error decoding JSON at line {i}: {e}")
                except Exception as e:
                    self.logger.warning(f"Error processing item at line {i}: {e}")
        
        df = pd.DataFrame(data)
        self.logger.info(f"Loaded {len(df)} items")
        return df
    
    def get_reviews(self, reviews_file: str, max_reviews: int = None) -> pd.DataFrame:
        """
        Load reviews from the reviews file.
        
        Args:
            reviews_file (str): Path to the reviews file
            max_reviews (int, optional): Maximum number of reviews to load
            
        Returns:
            pd.DataFrame: DataFrame containing the reviews
        """
        reviews_file = Path(reviews_file)
        if not reviews_file.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_file}")
            
        self.logger.info(f"Loading reviews from {reviews_file}")
        reviews = []
        
        try:
            with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, desc="Loading reviews")):
                    if max_reviews and i >= max_reviews:
                        break
                    try:
                        review = json.loads(line)
                        reviews.append({
                            'asin': review.get('asin'),
                            'reviewerID': review.get('reviewerID'),
                            'reviewerName': review.get('reviewerName'),
                            'reviewText': review.get('reviewText'),
                            'overall': review.get('overall'),
                            'summary': review.get('summary'),
                            'unixReviewTime': review.get('unixReviewTime'),
                            'reviewTime': review.get('reviewTime'),
                            'helpful': review.get('helpful', [0, 0])
                        })
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse review at line {i}")
                        continue
        except Exception as e:
            self.logger.error(f"Error reading reviews file: {str(e)}")
            raise
            
        self.logger.info(f"Loaded {len(reviews)} reviews")
        return pd.DataFrame(reviews)
    
    def merge_books_and_reviews(self, books_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge books and reviews data.
        
        Args:
            books_df (pd.DataFrame): Books data
            reviews_df (pd.DataFrame): Reviews data
            
        Returns:
            pd.DataFrame: Merged DataFrame
        """
        # Merge on ASIN
        merged_df = pd.merge(
            reviews_df,
            books_df[['asin', 'title', 'main_category']],
            on='asin',
            how='left'
        )
        
        self.logger.info(f"Merged data contains {len(merged_df)} reviews")
        return merged_df
    
    def prepare_for_analysis(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare reviews data for analysis.
        
        Args:
            reviews_df (pd.DataFrame): Reviews DataFrame
            
        Returns:
            pd.DataFrame: Prepared DataFrame
        """
        # Rename columns for consistency
        df = reviews_df.rename(columns={
            'reviewText': 'review_text',
            'overall': 'rating',
            'reviewTime': 'review_time',
            'unixReviewTime': 'unix_review_time'
        })
        
        # Convert review time to datetime
        df['review_time'] = pd.to_datetime(df['review_time'])
        
        # Add length of review
        df['review_length'] = df['review_text'].str.len()
        
        # Add word count
        df['word_count'] = df['review_text'].str.split().str.len()
        
        return df 