import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureEngineer:
    def __init__(self, n_components: int = 50):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        
    def extract_behavioral_features(self, 
                                  user_data: pd.DataFrame,
                                  time_window: str = '1D') -> pd.DataFrame:
        """Extract behavioral features from user data."""
        features = pd.DataFrame()
        
        # Review frequency
        features['review_frequency'] = user_data.groupby('user_id')['timestamp'].count()
        
        # Time-based features
        user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
        features['avg_time_between_reviews'] = user_data.groupby('user_id')['timestamp'].diff().mean()
        
        # Rating patterns
        features['avg_rating'] = user_data.groupby('user_id')['rating'].mean()
        features['rating_std'] = user_data.groupby('user_id')['rating'].std()
        
        # Text length features
        features['avg_review_length'] = user_data['text'].str.len().groupby(user_data['user_id']).mean()
        
        return features
    
    def combine_features(self, 
                        text_features: np.ndarray,
                        behavioral_features: pd.DataFrame) -> np.ndarray:
        """Combine text and behavioral features."""
        # Convert behavioral features to numpy array
        behavioral_array = behavioral_features.values
        
        # Combine features
        combined_features = np.hstack([text_features, behavioral_array])
        
        # Normalize
        normalized_features = self.scaler.fit_transform(combined_features)
        
        return normalized_features
    
    def reduce_dimensions(self, features: np.ndarray) -> np.ndarray:
        """Reduce feature dimensions using PCA."""
        return self.pca.fit_transform(features) 