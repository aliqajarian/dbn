import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from data.amazon_loader import AmazonBooksLoader
from models.dbn import TimeSeriesDBN
from utils.visualization import plot_reconstruction_error
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_results():
    # Load data
    loader = AmazonBooksLoader()
    df = loader.load_data()
    df = loader.preprocess_data(df)
    
    # Load model
    model = TimeSeriesDBN(layer_sizes=[1000, 500, 200, 100])
    model.load_state_dict(torch.load("models/amazon_anomaly_detector.pth"))
    model.eval()
    
    # Process data
    preprocessor = TextPreprocessor(max_features=1000)
    feature_engineer = FeatureEngineer()
    
    text_features = preprocessor.preprocess(df['text'].values)
    behavioral_features = pd.DataFrame({
        'price': df['price'],
        'rating': df['rating'],
        'review_count': df['review_count'],
        'category_depth': df['category_depth'],
        'subcategory_count': df['subcategory_count']
    })
    
    features = feature_engineer.combine_features(text_features, behavioral_features)
    sequences = feature_engineer.create_sequences(features)
    
    # Detect anomalies
    anomalies, errors = detect_anomalies(model, torch.FloatTensor(sequences))
    
    # Analyze results
    df['is_anomaly'] = anomalies.numpy()
    df['error_score'] = errors.numpy()
    
    # Print summary
    print("\nAnomaly Detection Results:")
    print(f"Total records: {len(df)}")
    print(f"Anomalies detected: {df['is_anomaly'].sum()}")
    print(f"Anomaly percentage: {(df['is_anomaly'].sum() / len(df)) * 100:.2f}%")
    
    # Analyze anomalous records
    anomalous_df = df[df['is_anomaly']]
    
    print("\nTop 5 Anomalies:")
    for _, row in anomalous_df.nlargest(5, 'error_score').iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Error Score: {row['error_score']:.4f}")
        print(f"Price: ${row['price']:.2f}")
        print(f"Rating: {row['rating']:.1f}")
        print(f"Review Count: {row['review_count']}")
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='error_score', hue='is_anomaly', bins=50)
    plt.title('Distribution of Reconstruction Errors')
    plt.show()
    
    # Save results
    df.to_csv("data/processed/anomaly_results.csv", index=False)
    logger.info("Results saved to data/processed/anomaly_results.csv")

if __name__ == "__main__":
    analyze_results() 