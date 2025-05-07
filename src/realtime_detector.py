import torch
import numpy as np
from typing import Dict, List, Tuple
import queue
import threading
import time
from models.dbn import TimeSeriesDBN
from data.preprocessor import TextPreprocessor
from data.feature_engineering import FeatureEngineer

class RealTimeAnomalyDetector:
    def __init__(self, 
                 model_path: str,
                 sequence_length: int = 10,
                 threshold: float = 0.1,
                 batch_size: int = 32):
        self.model = TimeSeriesDBN.load_from_checkpoint(model_path)
        self.model.eval()
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.batch_size = batch_size
        
        self.preprocessor = TextPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        self.data_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        
    def start(self):
        """Start the real-time detection process."""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.start()
    
    def stop(self):
        """Stop the real-time detection process."""
        self.is_running = False
        self.processing_thread.join()
    
    def add_data(self, data: Dict):
        """Add new data to the processing queue."""
        self.data_queue.put(data)
    
    def get_results(self) -> List[Dict]:
        """Get detection results."""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results
    
    def _process_queue(self):
        """Process data from the queue in batches."""
        batch = []
        last_process_time = time.time()
        
        while self.is_running:
            try:
                # Get data from queue with timeout
                data = self.data_queue.get(timeout=1.0)
                batch.append(data)
                
                # Process batch if it's full or enough time has passed
                current_time = time.time()
                if (len(batch) >= self.batch_size or 
                    (len(batch) > 0 and current_time - last_process_time >= 1.0)):
                    self._process_batch(batch)
                    batch = []
                    last_process_time = current_time
                    
            except queue.Empty:
                # Process remaining data in batch
                if batch:
                    self._process_batch(batch)
                    batch = []
                    last_process_time = time.time()
    
    def _process_batch(self, batch: List[Dict]):
        """Process a batch of data."""
        # Extract features
        features = self._extract_features(batch)
        
        # Create sequences
        sequences = self._create_sequences(features)
        
        # Detect anomalies
        with torch.no_grad():
            anomalies, errors = self.model.detect_anomalies(
                torch.FloatTensor(sequences), 
                self.threshold
            )
        
        # Prepare results
        for i, (is_anomaly, error) in enumerate(zip(anomalies, errors)):
            result = {
                'timestamp': batch[i]['timestamp'],
                'is_anomaly': bool(is_anomaly),
                'error_score': float(error),
                'data': batch[i]
            }
            self.result_queue.put(result)
    
    def _e 