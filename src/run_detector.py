import logging
from realtime_detector import RealTimeAnomalyDetector
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/model_config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize detector
    detector = RealTimeAnomalyDetector(
        model_path="models/anomaly_detector.pth",
        threshold=config['model']['threshold']
    )
    
    # Start detection
    logger.info("Starting anomaly detection...")
    detector.start()
    
    try:
        while True:
            # Simulate receiving new data
            # In production, this would come from your data source
            new_data = {
                'text': 'This is a sample review text',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'rating': 4.5
            }
            
            # Add data to detector
            detector.add_data(new_data)
            
            # Get and process results
            results = detector.get_results()
            for result in results:
                if result['is_anomaly']:
                    logger.warning(f"Anomaly detected: {result}")
                else:
                    logger.info(f"Normal data: {result}")
            
            time.sleep(1)  # Simulate real-time data arrival
            
    except KeyboardInterrupt:
        logger.info("Stopping anomaly detection...")
        detector.stop()

if __name__ == "__main__":
    main() 