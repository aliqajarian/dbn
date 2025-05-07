import sys
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Test the environment and dependencies"""
    logger.info("Testing environment...")
    
    # Test Python version
    logger.info(f"Python version: {sys.version}")
    
    # Test PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test pandas
    logger.info(f"Pandas version: {pd.__version__}")
    
    # Test numpy
    logger.info(f"NumPy version: {np.__version__}")
    
    # Test directory structure
    required_dirs = ['data/raw', 'data/processed', 'models']
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory {dir_path} exists")
    
    # Test GPU memory
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    logger.info("Environment test completed")

if __name__ == "__main__":
    test_environment() 