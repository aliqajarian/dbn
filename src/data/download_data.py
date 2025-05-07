import requests
import gzip
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data downloader.
        
        Args:
            data_dir (str): Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs for the datasets
        self.urls = {
            'books': "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz",
            'reviews': "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/reviews_categories/reviews_Books.jsonl.gz"
        }
    
    def download_file(self, url: str, filename: str) -> Path:
        """
        Download a file from a URL with progress bar.
        
        Args:
            url (str): URL to download from
            filename (str): Name to save the file as
            
        Returns:
            Path: Path to the downloaded file
        """
        file_path = self.data_dir / filename
        
        if file_path.exists():
            logger.info(f"File already exists at {file_path}")
            return file_path
        
        logger.info(f"Downloading {url} to {file_path}")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        logger.info(f"Downloaded {filename}")
        return file_path
    
    def download_all(self):
        """Download all required datasets."""
        downloaded_files = {}
        
        # Download books metadata
        downloaded_files['books'] = self.download_file(
            self.urls['books'],
            'meta_Books.jsonl.gz'
        )
        
        # Download reviews
        downloaded_files['reviews'] = self.download_file(
            self.urls['reviews'],
            'reviews_Books.jsonl.gz'
        )
        
        return downloaded_files

def main():
    """Main function to download the datasets."""
    downloader = DataDownloader()
    downloaded_files = downloader.download_all()
    
    logger.info("Downloaded files:")
    for key, path in downloaded_files.items():
        logger.info(f"{key}: {path}")

if __name__ == "__main__":
    main() 