import requests
import gzip
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import os
import json

class DataDownloader:
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data downloader.
        
        Args:
            data_dir (str): Directory to save downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Amazon Books dataset URLs
        self.urls = {
            'books': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz',
            'reviews': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Books_5.json.gz'
        }
    
    def verify_gzip_file(self, filepath: Path) -> bool:
        """
        Verify that a file is a valid gzip file.
        
        Args:
            filepath (Path): Path to the file to verify
            
        Returns:
            bool: True if file is valid gzip, False otherwise
        """
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                # Try to read first line
                f.readline()
            return True
        except Exception as e:
            self.logger.error(f"File {filepath} is not a valid gzip file: {str(e)}")
            return False
    
    def download_file(self, url: str, filename: str) -> str:
        """
        Download a file from URL with progress bar.
        
        Args:
            url (str): URL to download from
            filename (str): Name to save the file as
            
        Returns:
            str: Path to the downloaded file
        """
        filepath = self.data_dir / filename
        
        # Check if file exists and is valid
        if filepath.exists():
            if self.verify_gzip_file(filepath):
                self.logger.info(f"Valid file already exists: {filepath}")
                return str(filepath)
            else:
                self.logger.warning(f"Existing file is invalid, removing: {filepath}")
                filepath.unlink()
        
        try:
            # Stream the download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            # Verify the downloaded file
            if not self.verify_gzip_file(filepath):
                raise ValueError(f"Downloaded file {filename} is not a valid gzip file")
            
            self.logger.info(f"Successfully downloaded and verified {filename}")
            return str(filepath)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading {filename}: {str(e)}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            raise
    
    def download_all(self) -> dict:
        """
        Download all required data files.
        
        Returns:
            dict: Dictionary of downloaded file paths
        """
        downloaded_files = {}
        
        try:
            # Download books metadata
            books_path = self.download_file(
                self.urls['books'],
                'meta_Books.json.gz'
            )
            downloaded_files['books'] = books_path
            
            # Download reviews
            reviews_path = self.download_file(
                self.urls['reviews'],
                'reviews_Books.json.gz'
            )
            downloaded_files['reviews'] = reviews_path
            
            return downloaded_files
            
        except Exception as e:
            self.logger.error(f"Failed to download data: {str(e)}")
            raise

def main():
    """Main function to download data."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize downloader
    downloader = DataDownloader()
    
    try:
        # Download all files
        downloaded_files = downloader.download_all()
        
        # Print downloaded file paths
        for file_type, path in downloaded_files.items():
            print(f"Downloaded {file_type}: {path}")
            
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 