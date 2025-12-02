"""Utility to download model files from GitHub releases"""
import os
import requests
from pathlib import Path


class ModelDownloader:
    """Downloads model files from GitHub releases"""
    
    # GitHub release URLs for model files
    GITHUB_RELEASE_BASE = "https://github.com/CyberwizD/Drug-Recommendation-System/releases/download/v1.0.0"
    
    MODEL_FILES = {
        "best_model.pkl": f"{GITHUB_RELEASE_BASE}/best_model.pkl",
        "recommendation_engine.pkl": f"{GITHUB_RELEASE_BASE}/recommendation_engine.pkl",
        "feature_engineer.pkl": f"{GITHUB_RELEASE_BASE}/feature_engineer.pkl",
    }
    
    @staticmethod
    def download_file(url: str, destination: str) -> bool:
        """Download a file from URL to destination"""
        try:
            print(f"Downloading {os.path.basename(destination)}...")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"Progress: {progress:.1f}%", end='\r')
            
            print(f"\n✓ Downloaded {os.path.basename(destination)}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading {os.path.basename(destination)}: {e}")
            return False
    
    @classmethod
    def ensure_models_exist(cls, models_dir: str = "models") -> bool:
        """Ensure all required model files exist, download if missing"""
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        all_exist = True
        downloaded_any = False
        
        for filename, url in cls.MODEL_FILES.items():
            file_path = models_path / filename
            
            if not file_path.exists():
                print(f"\n⚠ Model file missing: {filename}")
                print(f"Downloading from GitHub releases...")
                
                if cls.download_file(url, str(file_path)):
                    downloaded_any = True
                else:
                    all_exist = False
            else:
                print(f"✓ Model file exists: {filename}")
        
        if downloaded_any:
            print("\n✓ All models downloaded successfully!")
        
        return all_exist
    
    @classmethod
    def download_all_models(cls, models_dir: str = "models", force: bool = False) -> bool:
        """Download all model files (optionally force re-download)"""
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        success_count = 0
        
        for filename, url in cls.MODEL_FILES.items():
            file_path = models_path / filename
            
            if force or not file_path.exists():
                if cls.download_file(url, str(file_path)):
                    success_count += 1
            else:
                print(f"✓ Skipping {filename} (already exists)")
                success_count += 1
        
        return success_count == len(cls.MODEL_FILES)


if __name__ == "__main__":
    # Test the downloader
    print("Testing model downloader...")
    ModelDownloader.ensure_models_exist()
