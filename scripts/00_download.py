import os
from datasets import load_dataset
from tqdm import tqdm
import shutil

# Configuration
DATASET_NAME = "vldsavelyev/guitar_tab"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "raw")

def download_and_save_dataset():
    """
    Downloads the guitar_tab dataset from Hugging Face and saves it locally.
    """
    from huggingface_hub import hf_hub_download
    import zipfile

    print(f"Downloading dataset: {DATASET_NAME}...")
    
    try:
        # download data.zip
        local_zip_path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename="data.zip",
            repo_type="dataset",
            local_dir=OUTPUT_DIR
        )
        
        print(f"Downloaded zip to {local_zip_path}")
        
        # Unzip
        extract_path = os.path.join(OUTPUT_DIR, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        
        print(f"Extracting to {extract_path}...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        print(f"Download and extraction complete. Data saved to {extract_path}")
        
    except Exception as e:
        import traceback
        print(f"Error downloading dataset: {e}")
        traceback.print_exc()
        print("Please ensure you have internet access and the 'huggingface_hub' library installed.")

if __name__ == "__main__":
    download_and_save_dataset()
