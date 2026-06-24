import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

repo_id = os.getenv("HF_MODEL_ID", "NCAIR1/NigerianAccentedEnglish")
token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")

print(f"ASR Model: {repo_id}")
print("Starting download from Hugging Face...")

try:
    snapshot_download(
        repo_id=repo_id,
        token=token,
        repo_type="model"
    )
    print("\nASR model downloaded and cached successfully!")
except Exception as e:
    print(f"\nDownload failed: {e}")
