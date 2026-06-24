import os
import sys
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "llm.gguf"
URL = "https://huggingface.co/openmed-community/AFM-4.5B-OpenMed-RL-CoT-GGUF/resolve/main/AFM-4.5B-OpenMed-RL-CoT-q4_k_m.gguf"

if not MODELS_DIR.exists():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Target path: {MODEL_PATH}")
print(f"Downloading from: {URL}")

# Check for Hugging Face Token in environment
hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")

class ProgressRequest(urllib.request.Request):
    pass

# Custom opener to handle headers and progress
opener = urllib.request.build_opener()
if hf_token:
    print("HF Token detected, attaching to download headers.")
    opener.addheaders = [("Authorization", f"Bearer {hf_token}")]
else:
    print("No HF Token detected in environment/env. Proceeding as public download.")

urllib.request.install_opener(opener)

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = f"\rProgress: {percent:5.1f}% ({readsofar / (1024**2):.1f} MB / {totalsize / (1024**2):.1f} MB)"
        sys.stdout.write(s)
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloaded {readsofar / (1024**2):.1f} MB")

try:
    urllib.request.urlretrieve(URL, str(MODEL_PATH), reporthook)
    print("\nDownload completed successfully!")
except Exception as e:
    print(f"\nDownload failed: {e}")
    sys.exit(1)
