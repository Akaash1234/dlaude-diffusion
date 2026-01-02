import os
from huggingface_hub import snapshot_download

# Set the cache directory to the local folder as requested
local_cache_dir = os.path.abspath("llada_model_cache")
os.environ["HF_HOME"] = local_cache_dir

model_id = "inclusionAI/LLaDA2.0-mini-preview"

print(f"Starting memory-efficient download to: {local_cache_dir}")
print("This script will NOT load the model into RAM. It only downloads the files.")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_cache_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("\n" + "="*40)
    print("DOWNLOAD SUCCESSFUL!")
    print("The model files are now stored locally in 'llada_model_cache'.")
    print("="*40)
except Exception as e:
    print(f"Download failed: {e}")
