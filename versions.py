# setup_env.py
import subprocess, sys

def pip_install(args):
    """args = list of tokens, e.g., ['torch','--index-url','https://...']"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

print("🔧 Upgrading pip...")
pip_install(["--upgrade", "pip"])

print("🧠 Installing PyTorch (CUDA 12.1 wheels)...")
# If you DON'T want CUDA, change 'cu121' to 'cpu' or drop the --index-url line
pip_install([
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121"
])

print("📦 Installing other packages...")
pip_install([
    "transformers",
    "accelerate",
    "requests",
    "beautifulsoup4",
    "pandas",
    "matplotlib",
    "seaborn",
])

print("✅ All packages installed successfully!")

# Optional: print versions to verify
print("\nVersions:")
import torch, transformers, pandas, bs4, matplotlib, seaborn, requests
print("torch:", torch._version_)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers._version_)
print("pandas:", pandas._version_)
print("bs4:", bs4._version_)
print("matplotlib:", matplotlib._version_)
print("seaborn:", seaborn._version_)
print("requests:", requests._version_)