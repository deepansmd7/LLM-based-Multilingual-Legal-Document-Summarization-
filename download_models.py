"""
download_models.py
------------------
Run this script FIRST to download AI models before starting the app.
This allows you to monitor download progress and handle any issues.

Usage: python download_models.py

This will download:
1. DistilBART summarization model (~1.2 GB)
2. MarianMT translation model (~300 MB)
"""

import os
import sys
from pathlib import Path

# Set up cache directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "hf_models")
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure environment for Hugging Face
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minute timeout


def check_disk_space():
    """Check if there's enough disk space (need ~2GB)"""
    import shutil
    stats = shutil.disk_usage(BASE_DIR)
    free_gb = stats.free / (1024**3)
    
    if free_gb < 3:
        print(f"⚠️  WARNING: Only {free_gb:.1f} GB free space available")
        print("   Recommended: At least 3 GB free space")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        print(f"✓ Disk space check passed ({free_gb:.1f} GB available)")


def download_summarizer():
    """Download the summarization model"""
    print("\n" + "="*70)
    print("DOWNLOADING SUMMARIZATION MODEL (DistilBART)")
    print("="*70)
    print("Model: sshleifer/distilbart-cnn-12-6")
    print("Size: ~1.2 GB")
    print("This may take 5-15 minutes depending on your internet speed")
    print("="*70 + "\n")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("Step 1/2: Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=CACHE_DIR,
            resume_download=True,
            force_download=False
        )
        print("✓ Tokenizer downloaded successfully\n")
        
        print("Step 2/2: Downloading model (this is the large file)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=CACHE_DIR,
            resume_download=True,
            force_download=False
        )
        print("✓ Model downloaded successfully\n")
        
        print("="*70)
        print("✓ SUMMARIZATION MODEL READY!")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        print("TROUBLESHOOTING TIPS:")
        print("1. Check your internet connection")
        print("2. Make sure you have stable WiFi/Ethernet")
        print("3. Try running the script again (it will resume)")
        print("4. If issue persists, try: pip install --upgrade transformers torch")
        print("5. Clear cache if corrupted: rm -rf hf_models/\n")
        return False


def download_translator():
    """Download the translation model"""
    print("\n" + "="*70)
    print("DOWNLOADING TRANSLATION MODEL (MarianMT)")
    print("="*70)
    print("Model: Helsinki-NLP/opus-mt-en-mul")
    print("Size: ~300 MB")
    print("="*70 + "\n")
    
    try:
        from transformers import MarianMTModel, MarianTokenizer
        
        print("Step 1/2: Downloading tokenizer...")
        tokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-mul",
            cache_dir=CACHE_DIR,
            resume_download=True,
            force_download=False
        )
        print("✓ Tokenizer downloaded successfully\n")
        
        print("Step 2/2: Downloading model...")
        model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-mul",
            cache_dir=CACHE_DIR,
            resume_download=True,
            force_download=False
        )
        print("✓ Model downloaded successfully\n")
        
        print("="*70)
        print("✓ TRANSLATION MODEL READY!")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        print("TROUBLESHOOTING TIPS:")
        print("1. Check your internet connection")
        print("2. Try running the script again (it will resume)")
        print("3. If issue persists, try: pip install --upgrade transformers\n")
        return False


def verify_models():
    """Verify that models are accessible"""
    print("\n" + "="*70)
    print("VERIFYING MODELS")
    print("="*70 + "\n")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from transformers import MarianMTModel, MarianTokenizer
        
        print("Checking summarization model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=CACHE_DIR,
            local_files_only=True  # Don't download, just verify
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=CACHE_DIR,
            local_files_only=True
        )
        print("✓ Summarization model verified\n")
        
        print("Checking translation model...")
        tokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-mul",
            cache_dir=CACHE_DIR,
            local_files_only=True
        )
        model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-mul",
            cache_dir=CACHE_DIR,
            local_files_only=True
        )
        print("✓ Translation model verified\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}\n")
        return False


def main():
    print("\n" + "="*70)
    print("LEXISUM AI - MODEL DOWNLOADER")
    print("="*70)
    print("\nThis script will download the required AI models for LexiSum.")
    print("Total download size: ~1.5 GB")
    print("\nIMPORTANT:")
    print("- Make sure you have a stable internet connection")
    print("- Don't close this window during download")
    print("- If download fails, you can re-run this script (it will resume)")
    print("="*70 + "\n")
    
    # Check disk space
    check_disk_space()
    
    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import transformers
        import torch
        print(f"✓ transformers {transformers.__version__}")
        print(f"✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install transformers torch sentencepiece")
        sys.exit(1)
    
    input("\nPress ENTER to start downloading models...")
    
    # Download models
    success = True
    
    print("\n[1/2] Downloading Summarization Model...")
    if not download_summarizer():
        success = False
    
    if success:
        print("\n[2/2] Downloading Translation Model...")
        if not download_translator():
            success = False
    
    # Verify everything
    if success:
        if verify_models():
            print("\n" + "="*70)
            print("✓✓✓ ALL MODELS DOWNLOADED AND VERIFIED! ✓✓✓")
            print("="*70)
            print("\nYou can now run the main application:")
            print("  python app.py")
            print("\nModels are cached in:", CACHE_DIR)
            print("="*70 + "\n")
        else:
            print("\n⚠️  Models downloaded but verification failed")
            print("Try running the app anyway - it might still work.\n")
    else:
        print("\n" + "="*70)
        print("❌ DOWNLOAD FAILED")
        print("="*70)
        print("\nPlease:")
        print("1. Check your internet connection")
        print("2. Make sure you have enough disk space")
        print("3. Run this script again (it will resume from where it stopped)")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        print("You can re-run this script to resume the download.\n")
        sys.exit(0)
