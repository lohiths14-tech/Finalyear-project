"""
AgriPulse 2.0 — DINOv2 Pretrained Weights Downloader
Downloads the official Meta Research DINOv2 ViT-S/14 checkpoint (approx 84 MB)
directly to backend/model_weights/dinov2_vits14_pretrain.pth for offline,
zero-network-hang inference and training.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
TARGET_DIR = Path(__file__).resolve().parent.parent / "backend" / "model_weights"
TARGET_FILE = TARGET_DIR / "dinov2_vits14_pretrain.pth"

def download_weights(target_path: Path = TARGET_FILE) -> bool:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 10_000_000:
        print(f"✅ DINOv2 checkpoint already present: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return True

    print(f"📥 Downloading DINOv2 ViT-S/14 weights from {WEIGHTS_URL}...")
    print(f"   Destination: {target_path}")

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   Progress: {percent}% [{downloaded_mb:.1f} MB / {total_mb:.1f} MB]")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(WEIGHTS_URL, target_path, reporthook=progress_hook)
        print("\n✅ Download completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if target_path.exists():
            target_path.unlink()
        return False

if __name__ == "__main__":
    success = download_weights()
    sys.exit(0 if success else 1)
