import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "space-ml-mission-control-3.1.10.zip"
OUTPUT_PATH = REPO_ROOT / ZIP_NAME

EXCLUDE_DIRS = {
    "node_modules",
    "dist",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "data",
    "models",
    "storage",
    "backup",
    "audit-results",
    "audit-results-sandbox"
}

EXCLUDE_EXTS = {
    ".part",
    ".pyc",
    ".pyo"
}

EXCLUDE_FILES = {
    ".env",
    ZIP_NAME,
    "archive.zip"
}

def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith(".pytest_cache"):
            return True
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix in EXCLUDE_EXTS:
        return True
    if "secret" in path.name.lower():
        return True
    return False

def make_zip():
    print(f"Creating clean release archive: {OUTPUT_PATH}...")
    file_count = 0
    total_size = 0
    
    # Remove existing zip if present
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        
    with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(REPO_ROOT):
            root_path = Path(root)
            
            # Prune dirs in-place
            dirs[:] = [d for d in dirs if not should_exclude(root_path / d)]
            
            for f in files:
                file_path = root_path / f
                if not should_exclude(file_path):
                    rel_path = file_path.relative_to(REPO_ROOT)
                    zf.write(file_path, arcname=str(rel_path))
                    file_count += 1
                    total_size += file_path.stat().st_size
                    
    archive_size = OUTPUT_PATH.stat().st_size
    print(f"Release ZIP created successfully!")
    print(f"Total files packed: {file_count}")
    print(f"Uncompressed size: {total_size / (1024*1024):.2f} MB")
    print(f"Archive size: {archive_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    make_zip()
