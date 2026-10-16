import hashlib
import json
import os
import sys
import glob
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Backend"))
sys.path.insert(0, str(ROOT_DIR / "Backend" / "Rag"))

try:
    from Backend.Rag.loader import lazy_load_pdf
    from Backend.Rag.splitter import split_documents
    from Backend.Rag.vectordb import add_documents
except ImportError:
    from loader import lazy_load_pdf
    from splitter import split_documents
    from vectordb import add_documents

# Project Data Directories
DATA2_PATH = ROOT_DIR / "data 2"
DATA_PATH = ROOT_DIR / "data"
PROCESSED_FILE = Path(__file__).resolve().parent / "processed_files.json"


def calculate_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_processed_files():
    if not PROCESSED_FILE.exists():
        return {}
    try:
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_processed_files(processed):
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=4)


def ingest_agricultural_documents():
    """
    Scans data 2/ (and data/ if present) for agricultural PDFs:
    - Beans, Brinjal, Chilli, Cucurbits, Okra, Spinach, Tomato, and Agritech guidebooks.
    Parses pages, chunks with canonical agricultural metadata, and embeds into ChromaDB.
    """
    processed = load_processed_files()

    candidate_dirs = [DATA2_PATH, DATA_PATH]
    pdf_files = []
    for d in candidate_dirs:
        if d.exists():
            pdf_files.extend(list(d.glob("*.pdf")))
            pdf_files.extend(list(d.glob("**/*.pdf")))

    # Deduplicate paths
    pdf_files = sorted(list(set(pdf_files)))
    print(f"\n🌱 LeafLine Agritech Ingestion: Found {len(pdf_files)} PDF guidebooks.")

    total_new_chunks = 0

    for pdf_path in pdf_files:
        current_hash = calculate_hash(pdf_path)
        path_key = str(pdf_path.resolve())

        if path_key in processed and processed[path_key] == current_hash:
            print(f"⏩ [Skipping] Already indexed: {pdf_path.name}")
            continue

        print(f"\n📖 [Processing] Ingesting agricultural guide: {pdf_path.name}")
        pages = list(lazy_load_pdf(pdf_path))
        if not pages:
            print(f"⚠️ [Warning] No text extracted from {pdf_path.name}")
            continue

        chunks = split_documents(pages)
        if chunks:
            add_documents(chunks)
            total_new_chunks += len(chunks)
            processed[path_key] = current_hash
            save_processed_files(processed)
            print(f"✅ [Completed] Indexed {len(chunks)} chunks for {pdf_path.name}")

    print(f"\n🎉 Ingestion complete. Added {total_new_chunks} total agricultural chunks to ChromaDB.\n")


if __name__ == "__main__":
    ingest_agricultural_documents()