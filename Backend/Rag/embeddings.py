import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_embeddings_instance = None


def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        print("[INIT] Loading HuggingFaceEmbeddings model (sentence-transformers/all-MiniLM-L6-v2)...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings_instance