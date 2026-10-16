from pathlib import Path

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from .embeddings import get_embeddings
except ImportError:
    from embeddings import get_embeddings

CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "leafline_agritech_knowledge"

_vector_store_instance = None
_retriever_instance = None


def get_vector_store():
    global _vector_store_instance
    if _vector_store_instance is None:
        embeddings = get_embeddings()
        try:
            if chromadb:
                client = chromadb.PersistentClient(path=str(CHROMA_PATH))
                _vector_store_instance = Chroma(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings
                )
            else:
                _vector_store_instance = Chroma(
                    persist_directory=str(CHROMA_PATH),
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings
                )
        except Exception as e:
            print(f"[Chroma Client Info] Initializing persistent vector store fallback: {e}")
            _vector_store_instance = Chroma(
                persist_directory=str(CHROMA_PATH),
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )

        try:
            doc_count = _vector_store_instance._collection.count()
        except Exception:
            doc_count = "unknown"
        print(f"[INIT] LeafLine Chroma Vector Store loaded with {doc_count} agricultural chunks.")
    return _vector_store_instance


def get_retriever(k: int = 4, filter_dict: dict = None):
    vector_store = get_vector_store()
    search_kwargs = {"k": k}
    if filter_dict:
        clean_filter = {k: v for k, v in filter_dict.items() if v is not None}
        if clean_filter:
            search_kwargs["filter"] = clean_filter
    return vector_store.as_retriever(search_kwargs=search_kwargs)