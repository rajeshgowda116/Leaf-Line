from langchain_core.documents import Document

try:
    from .retriever import get_vector_store
    from .metadata_extractor import extract_canonical_metadata
except ImportError:
    from retriever import get_vector_store
    from metadata_extractor import extract_canonical_metadata


def add_documents(chunks):
    vector_store = get_vector_store()
    documents = []

    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            meta_input = chunk.get("metadata", chunk)
            meta = extract_canonical_metadata(
                file_path=chunk.get("source"),
                text_content=text,
                page=chunk.get("page", 1),
                chunk_index=idx,
                existing_meta=meta_input if isinstance(meta_input, dict) else {}
            )
            documents.append(Document(page_content=text, metadata=meta))
        elif isinstance(chunk, Document):
            meta = extract_canonical_metadata(
                text_content=chunk.page_content,
                existing_meta=chunk.metadata
            )
            chunk.metadata = meta
            documents.append(chunk)

    if documents:
        vector_store.add_documents(documents)
        print(f"   Added {len(documents)} chunks to ChromaDB with canonical metadata")