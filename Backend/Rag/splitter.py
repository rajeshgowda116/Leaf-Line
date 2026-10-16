from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from backend.rag.metadata_extractor import extract_canonical_metadata
except ImportError:
    from metadata_extractor import extract_canonical_metadata


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []
    for document in documents:
        doc_text = document.get("text", "")
        if not doc_text or not doc_text.strip():
            continue

        texts = splitter.split_text(doc_text)
        base_meta = document.get("metadata", {})

        for idx, text in enumerate(texts):
            meta = extract_canonical_metadata(
                file_path=document.get("source"),
                text_content=text,
                page=document.get("page", 1),
                chunk_index=idx,
                existing_meta=base_meta if isinstance(base_meta, dict) else {}
            )
            chunks.append({
                "text": text,
                "page": document.get("page", 1),
                "source": document.get("source", ""),
                "metadata": meta
            })

    return chunks