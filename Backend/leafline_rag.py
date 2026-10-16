"""
LeafLine AI - Agricultural RAG Retrieval Interface
--------------------------------------------------
Connects the vision diagnosis pipeline with the local ChromaDB vector store
(containing 406 indexed agricultural chunks from ICAR and TNAU guidebooks).
Provides grounded reference excerpts and agronomic context.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Import RAG pipeline and retriever modules
try:
    from Backend.Rag.pipeline import ask_leafline_agronomist
    from Backend.Rag.retriever import get_retriever, get_vector_store
    RAG_AVAILABLE = True
except ImportError:
    try:
        from Rag.pipeline import ask_leafline_agronomist
        from Rag.retriever import get_retriever, get_vector_store
        RAG_AVAILABLE = True
    except ImportError:
        logger.warning("RAG pipeline modules could not be imported.")
        RAG_AVAILABLE = False


def get_rag_treatment_context(crop: str, disease_name: str, k: int = 3) -> Dict[str, Any]:
    """
    Retrieves grounded context passages and document citations from ChromaDB
    for a given crop and suspected disease.
    """
    if not RAG_AVAILABLE:
        return {
            "has_rag_context": False,
            "citations": [],
            "excerpts": []
        }

    try:
        query = f"{crop} {disease_name} disease symptoms management treatment prevention"
        crop_clean = crop.lower().strip() if crop and crop.lower() not in ("none", "unknown") else None
        filter_dict = {"crop": crop_clean} if crop_clean else None

        retriever = get_retriever(k=k, filter_dict=filter_dict)
        docs = retriever.invoke(query)

        if not docs and filter_dict:
            retriever_broad = get_retriever(k=k)
            docs = retriever_broad.invoke(query)

        excerpts = []
        citations = []
        seen_sources = set()

        for doc in docs:
            content = doc.page_content.strip()
            meta = doc.metadata or {}
            source = meta.get("source", "Agricultural Research Guide")
            page = meta.get("page", None)

            excerpts.append(content[:400] + ("..." if len(content) > 400 else ""))

            citation_key = f"{source}_p{page}"
            if citation_key not in seen_sources:
                seen_sources.add(citation_key)
                citations.append({
                    "document": Path(source).name if source else "ICAR/TNAU Manual",
                    "page": page,
                    "crop": meta.get("crop", crop)
                })

        return {
            "has_rag_context": len(excerpts) > 0,
            "citations": citations,
            "excerpts": excerpts
        }

    except Exception as e:
        logger.error(f"Error querying RAG context: {e}")
        return {
            "has_rag_context": False,
            "citations": [],
            "excerpts": []
        }


def ask_rag_agronomist_wrapper(question: str, crop: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Wraps ask_leafline_agronomist for conversational queries."""
    if not RAG_AVAILABLE:
        return {
            "answer": "Agricultural RAG knowledge base is currently offline.",
            "sources": []
        }
    return ask_leafline_agronomist(question=question, history=history or [], crop_filter=crop)
