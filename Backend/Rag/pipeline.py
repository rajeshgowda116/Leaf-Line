import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Backend"))
sys.path.insert(0, str(ROOT_DIR / "Backend" / "Rag"))

try:
    from Backend.Rag.contextualizer import rewrite_question_with_history, is_greeting
    from Backend.Rag.relevance_gate import detect_query_crop, filter_and_validate_candidates
    from Backend.Rag.retriever import get_retriever
    from Backend.Rag.generate import get_llm, build_prompt, extract_text_from_chunk
except ImportError:
    from contextualizer import rewrite_question_with_history, is_greeting
    from relevance_gate import detect_query_crop, filter_and_validate_candidates
    from retriever import get_retriever
    from generate import get_llm, build_prompt, extract_text_from_chunk


GREETING_RESPONSE = """🙏 **Namaste! Welcome to LeafLine Agronomist AI.**

I am your agricultural health assistant powered by verified agronomic guidebooks and plant pathology research.

Here is what I can help you with:
- 🍅 **Tomato**: Early Blight, Late Blight, Bacterial Wilt, Leaf Curl, Blossom End Rot, cultivation practices.
- 🌶️ **Chilli**: Anthracnose, Dieback, Thrips, Mites, Powdery Mildew, nursery management.
- 🍆 **Brinjal**: Shoot & Fruit Borer, Little Leaf, Phomopsis Blight, damping-off.
- 🌿 **Okra**: Yellow Vein Mosaic Virus, Powdery Mildew, Fruit Borer, seed treatment.
- 🥒 **Cucurbits (Cucumber, Gourds)**: Downy Mildew, Powdery Mildew, Fruit Fly, spacing.
- 🫘 **Beans & Spinach**: Rust, Anthracnose, Leaf Spots, nutrient deficiencies, organic bio-controls.

Ask me any question about your crops, or upload a leaf photo to diagnose diseases!"""


def ask_leafline_agronomist(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    crop_filter: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    End-to-End LeafLine Agricultural RAG Pipeline:
    1. Greeting detection.
    2. Contextual query rewriting using conversation history.
    3. Crop identification & vector retrieval from ChromaDB.
    4. Relevance gating & deduplication.
    5. Agricultural prompt construction.
    6. LLM response generation with citations.
    """
    clean_q = (question or "").strip()
    if not clean_q:
        return {
            "answer": "Please ask an agricultural question about your crops, disease symptoms, or treatment options.",
            "sources": [],
            "crop": "General",
            "resolved_query": ""
        }

    # 1. Quick Greeting Check
    if is_greeting(clean_q):
        return {
            "answer": GREETING_RESPONSE,
            "sources": [],
            "crop": "General",
            "resolved_query": clean_q,
            "is_greeting": True
        }

    # 2. Contextual Query Rewriting with Conversation History
    resolved_query = rewrite_question_with_history(clean_q, history=history)

    # 3. Detect Crop Context
    detected_crop = crop_filter or detect_query_crop(resolved_query) or detect_query_crop(clean_q)

    # 4. Vector Retrieval from ChromaDB
    retriever = get_retriever(k=top_k * 2)
    candidate_docs = retriever.invoke(resolved_query)

    # 5. Relevance Gating & Validation
    resolved_dict = {
        "crop": detected_crop,
        "topic": resolved_query
    }
    accepted_docs, debug_records = filter_and_validate_candidates(candidate_docs, resolved_dict)

    if not accepted_docs:
        accepted_docs = candidate_docs[:top_k]
    else:
        accepted_docs = accepted_docs[:top_k]

    # 6. Format Context from Retrieved Chunks
    context_parts = []
    sources = []
    seen_sources = set()

    for i, doc in enumerate(accepted_docs):
        text = extract_text_from_chunk(doc).strip()
        meta = getattr(doc, "metadata", {}) or {}
        source_name = meta.get("filename", Path(meta.get("source", "")).name or "Agritech Guide")
        page_num = meta.get("page", 1)
        doc_crop = meta.get("crop", "General")

        context_parts.append(
            f"--- Reference Document {i+1} [{source_name}, Page {page_num}, Crop: {doc_crop}] ---\n{text}\n"
        )

        source_key = f"{source_name}_p{page_num}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": source_name,
                "page": page_num,
                "crop": doc_crop,
                "category": meta.get("category", "General"),
                "snippet": text[:250] + ("..." if len(text) > 250 else "")
            })

    full_context = "\n".join(context_parts) if context_parts else "No direct matching document chunks found."

    # 7. Build Grounded Agricultural Prompt
    prompt = build_prompt(
        question=clean_q,
        context=full_context,
        crop=detected_crop or "General Horticulture"
    )

    # 8. Call LLM
    llm = get_llm()
    if not llm:
        return {
            "answer": "LeafLine LLM is currently not configured. Please ensure GEMINI_API_KEY is added to your .env file.",
            "sources": sources,
            "crop": detected_crop or "General",
            "resolved_query": resolved_query,
            "error": "LLM_NOT_CONFIGURED"
        }

    try:
        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        answer_text = f"An error occurred while generating agricultural guidance: {str(e)}"

    return {
        "answer": answer_text,
        "resolved_query": resolved_query,
        "crop": detected_crop or "General",
        "sources": sources,
        "total_chunks": len(accepted_docs),
        "debug_records": debug_records[:4]
    }


if __name__ == "__main__":
    test_q = "What are the common symptoms and organic remedies for tomato early blight?"
    print(f"\n🌾 Testing LeafLine RAG Pipeline with: '{test_q}'\n")
    res = ask_leafline_agronomist(test_q)
    print("Crop:", res.get("crop"))
    print("Resolved Query:", res.get("resolved_query"))
    print("\nAnswer:\n", res.get("answer"))
    print("\nSources:\n", res.get("sources"))
