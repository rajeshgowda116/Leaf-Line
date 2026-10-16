import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from .generate import get_llm, extract_text_from_chunk
    from .metadata_extractor import CROP_NAME_MAP, DISEASE_KEYWORDS
except ImportError:
    from generate import get_llm, extract_text_from_chunk
    from metadata_extractor import CROP_NAME_MAP, DISEASE_KEYWORDS

load_dotenv()

GREETINGS = {
    "hi", "hii", "hiii", "hello", "hey", "heyy", "namaste", "vanakkam", "namaskara",
    "good morning", "good afternoon", "good evening", "howdy",
    "who are you", "what is leafline", "thanks", "thank you", "bye", "goodbye",
    "good", "nice", "great", "okay", "ok", "got it", "understood", "kisan", "farmer"
}

FOLLOWUP_PRONOUNS = {
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "the crop", "my plant", "my plants", "the leaves", "the leaf", "the fruit",
    "this disease", "the fungus", "the pest", "same", "above", "such"
}

FOLLOWUP_PHRASES = [
    "give remedy", "give remedies", "how to cure", "how to treat", "how to treat it",
    "how to cure it", "cure for this", "treatment for it", "treatment for this",
    "how to spray", "how much to spray", "how much should i spray", "how much should we spray",
    "dosage", "what dosage", "how much dosage", "how much to use", "how to use",
    "how much per liter", "how much ml", "organic remedy", "organic method", "chemical option",
    "chemical spray", "can i use neem oil", "can we spray", "how often", "when to spray",
    "when should i spray", "why is it happening", "why so", "what causes it", "is it fungal",
    "is it bacterial", "is it viral", "symptoms of it", "signs of it", "how does it spread",
    "how to prevent it", "prevention for this", "prevent this", "avoid this",
    "crop rotation for it", "seed treatment for it", "more details", "explain more",
    "tell me more", "what should i do", "what to do now", "what to do", "solution for it"
]


def is_greeting(question: str) -> bool:
    if not question:
        return False
    q_clean = question.lower().strip().strip("!.,?")
    return q_clean in GREETINGS


def is_follow_up_question(question: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
    if not question or not question.strip():
        return False
    if is_greeting(question):
        return False
    if not history or len(history) == 0:
        return False

    q_lower = question.lower().strip()
    for phrase in FOLLOWUP_PHRASES:
        if phrase in q_lower:
            return True

    words = set(re.findall(r'\b\w+\b', q_lower))
    if words.intersection(FOLLOWUP_PRONOUNS):
        return True

    # Common agricultural action inquiries when context was already established
    action_keywords = {
        "spray", "spraying", "dose", "dosage", "remedy", "cure", "treat", "treatment",
        "prevent", "prevention", "control", "apply", "application", "medicine", "fungicide",
        "pesticide", "insecticide", "chemical", "organic"
    }
    if words.intersection(action_keywords):
        return True

    return False


def rewrite_question_with_history(current_question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Rewrites ambiguous farmer follow-up questions into standalone agricultural search queries.
    Example:
      History: User: "My tomato leaves have brown rings" -> Asst: "That is early blight."
      Current: "How much neem oil to spray for it?"
      Output:  "How much neem oil dosage to spray for tomato early blight?"
    """
    if not current_question or not current_question.strip():
        return current_question

    current_question_clean = current_question.strip()
    if not history or len(history) == 0:
        return current_question_clean
    if not is_follow_up_question(current_question_clean, history):
        return current_question_clean

    recent_history = history[-4:]
    formatted_turns = []
    for h in recent_history:
        q = h.get("question", "").strip()
        a = h.get("answer", "").strip()
        a_short = a[:300] + "..." if len(a) > 300 else a
        if q:
            formatted_turns.append(f"Farmer: {q}\nAgronomist: {a_short}")

    if not formatted_turns:
        return current_question_clean

    conversation_history = "\n\n".join(formatted_turns)
    prompt = f"""Given the following conversation history and a follow-up question from a farmer, rewrite the follow-up question into a single, complete, standalone agricultural search query that includes all necessary context (such as the specific crop name e.g. Tomato, Chilli, Brinjal, Okra, Cucurbits, Beans, Spinach, specific disease name e.g. Early Blight, Powdery Mildew, or terms referenced by pronouns like 'it', 'my plants', 'the spray', 'dosage').

DO NOT answer the question. ONLY return the rewritten search query in plain text.

Conversation History:
{conversation_history}

Follow-up Question: {current_question_clean}
Rewritten Standalone Query:"""

    try:
        llm = get_llm()
        if llm:
            resp = llm.invoke(prompt)
            rewritten = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            # Remove any wrapping quotes or prefixes
            rewritten = re.sub(r'^(Rewritten\s*Query\s*:\s*|Query\s*:\s*|["\'])', '', rewritten, flags=re.IGNORECASE)
            rewritten = rewritten.strip('"\'')
            if rewritten and len(rewritten) > 3:
                return rewritten
    except Exception as e:
        print(f"[Contextualizer Notice] Fallback to raw question: {e}")

    return current_question_clean
