import os
import time
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from Backend.Rag.metadata_extractor import extract_canonical_metadata
except ImportError:
    from metadata_extractor import extract_canonical_metadata

OCR_CACHE_DIR = Path(__file__).resolve().parent / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_api_key():
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _ocr_page_with_gemini(pdf_path: Path, page_number: int) -> str:
    """
    Extracts text from scanned/raster PDF pages using Google Gemini Multimodal Vision,
    persisting results to an on-disk cache for instantaneous future reloads.
    """
    safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in pdf_path.stem)
    cache_file = OCR_CACHE_DIR / f"{safe_stem}_p{page_number + 1}.txt"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if len(content) > 20:
                    return content
        except Exception:
            pass

    api_key = _get_api_key()
    if not api_key:
        print("[OCR Warning] GEMINI_API_KEY not configured, cannot run OCR.")
        return ""

    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        if page_number >= len(doc):
            return ""

        page = doc[page_number]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        prompt_text = (
            "You are an expert agricultural document transcriber. "
            "Extract all readable agricultural text, headings, tables, pest and disease symptoms, "
            "chemical and organic treatments, spray dosages, and cultivation guidelines from this document page "
            "into clean, clear English markdown text:"
        )

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_img
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        # Retry with backoff for rate limits
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if content_parts:
                            extracted = content_parts[0].get("text", "").strip()
                            if extracted:
                                with open(cache_file, "w", encoding="utf-8") as f:
                                    f.write(extracted)
                                return extracted
                    return ""
            except urllib.error.HTTPError as he:
                if he.code == 429:
                    wait_time = (attempt + 1) * 3
                    print(f"      ⏳ Rate limit (429), pausing {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"      [OCR HTTP Error] {pdf_path.name} p.{page_number+1}: {he}")
                    break
            except Exception as e:
                print(f"      [OCR Network Error] {pdf_path.name} p.{page_number+1}: {e}")
                time.sleep(2)

    except Exception as e:
        print(f"[OCR Error] {pdf_path.name} p.{page_number+1}: {e}")

    return ""


def lazy_load_pdf(file_path):
    """
    Extracts text from PDF documents page by page.
    1. Checks for selectable digital text via PyMuPDF (fast).
    2. If page contains scanned raster graphics, transcribes with Gemini Multimodal OCR (cached).
    """
    pdf_file = Path(file_path)
    print(f"\n📂 Reading: {pdf_file.name}")

    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_file))
        total_pages = len(doc)

        for page_number in range(total_pages):
            page = doc[page_number]
            text = (page.get_text() or "").strip()

            # If no selectable text, invoke Gemini Multimodal OCR with disk cache
            if not text or len(text) < 40:
                print(f"   🔍 Page {page_number + 1}/{total_pages}: Scanned image detected, running Gemini OCR...")
                ocr_text = _ocr_page_with_gemini(pdf_file, page_number)
                if ocr_text:
                    text = ocr_text

            if text and text.strip():
                meta = extract_canonical_metadata(
                    file_path=str(pdf_file),
                    page=page_number + 1,
                    text_content=text
                )
                yield {
                    "text": text,
                    "page": page_number + 1,
                    "source": str(pdf_file),
                    "metadata": meta
                }
    except Exception as e:
        print(f"[Loader Error] {pdf_file.name}: {e}")