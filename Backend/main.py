import sys
import importlib.util
from pathlib import Path

# Add Disease directory to sys.path so modules resolve cleanly
DISEASE_DIR = Path(__file__).resolve().parent / "Disease"
if str(DISEASE_DIR) not in sys.path:
    sys.path.insert(0, str(DISEASE_DIR))

# Load the FastAPI app from Disease/main.py
spec = importlib.util.spec_from_file_location("disease_main", DISEASE_DIR / "main.py")
disease_main = importlib.util.module_from_spec(spec)
sys.modules["disease_main"] = disease_main
spec.loader.exec_module(disease_main)

app = disease_main.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
