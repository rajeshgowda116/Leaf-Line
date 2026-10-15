import os
import sys
from pathlib import Path

# Ensure Backend/Disease is on the Python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Enable test mode so unit tests do not attempt downloading large Hugging Face weights
os.environ["TESTING"] = "1"
