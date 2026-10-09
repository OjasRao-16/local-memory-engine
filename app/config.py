from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "memories.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
# Tune this with the evaluation set; similarity scores vary by model and data.
DEFAULT_RELEVANCE_THRESHOLD = 0.25
