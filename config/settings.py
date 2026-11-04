import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MD_DIR = DATA_DIR / "md"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"

for directory in [MD_DIR, OUTPUT_DIR, LOG_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = OUTPUT_DIR / "query_history.json"
LOG_FILE = LOG_DIR / "courseqa.log"

DEFAULT_TOP_K = 5
CHUNK_MIN_LENGTH = 20
BM25_K1 = 1.5
BM25_B = 0.75

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CourseQA")
