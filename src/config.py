
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_env_variable(name: str, default: str | None = None) -> str | None:
    """Return an environment variable or its default value."""
    return os.getenv(name, default)


APP_NAME = "SmartHire GenAI"

# API keys (keep real values in .env, never in code)
GEMINI_API_KEY = get_env_variable("GEMINI_API_KEY")

# Paths (work from any folder)
BASE_DIR = Path(__file__).resolve().parent.parent   # adjust if config.py is not in src/
JOBS_CSV = BASE_DIR / "data" / "jobs" / "Realistic job dataset.csv"
CLEANED_JOBS_CSV = BASE_DIR / "data" / "jobs" / "cleaned_jobs.csv"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
EMBEDDINGS_CACHE = VECTORSTORE_DIR / "job_embeddings.npy"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "jobs.faiss"
JOBS_METADATA_PATH = VECTORSTORE_DIR / "jobs_metadata.csv"
CAREER_NOTES_DIR = BASE_DIR / "data" / "career_notes"
MENTOR_VECTORSTORE_DIR = VECTORSTORE_DIR / "mentor"
MENTOR_INDEX_PATH = MENTOR_VECTORSTORE_DIR / "mentor.faiss"
MENTOR_METADATA_PATH = MENTOR_VECTORSTORE_DIR / "mentor_metadata.json"
MENTOR_EMBEDDINGS_CACHE = MENTOR_VECTORSTORE_DIR / "mentor_embeddings.npy"


# Models
MODEL_NAME = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

# Resume / PDF limits
MAX_RESUME_SIZE_MB = 5
MAX_RESUME_SIZE_BYTES = MAX_RESUME_SIZE_MB * 1024 * 1024
MAX_PDF_PAGES = 20
MIN_EXTRACTED_TEXT_CHARS = 50

# Job search settings
MAX_JOBS = 1000      # raise to 1000 if the free tier allows
MIN_DESCRIPTION_CHARS = 50
DESC_CHARS_FOR_EMBEDDING = 1000
EMBED_BATCH_SIZE = 50
EMBED_SLEEP_SECONDS = 2
DEFAULT_TOP_N_JOBS = 25
MAX_TOP_N_JOBS = 50
DEFAULT_SIMILARITY_THRESHOLD = float(get_env_variable("SIMILARITY_THRESHOLD", "0.50"))
MIN_RETRIEVAL_RESULTS = int(get_env_variable("MIN_RETRIEVAL_RESULTS", "5"))

