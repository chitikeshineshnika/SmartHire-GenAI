import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Literal

from pydantic import BaseModel, Field
from google.genai import types

from src.generate.prompts import CANDIDATE_ANALYSIS_PROMPT
from src.parsing.resume_parser import client
from src.config import MODEL_NAME

logger = logging.getLogger(__name__)

ANALYSIS_CACHE_DIR = Path("data/.analysis_cache")
ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MEM_ANALYSIS_CACHE: dict[str, dict] = {}


# ---------- Expected output structure ----------

class CandidateFilters(BaseModel):
    levels: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)


class CandidateAnalysis(BaseModel):
    level: Literal["fresher", "junior", "mid", "senior"]
    search_text: str
    filters: CandidateFilters


# ---------- Candidate analysis ----------

def analyze_candidate(
    profile: dict,
    allowed_levels: list[str],
    allowed_locations: list[str],
) -> dict:
    """
    Analyze a structured resume profile for job matching.

    Uses the existing Gemini client from resume_parser.py.
    Returns validated candidate-analysis data as a dictionary.
    Caches results deterministically by profile hash so repeated uploads reuse the fixed query.
    """

    if not isinstance(profile, dict):
        raise ValueError("profile must be a dictionary.")

    if not allowed_levels:
        raise ValueError("allowed_levels cannot be empty.")

    # -----------------------------------------------------------------------
    # Deterministic Cache Lookup (Fixed Query for Identical Profile)
    # -----------------------------------------------------------------------
    cache_payload = {
        "profile": profile,
        "allowed_levels": sorted(allowed_levels),
        "allowed_locations": sorted(allowed_locations),
    }
    cache_key = hashlib.md5(
        json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    if cache_key in _MEM_ANALYSIS_CACHE:
        return dict(_MEM_ANALYSIS_CACHE[cache_key])

    cache_file = ANALYSIS_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            cached_json = cache_file.read_text(encoding="utf-8", errors="replace")
            if cached_json and cached_json.strip():
                cached_data = json.loads(cached_json)
                _MEM_ANALYSIS_CACHE[cache_key] = cached_data
                return dict(cached_data)
        except Exception as cache_err:
            logger.warning("Could not read analysis cache %s: %s", cache_file.name, cache_err)

    # Insert dataset values without changing the prompt's JSON braces.
    prompt = (
        CANDIDATE_ANALYSIS_PROMPT
        .replace("{LEVELS}", json.dumps(allowed_levels))
        .replace("{LOCATIONS}", json.dumps(allowed_locations))
    )

    attempts = 3
    delay = 1.0
    response = None
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=(
                    "Analyze this structured resume profile:\n\n"
                    + json.dumps(profile, ensure_ascii=False)
                ),
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=CandidateAnalysis,
                ),
            )
            break
        except Exception as e:
            last_error = e
            code = getattr(e, "code", None)
            if code in (429, 500, 503, 504) and attempt < attempts:
                time.sleep(delay)
                delay = min(delay * 2, 4.0)
            else:
                break

    if response is None:
        if last_error:
            raise RuntimeError(f"Gemini API failure during candidate analysis: {last_error}") from last_error
        raise RuntimeError("Failed to obtain response from Gemini API.")

    response_text = getattr(response, "text", None)

    if not response_text or not response_text.strip():
        raise ValueError("Gemini returned an empty candidate analysis.")

    result = CandidateAnalysis.model_validate_json(response_text)
    data = result.model_dump()

    # Validate the LLM's output against actual dataset values.
    invalid_levels = set(data["filters"]["levels"]) - set(allowed_levels)
    if invalid_levels:
        raise ValueError(
            f"Invalid job levels returned: {sorted(invalid_levels)}"
        )

    invalid_locations = (
        set(data["filters"]["locations"]) - set(allowed_locations)
    )
    if invalid_locations:
        raise ValueError(
            f"Invalid job locations returned: {sorted(invalid_locations)}"
        )

    # Persist validated output to disk and memory cache
    try:
        ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        _MEM_ANALYSIS_CACHE[cache_key] = data
    except Exception as save_err:
        logger.warning("Could not write analysis cache: %s", save_err)

    return data