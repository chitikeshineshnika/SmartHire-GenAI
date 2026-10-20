"""
Embedding generation module for SmartHire GenAI.

Handles vector embedding generation using the Gemini Embedding model
(configured as EMBED_MODEL in src.config).

Supports:
1. Single string embedding: Used for candidate resumes and query search.
2. Batch embedding: Used for bulk ingestion of 500-1,000 job descriptions
   with chunking (50 per batch), rate-limit pacing, and automatic retries.
"""

import logging
import time
from typing import List, Union

from src.config import EMBED_BATCH_SIZE, EMBED_MODEL, EMBED_SLEEP_SECONDS
from src.parsing.resume_parser import client

import re

logger = logging.getLogger(__name__)


def _extract_retry_delay(err: Exception, default_delay: float = 55.0) -> float:
    """Extract retry delay in seconds from Gemini API error message or return default."""
    err_str = str(err)
    match = re.search(r"retry in ([\d\.]+)s", err_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            pass
    match_delay = re.search(r"retrydelay['\":\s]+(\d+)", err_str, re.IGNORECASE)
    if match_delay:
        try:
            return float(match_delay.group(1)) + 1.0
        except ValueError:
            pass
    return default_delay


def embed_text(text: str, max_retries: int = 5) -> List[float]:
    """
    Generate an embedding vector for a single string.

    Ideal for real-time candidate search queries and resume profile matching.

    Args:
        text: The text string to embed.
        max_retries: Maximum number of retry attempts on transient API errors.
        
    Returns:
        A list of floats representing the embedding vector (3072 dimensions).

    Raises:
        ValueError: If text is empty or not a string.
        RuntimeError: If the Gemini API fails after retries.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected string input, got {type(text).__name__}")
    
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text to embed cannot be empty or whitespace.")

    delay = 1.0
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=cleaned_text,
            )
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values
            raise RuntimeError("Gemini API returned an empty embedding response.")
        except Exception as e:
            last_error = e
            status_code = getattr(e, "code", None)
            logger.warning(
                f"[embed_text] Attempt {attempt}/{max_retries} failed (Status: {status_code}): {e}"
            )
            if attempt < max_retries:
                if status_code == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = _extract_retry_delay(e, default_delay=55.0)
                    logger.warning(
                        f"[embed_text] Rate limit encountered. Pausing {wait_time:.1f}s for quota window to reset..."
                    )
                    time.sleep(wait_time)
                else:
                    time.sleep(delay)
                    delay = min(delay * 2, 16.0)
            else:
                break

    raise RuntimeError(
        f"Failed to generate embedding after {max_retries} attempts: {last_error}"
    ) from last_error


def embed_batch(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    """
    Embed a single batch of texts (up to EMBED_BATCH_SIZE) in one API call.
    
    Args:
        texts: List of strings to embed in a single API call.
        max_retries: Maximum number of retry attempts on transient API errors.
        
    Returns:
        A list of embedding vectors corresponding to the input texts.

    Raises:
        ValueError: If texts is not a list, contains non-string items, or has empty/blank items.
        RuntimeError: If the Gemini API fails after retries.
    """
    if not isinstance(texts, list):
        raise ValueError(f"Expected a list of strings, got {type(texts).__name__}")

    if not texts:
        return []

    sanitized_texts: List[str] = []
    for idx, item in enumerate(texts):
        if not isinstance(item, str):
            raise ValueError(
                f"All items in texts must be strings. Item at index {idx} is of type {type(item).__name__}."
            )
        cleaned = item.strip()
        if not cleaned:
            raise ValueError(
                f"Item at index {idx} is empty or contains only whitespace."
            )
        sanitized_texts.append(cleaned)

    delay = 1.0
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=sanitized_texts,
            )
            if response.embeddings and len(response.embeddings) == len(sanitized_texts):
                return [emb.values for emb in response.embeddings]
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(sanitized_texts)}, "
                f"got {len(response.embeddings) if response.embeddings else 0}"
            )
        except Exception as e:
            last_error = e
            status_code = getattr(e, "code", None)
            logger.warning(
                f"[embed_batch] Attempt {attempt}/{max_retries} failed (Status: {status_code}): {e}"
            )
            if attempt < max_retries:
                if status_code == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = _extract_retry_delay(e, default_delay=55.0)
                    logger.warning(
                        f"[embed_batch] Rate limit encountered. Pausing {wait_time:.1f}s for quota window to reset..."
                    )
                    time.sleep(wait_time)
                else:
                    time.sleep(delay)
                    delay = min(delay * 2, 16.0)
            else:
                break

    raise RuntimeError(
        f"Failed to generate batch embeddings after {max_retries} attempts: {last_error}"
    ) from last_error



def embed_texts(
    texts: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
    sleep_seconds: float = EMBED_SLEEP_SECONDS,
) -> List[List[float]]:
    """
    Embed a large collection of texts (e.g. 500-1,000 job descriptions) in batches.
    
    Breaks input into chunks of size `batch_size` and paces calls using `sleep_seconds`
    to respect API rate limits.
    
    Args:
        texts: Full list of job descriptions or text documents to embed.
        batch_size: Number of texts per API call (default: 50).
        sleep_seconds: Pacing delay in seconds between consecutive batch calls (default: 2.0).
        
    Returns:
        A list of embedding vectors matching the order of the input texts.
    """
    if not isinstance(texts, list):
        raise ValueError(f"Expected a list of strings, got {type(texts).__name__}")
    
    if not texts:
        return []

    all_embeddings: List[List[float]] = []
    total_count = len(texts)

    for i in range(0, total_count, batch_size):
        chunk = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_count + batch_size - 1) // batch_size
        
        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(chunk)} items)...")
        chunk_embeddings = embed_batch(chunk)
        all_embeddings.extend(chunk_embeddings)

        # Sleep between batches if more remain to avoid triggering rate limits
        if i + batch_size < total_count and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return all_embeddings


def generate_embeddings(
    content: Union[str, List[str]],
    batch_size: int = EMBED_BATCH_SIZE,
    sleep_seconds: float = EMBED_SLEEP_SECONDS,
) -> Union[List[float], List[List[float]]]:
    """
    Unified entry point for generating embeddings.
    
    - If `content` is a string (e.g. candidate query), returns a single 1D vector (List[float]).
    - If `content` is a list of strings (e.g. job dataset), returns a 2D list of vectors (List[List[float]]).
    """
    if isinstance(content, str):
        return embed_text(content)
    elif isinstance(content, list):
        return embed_texts(content, batch_size=batch_size, sleep_seconds=sleep_seconds)
    else:
        raise ValueError(
            f"Unsupported content type '{type(content).__name__}'. "
            "Must be either a string or a list of strings."
        )
