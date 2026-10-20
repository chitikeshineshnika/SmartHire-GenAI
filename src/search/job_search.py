import logging
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd

from src.config import (
    CLEANED_JOBS_CSV,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_N_JOBS,
    DESC_CHARS_FOR_EMBEDDING,
    EMBED_BATCH_SIZE,
    EMBED_SLEEP_SECONDS,
    EMBEDDINGS_CACHE,
    FAISS_INDEX_PATH,
    JOBS_CSV,
    JOBS_METADATA_PATH,
    MAX_JOBS,
    MAX_TOP_N_JOBS,
    MIN_DESCRIPTION_CHARS,
    MIN_RETRIEVAL_RESULTS,
    VECTORSTORE_DIR,
)
from src.search.embed import embed_text, embed_texts


logger = logging.getLogger(__name__)

KEEP_COLUMNS = ["job_id", "job_title", "company", "location", "skills", "description"]


def load_and_clean_jobs(csv_path: Optional[Path] = None) -> pd.DataFrame:
    source = csv_path or JOBS_CSV
    df = pd.read_csv(source)
    print("loaded:", len(df))

    # Support datasets with 'role' column (mapping to standard 'job_title')
    if "role" in df.columns and "job_title" not in df.columns:
        df = df.rename(columns={"role": "job_title"})

    # 1. keep only the needed columns
    df = df[KEEP_COLUMNS].copy()

    # 2. normalize text: fill NaN with "", strip spaces
    for col in ["job_title", "company", "location", "skills", "description"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # 3. drop missing / too-short descriptions
    df = df[df["description"].str.len() >= MIN_DESCRIPTION_CHARS]
    print("after dropping bad descriptions:", len(df))

    # 4. drop duplicates on a lowercased key
    key = (df["job_title"] + df["company"] + df["description"]).str.lower()
    df = df[~key.duplicated(keep="first")]
    print("after dropping duplicates:", len(df))

    # 5. apply the cap
    df = df.head(MAX_JOBS).reset_index(drop=True)
    print("final:", len(df))
    df.to_csv(CLEANED_JOBS_CSV, index=False)
    return df


def format_job_for_embedding(
    row: Union[pd.Series, dict],
    max_desc_chars: int = DESC_CHARS_FOR_EMBEDDING,
) -> str:
    """
    Format a job record into a structured string for vector embedding.
    """
    title = str(row.get("job_title", "")).strip()
    company = str(row.get("company", "")).strip()
    location = str(row.get("location", "")).strip()
    skills = str(row.get("skills", "")).strip()
    desc = str(row.get("description", "")).strip()

    if max_desc_chars and len(desc) > max_desc_chars:
        desc = desc[:max_desc_chars]

    return (
        f"Job Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Skills: {skills}\n"
        f"Description: {desc}"
    )


def validate_job_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, clean, and deduplicate job records before embedding.
    Ensures job_id is present, non-null, and unique.
    """
    if "job_id" not in df.columns:
        raise ValueError("Dataset missing required 'job_id' column.")

    valid_df = df.copy()

    # Ensure required columns exist
    for col in KEEP_COLUMNS:
        if col not in valid_df.columns:
            valid_df[col] = ""
        else:
            valid_df[col] = valid_df[col].fillna("").astype(str).str.strip()

    # Drop missing or empty job_id
    valid_df = valid_df[valid_df["job_id"].str.strip() != ""]
    valid_df = valid_df.dropna(subset=["job_id"])

    # Standardize job_id (convert numeric IDs to integer representation)
    numeric_ids = pd.to_numeric(valid_df["job_id"], errors="coerce")
    if numeric_ids.notnull().all():
        valid_df["job_id"] = numeric_ids.astype(int)

    # Deduplicate by job_id keeping the first occurrence
    initial_len = len(valid_df)
    valid_df = valid_df.drop_duplicates(subset=["job_id"], keep="first").reset_index(drop=True)
    dropped = initial_len - len(valid_df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} duplicate job_id record(s) from dataset.")

    return valid_df


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vector embeddings for cosine similarity indexing.
    """
    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors_f32, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors_f32 / norms


def load_vectorstore(
    vectorstore_dir: Optional[Path] = None,
) -> Tuple[Optional[faiss.Index], Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Load existing FAISS index, embeddings array, and metadata DataFrame from disk.
    Returns (None, None, None) if any required artifact is missing or invalid.
    """
    vdir = vectorstore_dir or VECTORSTORE_DIR
    index_path = vdir / "jobs.faiss"
    emb_path = vdir / "job_embeddings.npy"
    meta_path = vdir / "jobs_metadata.csv"

    if not (index_path.exists() and emb_path.exists() and meta_path.exists()):
        return None, None, None

    try:
        index = faiss.read_index(str(index_path))
        embeddings = np.load(emb_path)
        metadata_df = pd.read_csv(meta_path)

        # Standardize job_id in metadata if numeric
        numeric_ids = pd.to_numeric(metadata_df["job_id"], errors="coerce")
        if numeric_ids.notnull().all():
            metadata_df["job_id"] = numeric_ids.astype(int)

        # Verify integrity and 1:1 row alignment
        if not (len(embeddings) == index.ntotal == len(metadata_df)):
            logger.warning(
                f"Vector store size mismatch: embeddings={len(embeddings)}, "
                f"faiss={index.ntotal}, metadata={len(metadata_df)}. Rebuilding needed."
            )
            return None, None, None

        return index, embeddings, metadata_df
    except Exception as e:
        logger.warning(f"Error loading existing vector store: {e}")
        return None, None, None


def save_vectorstore_atomic(
    index: faiss.Index,
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    vectorstore_dir: Optional[Path] = None,
) -> None:
    """
    Atomically save FAISS index, embeddings array, and metadata DataFrame.
    Uses temporary files and atomic replacement to prevent corruption on interruption.
    """
    vdir = vectorstore_dir or VECTORSTORE_DIR
    vdir.mkdir(parents=True, exist_ok=True)

    # Validate alignment before writing
    if not (len(embeddings) == index.ntotal == len(metadata_df)):
        raise ValueError(
            f"Cannot save unaligned vector store: embeddings={len(embeddings)}, "
            f"faiss={index.ntotal}, metadata={len(metadata_df)}"
        )

    pid = os.getpid()
    temp_emb = vdir / f"job_embeddings_{pid}.tmp.npy"
    temp_faiss = vdir / f"jobs_{pid}.tmp.faiss"
    temp_meta = vdir / f"jobs_metadata_{pid}.tmp.csv"

    target_emb = vdir / "job_embeddings.npy"
    target_faiss = vdir / "jobs.faiss"
    target_meta = vdir / "jobs_metadata.csv"

    try:
        np.save(temp_emb, embeddings)
        faiss.write_index(index, str(temp_faiss))
        metadata_df.to_csv(temp_meta, index=False)

        # Atomic replacement
        os.replace(temp_emb, target_emb)
        os.replace(temp_faiss, target_faiss)
        os.replace(temp_meta, target_meta)
    except Exception as e:
        # Clean up temporary files on failure
        for tmp_file in [temp_emb, temp_faiss, temp_meta]:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
        raise RuntimeError(f"Failed to atomically persist vector store: {e}") from e


def build_or_update_job_embeddings(
    csv_path: Optional[Path] = None,
    vectorstore_dir: Optional[Path] = None,
    batch_size: int = EMBED_BATCH_SIZE,
    sleep_seconds: float = EMBED_SLEEP_SECONDS,
    embedding_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
) -> dict:
    """
    Safely build or incrementally update job embeddings and FAISS index.

    - Reads jobs from csv_path (defaults to CLEANED_JOBS_CSV).
    - If a valid saved vector store exists, only jobs with missing job_id are embedded.
    - If no new jobs need embedding, generates zero API calls.
    - If no saved vector store exists, performs an initial baseline embedding pass.
    - Preserves atomicity: failures during embedding or saving do not corrupt existing data.
    """
    source_csv = csv_path or CLEANED_JOBS_CSV
    vdir = vectorstore_dir or VECTORSTORE_DIR
    embedder = embedding_fn or embed_texts

    if not source_csv.exists():
        raise FileNotFoundError(f"Source jobs CSV not found at {source_csv}")

    raw_df = pd.read_csv(source_csv)
    df = validate_job_records(raw_df)

    if df.empty:
        raise ValueError("Source dataset has no valid job records to embed.")

    existing_index, existing_embeddings, existing_metadata = load_vectorstore(vdir)

    if existing_index is not None and existing_embeddings is not None and existing_metadata is not None:
        saved_job_ids = set(existing_metadata["job_id"].tolist())
        jobs_to_embed_df = df[~df["job_id"].isin(saved_job_ids)].copy().reset_index(drop=True)
        index = existing_index
        embeddings = existing_embeddings
        metadata = existing_metadata
        is_new_store = False
    else:
        logger.info("No existing saved vector store found. Establishing baseline...")
        jobs_to_embed_df = df.copy().reset_index(drop=True)
        index = None
        embeddings = None
        metadata = None
        is_new_store = True

    if jobs_to_embed_df.empty:
        logger.info("All jobs are already embedded. Vector store is up to date.")
        return {
            "status": "up_to_date",
            "new_embedded": 0,
            "total_jobs": len(metadata),
        }

    total_to_embed = len(jobs_to_embed_df)
    total_batches = (total_to_embed + batch_size - 1) // batch_size
    logger.info(
        f"Processing {total_to_embed} jobs to embed in {total_batches} batch(es) (batch_size={batch_size})..."
    )

    embedded_count = 0
    for i in range(0, total_to_embed, batch_size):
        chunk_df = jobs_to_embed_df.iloc[i : i + batch_size].copy().reset_index(drop=True)
        batch_num = (i // batch_size) + 1
        logger.info(
            f"Embedding batch {batch_num}/{total_batches} ({len(chunk_df)} jobs)..."
        )

        texts = [format_job_for_embedding(row) for _, row in chunk_df.iterrows()]
        batch_vectors = embedder(texts, batch_size=batch_size, sleep_seconds=sleep_seconds)
        batch_vectors_norm = normalize_vectors(np.array(batch_vectors, dtype=np.float32))

        if index is None:
            dim = batch_vectors_norm.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(batch_vectors_norm)
            embeddings = batch_vectors_norm
            metadata = chunk_df
        else:
            index.add(batch_vectors_norm)
            embeddings = np.vstack([embeddings, batch_vectors_norm])
            metadata = pd.concat([metadata, chunk_df], ignore_index=True)

        # Checkpoint save after every batch
        save_vectorstore_atomic(index, embeddings, metadata, vdir)
        embedded_count += len(chunk_df)
        logger.info(f"Checkpoint saved: {len(metadata)} total jobs now safely persisted to disk.")

    status_str = "created" if is_new_store and embedded_count == len(metadata) else "updated"
    return {
        "status": status_str,
        "new_embedded": embedded_count,
        "total_jobs": len(metadata),
    }


def search_jobs(
    query: Union[str, np.ndarray, list[float]],
    top_k: int = DEFAULT_TOP_N_JOBS,
    similarity_threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD,
    min_results: int = MIN_RETRIEVAL_RESULTS,
    vectorstore_dir: Optional[Path] = None,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
) -> list[dict]:
    """
    Retrieve the closest job records using the existing FAISS index with soft floor relevance filtering.

    Args:
        query: Either a pre-computed vector (1D list/numpy array of floats) or a query text string.
        top_k: Maximum number of closest job records to return (defaults to DEFAULT_TOP_N_JOBS = 25).
        similarity_threshold: Cosine similarity soft floor (defaults to DEFAULT_SIMILARITY_THRESHOLD).
                              If None, no threshold filtering is applied.
        min_results: Minimum number of job records to return (defaults to MIN_RETRIEVAL_RESULTS = 5).
                     If fewer jobs meet similarity_threshold, the top min_results are returned (soft floor).
        vectorstore_dir: Path to directory containing saved vector store artifacts.
        embed_fn: Optional custom embedding function (defaults to embed_text).

    Returns:
        A list of dictionaries representing the qualifying job records, with keys:
        'rank', 'job_id', 'job_title', 'company', 'location', 'skills', 'description', and 'similarity_score'.
        Ordered from most to least similar. Only jobs meeting similarity_threshold are returned.

    Raises:
        ValueError: If input query, top_k, or similarity_threshold is invalid.
        RuntimeError: If saved FAISS index, embeddings, or metadata are missing or inconsistent.
    """
    # 1. Validate top_k and similarity_threshold
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")

    if similarity_threshold is not None:
        if not isinstance(similarity_threshold, (int, float)) or not (-1.0 <= similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be a float between -1.0 and 1.0, got {similarity_threshold}"
            )

    # 2. Check and load existing vector store without rebuilding
    vdir = vectorstore_dir or VECTORSTORE_DIR
    index, embeddings, metadata_df = load_vectorstore(vdir)

    if index is None or embeddings is None or metadata_df is None:
        raise RuntimeError(
            f"Saved FAISS index, embeddings, or metadata are missing or inconsistent in '{vdir}'. "
            "Cannot perform search. Vector store must be established before querying."
        )

    if index.ntotal == 0 or len(metadata_df) == 0:
        logger.warning("FAISS index or metadata is empty. Returning 0 search results.")
        return []

    # 3. Process query: vector vs text
    if isinstance(query, (list, np.ndarray)):
        query_arr = np.array(query, dtype=np.float32)
        if query_arr.size == 0 or not np.isfinite(query_arr).all():
            raise ValueError("Query vector cannot be empty or contain NaN/Inf values.")
        if query_arr.ndim == 1:
            query_arr = query_arr.reshape(1, -1)
        elif query_arr.ndim == 2:
            if query_arr.shape[0] != 1:
                raise ValueError(
                    f"Expected a single query vector of shape (1, d) or (d,), got shape {query_arr.shape}"
                )
        else:
            raise ValueError(f"Invalid query vector shape: {query_arr.shape}")
        query_vector = normalize_vectors(query_arr)
    elif isinstance(query, str):
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query text cannot be empty or whitespace.")
        embedder = embed_fn or embed_text
        raw_vec = embedder(cleaned_query)
        query_arr = np.array([raw_vec], dtype=np.float32)
        query_vector = normalize_vectors(query_arr)
    else:
        raise ValueError(
            f"Unsupported query type '{type(query).__name__}'. "
            "Must be either a text string or a vector (list or numpy array)."
        )

    # 4. Check dimension match
    if query_vector.shape[1] != index.d:
        raise ValueError(
            f"Query vector dimension ({query_vector.shape[1]}) does not match "
            f"FAISS index dimension ({index.d})."
        )

    # 5. Search FAISS index for top_k nearest candidates
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vector, k)

    # 6. Map positions to metadata and apply soft floor relevance threshold
    candidates = []
    for pos, score in zip(indices[0], distances[0]):
        # Handle invalid FAISS positions (-1 or out-of-range)
        if pos < 0 or pos >= len(metadata_df):
            continue

        score_f = round(float(score), 4)
        row = metadata_df.iloc[pos].to_dict()
        record = {
            "job_id": row.get("job_id"),
            "job_title": row.get("job_title", ""),
            "company": row.get("company", ""),
            "location": row.get("location", ""),
            "skills": row.get("skills", ""),
            "description": row.get("description", ""),
            "similarity_score": score_f,
        }
        candidates.append(record)

    # Relevance filtering with a soft floor:
    if similarity_threshold is not None:
        qualifying = [c for c in candidates if c["similarity_score"] >= similarity_threshold]
    else:
        qualifying = candidates

    # Soft floor: if fewer than min_results meet threshold, take top min_results
    # Ensures we never return empty and always return at least min_results when candidates exist
    if len(qualifying) < min_results and candidates:
        qualifying = candidates[:min(min_results, len(candidates))]

    # Assign sequential ranks to qualifying jobs
    for rank_idx, record in enumerate(qualifying, start=1):
        record["rank"] = rank_idx

    return qualifying




if __name__ == "__main__":
    df = load_and_clean_jobs()
    print(df.head())
