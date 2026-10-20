"""
Tests for FAISS job retrieval with relevance filtering (src.search.job_search.search_jobs).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import faiss

from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_TOP_N_JOBS, VECTORSTORE_DIR
from src.search.job_search import search_jobs, save_vectorstore_atomic


@pytest.fixture
def test_vectorstore_dir(tmp_path: Path) -> Path:
    """
    Fixture creating a temporary FAISS vector store with 35 jobs:
    - Jobs 1-12 have high similarity (>= 0.85) to query base vector.
    - Jobs 13-25 have moderate-high similarity (0.76 - 0.84) to query base vector.
    - Jobs 26-35 have low similarity (< 0.70) to query base vector.
    """
    vdir = tmp_path / "mock_vectorstore"
    vdir.mkdir(parents=True, exist_ok=True)

    dim = 64
    num_jobs = 35

    base_vec = np.zeros(dim, dtype=np.float32)
    base_vec[0] = 1.0  # Unit vector along axis 0

    rng = np.random.RandomState(42)
    vectors = []

    for i in range(num_jobs):
        noise = rng.randn(dim).astype(np.float32)
        noise[0] = 0.0
        noise /= np.linalg.norm(noise)

        if i < 12:
            # High similarity: target cosine sim ~ 0.88 - 0.95
            alpha = 0.90 + (i * 0.005)
        elif i < 25:
            # Moderate similarity: target cosine sim ~ 0.76 - 0.84
            alpha = 0.76 + ((i - 12) * 0.006)
        else:
            # Low similarity: target cosine sim ~ 0.20 - 0.60
            alpha = 0.30 + ((i - 25) * 0.03)

        beta = np.sqrt(max(0.0, 1.0 - alpha**2))
        vec = alpha * base_vec + beta * noise
        vec /= np.linalg.norm(vec)
        vectors.append(vec)

    embeddings = np.array(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    records = [
        {
            "job_id": i + 1,
            "job_title": f"Engineer Role {i + 1}",
            "company": f"Company {chr(65 + (i % 26))}",
            "location": "Bengaluru" if i % 2 == 0 else "Remote",
            "skills": "Python, FastAPI",
            "description": f"Description for job {i + 1}",
        }
        for i in range(num_jobs)
    ]
    metadata_df = pd.DataFrame(records)

    save_vectorstore_atomic(index, embeddings, metadata_df, vdir)
    return vdir


def test_search_jobs_exactly_25_qualifying(test_vectorstore_dir: Path):
    """Test retrieving exactly 25 qualifying jobs when >= 25 meet the similarity threshold."""
    dim = 64
    query_vec = np.zeros(dim, dtype=np.float32)
    query_vec[0] = 1.0  # Perfectly aligned with base_vec

    # Jobs 1 to 25 all have similarity >= 0.75
    results = search_jobs(
        query=query_vec,
        top_k=25,
        similarity_threshold=0.75,
        vectorstore_dir=test_vectorstore_dir,
    )

    assert len(results) == 25
    assert all(r["similarity_score"] >= 0.75 for r in results)
    assert results[0]["rank"] == 1
    assert results[-1]["rank"] == 25


def test_search_jobs_fewer_than_25_qualifying(test_vectorstore_dir: Path):
    """
    Test returning only the qualifying jobs (e.g. 12) when fewer than 25 meet threshold.
    Verifies that less-similar jobs are not added just to reach 25.
    """
    dim = 64
    query_vec = np.zeros(dim, dtype=np.float32)
    query_vec[0] = 1.0

    # With threshold 0.85, only jobs 0..11 qualify (exactly 12 jobs)
    results = search_jobs(
        query=query_vec,
        top_k=25,
        similarity_threshold=0.85,
        vectorstore_dir=test_vectorstore_dir,
    )

    assert len(results) == 12
    assert all(r["similarity_score"] >= 0.85 for r in results)
    # Ensure no padding with lower scores
    assert results[-1]["similarity_score"] >= 0.85
    assert results[-1]["rank"] == 12


def test_search_jobs_soft_floor_never_returns_empty(test_vectorstore_dir: Path):
    """Test that search acts as a soft floor and returns at least min_results (5) even with unattainable threshold."""
    dim = 64
    query_vec = np.zeros(dim, dtype=np.float32)
    query_vec[0] = 1.0

    # Unattainable threshold (e.g. 0.99) with default min_results=5
    results = search_jobs(
        query=query_vec,
        top_k=25,
        similarity_threshold=0.99,
        vectorstore_dir=test_vectorstore_dir,
    )

    # Soft floor guarantee: returns at least 5 results, never empty
    assert len(results) == 5
    assert results[0]["rank"] == 1
    assert results[-1]["rank"] == 5

    # When min_results is explicitly set to 0, strict filtering applies
    strict_results = search_jobs(
        query=query_vec,
        top_k=25,
        similarity_threshold=0.99,
        min_results=0,
        vectorstore_dir=test_vectorstore_dir,
    )
    assert strict_results == []


def test_search_jobs_score_ordering_and_fields(test_vectorstore_dir: Path):
    """Test that returned jobs are strictly ordered from highest to lowest similarity score."""
    dim = 64
    query_vec = np.zeros(dim, dtype=np.float32)
    query_vec[0] = 1.0

    results = search_jobs(
        query=query_vec,
        top_k=25,
        similarity_threshold=0.75,
        vectorstore_dir=test_vectorstore_dir,
    )

    assert len(results) > 0
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    # Check that rank increments from 1
    for idx, r in enumerate(results, start=1):
        assert r["rank"] == idx
        assert "job_id" in r
        assert "job_title" in r
        assert "company" in r
        assert "location" in r
        assert "skills" in r
        assert "description" in r


def test_search_jobs_text_query_with_mock_embed(test_vectorstore_dir: Path):
    """Test string query with custom embed_fn and threshold."""
    dim = 64

    def mock_embed(text: str) -> list[float]:
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 1.0  # Align with base vector
        return vec.tolist()

    results = search_jobs(
        query="Senior Software Engineer",
        top_k=25,
        similarity_threshold=0.85,
        vectorstore_dir=test_vectorstore_dir,
        embed_fn=mock_embed,
    )

    assert len(results) == 12
    assert all(r["similarity_score"] >= 0.85 for r in results)


def test_search_jobs_invalid_inputs(test_vectorstore_dir: Path):
    """Test handling of invalid thresholds, empty strings, and bad query types."""
    # Invalid similarity_threshold
    with pytest.raises(ValueError, match="similarity_threshold must be a float between -1.0 and 1.0"):
        search_jobs(query=[1.0] * 64, similarity_threshold=1.5, vectorstore_dir=test_vectorstore_dir)

    with pytest.raises(ValueError, match="similarity_threshold must be a float between -1.0 and 1.0"):
        search_jobs(query=[1.0] * 64, similarity_threshold=-2.0, vectorstore_dir=test_vectorstore_dir)

    # Empty string query
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        search_jobs(query="   ", vectorstore_dir=test_vectorstore_dir)

    # Invalid top_k
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_jobs(query="test", top_k=0, vectorstore_dir=test_vectorstore_dir)

    # Dimension mismatch
    with pytest.raises(ValueError, match="Query vector dimension \\(10\\) does not match"):
        search_jobs(query=[1.0] * 10, vectorstore_dir=test_vectorstore_dir)


def test_search_jobs_missing_store_fails_without_rebuilding(tmp_path: Path):
    """Test that missing vector store raises RuntimeError immediately without rebuilding."""
    empty_dir = tmp_path / "non_existent_vectorstore"
    empty_dir.mkdir(parents=True, exist_ok=True)

    dummy_vec = np.zeros(64, dtype=np.float32)
    with pytest.raises(RuntimeError, match="Saved FAISS index, embeddings, or metadata are missing or inconsistent"):
        search_jobs(query=dummy_vec, vectorstore_dir=empty_dir)


def test_search_jobs_against_actual_persisted_vectorstore():
    """Test search retrieval directly against the real 850-job saved vectorstore."""
    index_file = VECTORSTORE_DIR / "jobs.faiss"
    if not index_file.exists():
        pytest.skip("Actual vectorstore not yet initialized.")

    # Query using the embedding of the first job in the real dataset
    from src.search.job_search import load_vectorstore
    _, emb, _ = load_vectorstore()
    query_vec = emb[0]

    # Search with default N=25 and default threshold=0.75
    results = search_jobs(query=query_vec)

    assert len(results) <= DEFAULT_TOP_N_JOBS
    assert len(results) > 0  # Should find matching jobs
    for r in results:
        assert r["similarity_score"] >= DEFAULT_SIMILARITY_THRESHOLD
        assert r["job_id"] is not None

    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
