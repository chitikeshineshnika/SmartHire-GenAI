"""
Tests for safe incremental job embeddings and vector store persistence.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
import faiss

from src.search.job_search import (
    build_or_update_job_embeddings,
    format_job_for_embedding,
    load_vectorstore,
    save_vectorstore_atomic,
    validate_job_records,
)


def mock_embed_fn(dim: int = 64):
    """Factory creating a deterministic mock embedding function."""
    call_tracker = {"call_count": 0, "texts_embedded": []}

    def _embed(texts, batch_size=50, sleep_seconds=0):
        call_tracker["call_count"] += 1
        call_tracker["texts_embedded"].extend(texts)
        # Generate deterministic synthetic vectors
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 10000
            rng = np.random.RandomState(seed)
            vec = rng.randn(dim).astype(np.float32).tolist()
            vectors.append(vec)
        return vectors

    return _embed, call_tracker


@pytest.fixture
def sample_jobs_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "cleaned_jobs_sample.csv"
    data = [
        {
            "job_id": 1,
            "job_title": "Backend Python Developer",
            "company": "Tech Corp",
            "location": "Bengaluru",
            "skills": "Python, FastAPI, PostgreSQL",
            "description": "Design and build microservices in FastAPI.",
        },
        {
            "job_id": 2,
            "job_title": "Frontend React Developer",
            "company": "Designify",
            "location": "Remote",
            "skills": "React, TypeScript, CSS",
            "description": "Build modern responsive web interfaces.",
        },
        {
            "job_id": 3,
            "job_title": "Data Scientist",
            "company": "Data Insights",
            "location": "Mumbai",
            "skills": "Python, pandas, scikit-learn",
            "description": "Develop predictive ML models and pipelines.",
        },
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return csv_file


def test_validate_job_records():
    # Valid dataframe with duplicates and invalid IDs
    df = pd.DataFrame([
        {"job_id": "1", "job_title": "Role A", "company": "Co", "location": "Loc", "skills": "S", "description": "D"},
        {"job_id": "1", "job_title": "Role A duplicate", "company": "Co", "location": "Loc", "skills": "S", "description": "D"},
        {"job_id": "", "job_title": "Role B empty id", "company": "Co", "location": "Loc", "skills": "S", "description": "D"},
        {"job_id": "2", "job_title": "Role C", "company": "Co", "location": "Loc", "skills": "S", "description": "D"},
    ])

    validated = validate_job_records(df)
    assert len(validated) == 2
    assert list(validated["job_id"]) == [1, 2]
    assert validated.iloc[0]["job_title"] == "Role A"


def test_format_job_for_embedding():
    row = {
        "job_title": "ML Engineer",
        "company": "AI Labs",
        "location": "Remote",
        "skills": "PyTorch, Transformers",
        "description": "Build LLM applications." * 50,
    }
    formatted = format_job_for_embedding(row, max_desc_chars=50)
    assert "Job Title: ML Engineer" in formatted
    assert "Company: AI Labs" in formatted
    assert "Location: Remote" in formatted
    assert "Skills: PyTorch, Transformers" in formatted
    assert len(formatted.split("Description: ")[1]) <= 50


def test_initial_baseline_embedding(tmp_path: Path, sample_jobs_csv: Path):
    vdir = tmp_path / "vectorstore"
    embed_fn, tracker = mock_embed_fn(dim=64)

    result = build_or_update_job_embeddings(
        csv_path=sample_jobs_csv,
        vectorstore_dir=vdir,
        embedding_fn=embed_fn,
    )

    assert result["status"] == "created"
    assert result["new_embedded"] == 3
    assert result["total_jobs"] == 3
    assert tracker["call_count"] == 1
    assert len(tracker["texts_embedded"]) == 3

    # Verify saved files
    index, embeddings, metadata = load_vectorstore(vdir)
    assert index is not None
    assert embeddings is not None
    assert metadata is not None

    assert index.ntotal == 3
    assert embeddings.shape == (3, 64)
    assert len(metadata) == 3
    assert list(metadata["job_id"]) == [1, 2, 3]


def test_no_new_jobs_does_not_embed(tmp_path: Path, sample_jobs_csv: Path):
    vdir = tmp_path / "vectorstore"
    embed_fn, tracker = mock_embed_fn(dim=64)

    # Initial baseline
    build_or_update_job_embeddings(
        csv_path=sample_jobs_csv,
        vectorstore_dir=vdir,
        embedding_fn=embed_fn,
    )
    assert tracker["call_count"] == 1

    # Second run with exact same dataset
    result_second = build_or_update_job_embeddings(
        csv_path=sample_jobs_csv,
        vectorstore_dir=vdir,
        embedding_fn=embed_fn,
    )

    assert result_second["status"] == "up_to_date"
    assert result_second["new_embedded"] == 0
    assert result_second["total_jobs"] == 3
    # Tracker should NOT have received any new calls
    assert tracker["call_count"] == 1


def test_incremental_embedding_adds_only_new_jobs(tmp_path: Path, sample_jobs_csv: Path):
    vdir = tmp_path / "vectorstore"
    embed_fn, tracker = mock_embed_fn(dim=64)

    # 1. Establish baseline with 3 jobs
    build_or_update_job_embeddings(
        csv_path=sample_jobs_csv,
        vectorstore_dir=vdir,
        embedding_fn=embed_fn,
    )

    # 2. Append 2 new jobs to the CSV
    df = pd.read_csv(sample_jobs_csv)
    new_jobs = pd.DataFrame([
        {
            "job_id": 4,
            "job_title": "DevOps Engineer",
            "company": "CloudNative",
            "location": "Pune",
            "skills": "Kubernetes, Docker, AWS",
            "description": "Manage cloud infrastructure and CI/CD.",
        },
        {
            "job_id": 5,
            "job_title": "Mobile App Developer",
            "company": "Appify",
            "location": "Hyderabad",
            "skills": "Flutter, Dart, iOS",
            "description": "Create cross-platform mobile apps.",
        },
    ])
    updated_df = pd.concat([df, new_jobs], ignore_index=True)
    updated_csv = tmp_path / "updated_jobs.csv"
    updated_df.to_csv(updated_csv, index=False)

    # Reset tracker to inspect new calls
    tracker["call_count"] = 0
    tracker["texts_embedded"] = []

    # 3. Incremental update
    result = build_or_update_job_embeddings(
        csv_path=updated_csv,
        vectorstore_dir=vdir,
        embedding_fn=embed_fn,
    )

    assert result["status"] == "updated"
    assert result["new_embedded"] == 2
    assert result["total_jobs"] == 5
    assert tracker["call_count"] == 1
    assert len(tracker["texts_embedded"]) == 2
    assert "DevOps Engineer" in tracker["texts_embedded"][0]
    assert "Mobile App Developer" in tracker["texts_embedded"][1]

    # Verify vector store integrity and alignment
    index, embeddings, metadata = load_vectorstore(vdir)
    assert index.ntotal == 5
    assert embeddings.shape == (5, 64)
    assert len(metadata) == 5
    assert list(metadata["job_id"]) == [1, 2, 3, 4, 5]

    # Verify each row maps to the correct record
    for i in range(5):
        row = metadata.iloc[i]
        expected_id = i + 1
        assert row["job_id"] == expected_id


def test_failure_handling_does_not_corrupt_existing_data(tmp_path: Path, sample_jobs_csv: Path):
    vdir = tmp_path / "vectorstore"
    embed_fn, _ = mock_embed_fn(dim=64)

    # 1. Establish valid baseline
    build_or_update_job_embeddings(
        csv_path=sample_jobs_csv,
        vectorstore_dir=vdir,
        embedding_fn=embed_fn,
    )

    # Snapshot baseline state
    idx_before, emb_before, meta_before = load_vectorstore(vdir)
    assert idx_before.ntotal == 3

    # 2. Append new job
    df = pd.read_csv(sample_jobs_csv)
    new_job = pd.DataFrame([{
        "job_id": 99,
        "job_title": "Security Analyst",
        "company": "SecOps",
        "location": "Delhi",
        "skills": "SIEM, Penetration Testing",
        "description": "Monitor and secure systems.",
    }])
    updated_csv = tmp_path / "updated_jobs_fail.csv"
    pd.concat([df, new_job], ignore_index=True).to_csv(updated_csv, index=False)

    # 3. Simulate failure in embedding function
    def failing_embed_fn(texts, **kwargs):
        raise RuntimeError("Simulated Gemini API 500 error!")

    with pytest.raises(RuntimeError, match="Simulated Gemini API 500 error!"):
        build_or_update_job_embeddings(
            csv_path=updated_csv,
            vectorstore_dir=vdir,
            embedding_fn=failing_embed_fn,
        )

    # 4. Verify existing store was NOT corrupted or altered
    idx_after, emb_after, meta_after = load_vectorstore(vdir)
    assert idx_after.ntotal == 3
    assert np.array_equal(emb_before, emb_after)
    assert list(meta_after["job_id"]) == [1, 2, 3]
