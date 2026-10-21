"""
Unit tests for the AI Career Mentor knowledge base module (src.mentor.knowledge_base).
"""

from pathlib import Path
import faiss
import numpy as np
import pytest

from src.mentor.knowledge_base import (
    chunk_career_notes,
    chunk_job_descriptions,
    extract_role_title_from_markdown,
    load_career_notes,
    load_mentor_index,
    load_relevant_jobs,
    normalize_vectors,
    retrieve_mentor_evidence,
)


def test_extract_role_title_from_markdown():
    md = "# Senior Backend Developer\n\n## Role guide\nSome text here."
    title = extract_role_title_from_markdown(md)
    assert title == "Senior Backend Developer"

    md_no_h1 = "## Role guide\nSome text here."
    title_fallback = extract_role_title_from_markdown(md_no_h1, fallback="Fallback Title")
    assert title_fallback == "Fallback Title"


def test_load_career_notes():
    notes = load_career_notes()
    assert len(notes) >= 19, "Expected at least 19 career notes to be loaded"
    for note in notes:
        assert "source" in note
        assert note["source"].endswith(".md")
        assert "title" in note and len(note["title"]) > 0
        assert "content" in note and len(note["content"]) > 100


def test_chunk_career_notes_metadata():
    sample_notes = [
        {
            "source": "sample_role.md",
            "title": "Sample Engineer",
            "content": (
                "# Sample Engineer\n\n"
                "## Role guide\n"
                "An engineer designs and builds reliable systems at scale.\n\n"
                "### 1. Prerequisites and foundations\n"
                "- Core concepts: networks, algorithms, data structures.\n"
                "- Programming: Python and SQL."
            ),
        }
    ]
    chunks = chunk_career_notes(sample_notes, chunk_size=200, chunk_overlap=30)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["source_type"] == "career_note"
        assert chunk["source"] == "sample_role.md"
        assert chunk["title"] == "Sample Engineer"
        assert chunk["job_id"] is None
        assert "Section:" in chunk["text"]
        assert chunk["chunk_id"].startswith("note_sample_role_chunk_")


def test_chunk_job_descriptions():
    sample_jobs = [
        {
            "job_id": 101,
            "job_title": "Backend Developer",
            "company": "TechCorp",
            "location": "Bengaluru",
            "skills": "Python, Django, PostgreSQL",
            "description": "Design and build high-performance REST APIs.",
        }
    ]
    chunks = chunk_job_descriptions(sample_jobs, chunk_size=500)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["source_type"] == "job_description"
    assert chunk["job_id"] == 101
    assert chunk["company"] == "TechCorp"
    assert chunk["location"] == "Bengaluru"
    assert chunk["title"] == "Backend Developer"
    assert chunk["chunk_id"] == "job_101_chunk_0"
    assert "Job Posting: Backend Developer at TechCorp" in chunk["text"]


def test_normalize_vectors():
    vecs = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    normed = normalize_vectors(vecs)
    # First row norm should be 1.0
    assert np.isclose(np.linalg.norm(normed[0]), 1.0)
    # Zero vector should not produce NaN
    assert not np.isnan(normed[1]).any()
    # Third row norm should be 1.0
    assert np.isclose(np.linalg.norm(normed[2]), 1.0)


def test_load_persisted_mentor_index():
    index, metadata = load_mentor_index()
    assert isinstance(index, faiss.Index)
    assert index.ntotal > 0
    assert len(metadata) == index.ntotal
    # Check that both career notes and job descriptions are represented
    source_types = {item["source_type"] for item in metadata}
    assert "career_note" in source_types
    assert "job_description" in source_types


def test_retrieve_mentor_evidence_mock():
    # Test retrieving evidence with filtering
    results = retrieve_mentor_evidence(
        query="What are the prerequisites for python developer?",
        top_k=3,
        min_score=0.2,
    )
    assert len(results) > 0
    for r in results:
        assert "score" in r
        assert "text" in r
        assert "source" in r
        assert "source_type" in r
