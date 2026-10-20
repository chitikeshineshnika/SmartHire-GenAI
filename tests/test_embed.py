"""
Tests for embedding generation module (src.search.embed).
"""

import pytest
from src.search.embed import embed_text, embed_batch, embed_texts, generate_embeddings


def test_embed_single_text():
    sample_query = "Software Engineer with Python, React, and PostgreSQL experience"
    vec = embed_text(sample_query)
    
    assert isinstance(vec, list)
    assert len(vec) == 3072
    assert all(isinstance(x, float) for x in vec[:10])


def test_embed_batch_texts():
    sample_jobs = [
        "Frontend Developer proficient in React and TypeScript.",
        "Data Scientist specializing in Machine Learning and Python."
    ]
    vecs = embed_batch(sample_jobs)
    
    assert isinstance(vecs, list)
    assert len(vecs) == 2
    assert len(vecs[0]) == 3072
    assert len(vecs[1]) == 3072


def test_generate_embeddings_interface():
    # String test
    single_res = generate_embeddings("Backend Python Developer")
    assert isinstance(single_res, list)
    assert len(single_res) == 3072

    # List test
    batch_res = generate_embeddings(["DevOps Engineer", "Cloud Architect"])
    assert isinstance(batch_res, list)
    assert len(batch_res) == 2
    assert len(batch_res[0]) == 3072


def test_embed_empty_validation():
    with pytest.raises(ValueError):
        embed_text("")

    with pytest.raises(ValueError):
        embed_text("   ")
