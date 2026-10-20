"""
Unit tests for SmartHire end-to-end job matching workflow (src.search.matcher).
"""

from unittest.mock import MagicMock, patch
import pytest

from src.search.matcher import match_jobs_for_candidate


@pytest.fixture
def sample_profile():
    return {
        "name": "Alex Johnson",
        "target_role": "Backend Python Developer",
        "professional_summary": "Python Developer with 3 years experience building APIs with FastAPI and PostgreSQL.",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Git"],
        "experience": [
            {
                "title": "Backend Developer",
                "company": "Tech Corp",
                "dates": "2021 - Present",
                "description": "Built REST APIs with FastAPI and PostgreSQL.",
            }
        ],
        "projects": [
            {
                "title": "Payment Service",
                "technologies": ["Python", "FastAPI", "Stripe API"],
                "description": "Integrated payment gateway.",
            }
        ],
        "education": [
            {
                "degree": "B.S. in Computer Science",
                "institution": "Tech University",
                "year": "2021",
            }
        ],
    }


@pytest.fixture
def sample_analysis():
    return {
        "level": "mid",
        "search_text": "Experienced Python Backend Developer proficient in FastAPI, PostgreSQL, and Docker microservices.",
        "filters": {
            "levels": ["junior", "mid"],
            "locations": ["Bengaluru", "Remote"],
        },
    }


def test_match_jobs_end_to_end_success(sample_profile, sample_analysis):
    """Test full matching pipeline execution without rigid counts."""
    # 1. Mock embedder
    mock_embed = MagicMock(return_value=[0.1] * 64)

    # 2. Mock FAISS search returning 3 qualifying jobs
    mock_faiss_results = [
        {
            "rank": 1,
            "job_id": 1,
            "job_title": "Backend Python Developer",
            "company": "Alpha Tech",
            "location": "Bengaluru",
            "skills": "Python, FastAPI, Docker",
            "description": "Build high-throughput APIs.",
            "similarity_score": 0.88,
        },
        {
            "rank": 2,
            "job_id": 2,
            "job_title": "Junior Python Engineer",
            "company": "Beta Labs",
            "location": "Remote",
            "skills": "Python, Django, PostgreSQL",
            "description": "Develop and maintain web services.",
            "similarity_score": 0.82,
        },
        {
            "rank": 3,
            "job_id": 3,
            "job_title": "Full Stack Developer",
            "company": "Gamma Inc",
            "location": "Pune",
            "skills": "React, Python, Node.js",
            "description": "Full stack feature delivery.",
            "similarity_score": 0.77,
        },
    ]
    mock_search = MagicMock(return_value=mock_faiss_results)

    # 3. Mock LLM reranker evaluating the 3 jobs
    mock_reranked_results = [
        {
            **mock_faiss_results[0],
            "rank": 1,
            "suitability_score": 0.94,
            "suitability_reasoning": "Excellent match for Python and FastAPI requirements.",
            "matching_skills": ["Python", "FastAPI", "Docker"],
            "missing_skills": [],
            "experience_gaps": [],
        },
        {
            **mock_faiss_results[1],
            "rank": 2,
            "suitability_score": 0.82,
            "suitability_reasoning": "Good match, but candidate is slightly overqualified for junior title.",
            "matching_skills": ["Python", "PostgreSQL"],
            "missing_skills": ["Django"],
            "experience_gaps": [],
        },
        {
            **mock_faiss_results[2],
            "rank": 3,
            "suitability_score": 0.58,
            "suitability_reasoning": "Moderate fit; lacks React and Node.js frontend experience.",
            "matching_skills": ["Python"],
            "missing_skills": ["React", "Node.js"],
            "experience_gaps": ["No frontend experience."],
        },
    ]
    mock_rerank = MagicMock(return_value=mock_reranked_results)

    # Execute workflow with min_suitability_threshold=0.50
    results = match_jobs_for_candidate(
        candidate_profile=sample_profile,
        candidate_analysis=sample_analysis,
        embed_fn=mock_embed,
        search_fn=mock_search,
        rerank_fn=mock_rerank,
        min_suitability_threshold=0.50,
    )

    # Non-rigid count check: exactly 3 qualifying matches returned
    assert len(results) == 3
    assert results[0]["job_id"] == 1
    assert results[0]["suitability_score"] == 0.94
    assert results[1]["job_id"] == 2
    assert results[1]["suitability_score"] == 0.82
    assert results[2]["job_id"] == 3
    assert results[2]["suitability_score"] == 0.58

    # Verify functions were called with expected arguments
    mock_embed.assert_called_once_with(sample_analysis["search_text"])
    mock_search.assert_called_once()
    mock_rerank.assert_called_once()


def test_match_jobs_zero_qualifying_short_circuits(sample_profile, sample_analysis):
    """Test that zero FAISS results short-circuit without calling the LLM reranker."""
    mock_embed = MagicMock(return_value=[0.1] * 64)
    mock_search = MagicMock(return_value=[])  # 0 jobs passed similarity threshold
    mock_rerank = MagicMock()

    results = match_jobs_for_candidate(
        candidate_profile=sample_profile,
        candidate_analysis=sample_analysis,
        embed_fn=mock_embed,
        search_fn=mock_search,
        rerank_fn=mock_rerank,
    )

    assert results == []
    # Verify reranker was NEVER called
    assert mock_rerank.call_count == 0


def test_match_jobs_dynamic_suitability_filtering(sample_profile, sample_analysis):
    """
    Test dynamic final match count:
    Only jobs meeting min_suitability_threshold are returned, without padding.
    """
    mock_embed = MagicMock(return_value=[0.1] * 64)
    mock_search = MagicMock(return_value=[{"job_id": 1}, {"job_id": 2}, {"job_id": 3}, {"job_id": 4}])

    # 4 jobs evaluated with suitability scores: [0.90, 0.75, 0.55, 0.30]
    mock_reranked = [
        {"job_id": 1, "suitability_score": 0.90, "job_title": "Backend Dev"},
        {"job_id": 2, "suitability_score": 0.75, "job_title": "Python Eng"},
        {"job_id": 3, "suitability_score": 0.55, "job_title": "DevOps Eng"},
        {"job_id": 4, "suitability_score": 0.30, "job_title": "Frontend Lead"},
    ]
    mock_rerank = MagicMock(return_value=mock_reranked)

    # Filter with min_suitability_threshold = 0.70 and min_results = 0 (strict mode)
    results = match_jobs_for_candidate(
        candidate_profile=sample_profile,
        candidate_analysis=sample_analysis,
        embed_fn=mock_embed,
        search_fn=mock_search,
        rerank_fn=mock_rerank,
        min_suitability_threshold=0.70,
        min_results=0,
    )

    # Exactly 2 jobs meet the threshold (0.90 and 0.75)
    assert len(results) == 2
    assert results[0]["job_id"] == 1
    assert results[0]["rank"] == 1
    assert results[1]["job_id"] == 2
    assert results[1]["rank"] == 2


def test_match_jobs_soft_floor_never_returns_empty(sample_profile, sample_analysis):
    """Test that soft floor ensures at least min_results (or available evaluated jobs) are returned."""
    mock_embed = MagicMock(return_value=[0.1] * 64)
    mock_search = MagicMock(return_value=[{"job_id": i} for i in range(1, 6)])

    # All 5 jobs have low suitability scores below 0.50
    mock_reranked = [
        {"job_id": i, "suitability_score": 0.30 - (i * 0.02), "job_title": f"Job {i}"}
        for i in range(1, 6)
    ]
    mock_rerank = MagicMock(return_value=mock_reranked)

    results = match_jobs_for_candidate(
        candidate_profile=sample_profile,
        candidate_analysis=sample_analysis,
        embed_fn=mock_embed,
        search_fn=mock_search,
        rerank_fn=mock_rerank,
        min_suitability_threshold=0.50,
    )

    # Soft floor ensures at least 5 results are returned rather than empty
    assert len(results) == 5
    assert results[0]["rank"] == 1
    assert results[-1]["rank"] == 5



def test_match_jobs_invalid_profile():
    """Test ValueError raised when profile is not a dictionary."""
    with pytest.raises(ValueError, match="candidate_profile must be a dictionary"):
        match_jobs_for_candidate(candidate_profile=123)
