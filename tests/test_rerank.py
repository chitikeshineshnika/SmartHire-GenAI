"""
Unit tests for LLM job reranking module (src.search.rerank).
"""

import json
from unittest.mock import MagicMock
import pytest

from src.search.rerank import (
    JobEvaluation,
    RerankResponse,
    rerank_jobs,
)


@pytest.fixture
def sample_candidate_profile():
    return {
        "name": "Jane Doe",
        "target_role": "Backend Engineer",
        "professional_summary": "Experienced Python Backend Developer with 3 years building microservices with FastAPI and PostgreSQL.",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Git"],
        "experience": [
            {
                "title": "Backend Developer",
                "company": "CloudTech",
                "dates": "2021 - Present",
                "description": "Developed REST APIs using FastAPI and PostgreSQL, containerized with Docker.",
            }
        ],
        "projects": [
            {
                "title": "Task Manager API",
                "technologies": ["Python", "FastAPI", "Redis"],
                "description": "Built distributed task queue.",
            }
        ],
        "education": [
            {
                "degree": "B.Tech Computer Science",
                "institution": "State University",
                "year": "2021",
            }
        ],
        "certifications": ["AWS Certified Developer"],
    }


@pytest.fixture
def sample_qualifying_jobs():
    return [
        {
            "rank": 1,
            "job_id": 101,
            "job_title": "Senior Python Architect",
            "company": "ScaleUp",
            "location": "Bengaluru",
            "skills": "Python, Distributed Systems, Kubernetes, 8+ years",
            "description": "Lead architecture of large-scale distributed systems.",
            "similarity_score": 0.88,
        },
        {
            "rank": 2,
            "job_id": 102,
            "job_title": "Backend Developer (FastAPI)",
            "company": "Fintech Innovations",
            "location": "Remote",
            "skills": "Python, FastAPI, PostgreSQL, Docker, 2-4 years",
            "description": "Build high-throughput REST APIs for financial services.",
            "similarity_score": 0.84,
        },
        {
            "rank": 3,
            "job_id": 103,
            "job_title": "Junior Frontend Engineer",
            "company": "DesignHub",
            "location": "Mumbai",
            "skills": "React, TypeScript, CSS",
            "description": "Build interactive user interfaces.",
            "similarity_score": 0.76,
        },
    ]


def test_rerank_empty_qualifying_jobs(sample_candidate_profile):
    """Test that empty qualifying jobs list returns [] immediately without calling Gemini."""
    mock_client = MagicMock()

    result = rerank_jobs(
        candidate_profile=sample_candidate_profile,
        qualifying_jobs=[],
        gemini_client=mock_client,
    )

    assert result == []
    # Verify no LLM call was made
    assert mock_client.models.generate_content.call_count == 0


def test_rerank_invalid_profile():
    """Test that passing a non-dict profile raises ValueError."""
    with pytest.raises(ValueError, match="candidate_profile must be a dictionary"):
        rerank_jobs(candidate_profile="invalid profile string", qualifying_jobs=[{"job_id": 1}])


def test_rerank_successful_evaluation(sample_candidate_profile, sample_qualifying_jobs):
    """Test that LLM evaluations properly augment and rerank jobs by suitability score."""
    # Mock LLM evaluation: Job 102 (FastAPI Mid) is highest fit (0.92), Job 101 (Senior) is 0.65, Job 103 (Frontend) is 0.20
    mock_evaluations = [
        JobEvaluation(
            job_id=101,
            suitability_score=0.65,
            suitability_reasoning="Candidate has strong Python fundamentals but lacks the 8+ years and distributed systems depth required for a Principal/Senior Architect role.",
            matching_skills=["Python"],
            missing_skills=["Kubernetes", "Distributed Systems Architecture"],
            experience_gaps=["Needs 5 more years of enterprise experience."],
        ),
        JobEvaluation(
            job_id=102,
            suitability_score=0.92,
            suitability_reasoning="Exceptional alignment. Candidate's primary stack (FastAPI, PostgreSQL, Docker) and 3 years experience directly match the job requirements.",
            matching_skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
            missing_skills=[],
            experience_gaps=[],
        ),
        JobEvaluation(
            job_id=103,
            suitability_score=0.20,
            suitability_reasoning="Significant role mismatch. Job requires frontend development (React, TypeScript), whereas candidate is a dedicated Python backend engineer.",
            matching_skills=[],
            missing_skills=["React", "TypeScript", "CSS"],
            experience_gaps=["No frontend professional experience."],
        ),
    ]

    mock_response = MagicMock()
    mock_response.text = RerankResponse(evaluations=mock_evaluations).model_dump_json()

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    reranked = rerank_jobs(
        candidate_profile=sample_candidate_profile,
        qualifying_jobs=sample_qualifying_jobs,
        gemini_client=mock_client,
    )

    assert len(reranked) == 3

    # Check that job 102 was reranked to rank 1 based on suitability score 0.92
    assert reranked[0]["job_id"] == 102
    assert reranked[0]["rank"] == 1
    assert reranked[0]["suitability_score"] == 0.92
    assert "FastAPI" in reranked[0]["matching_skills"]
    assert reranked[0]["missing_skills"] == []

    # Check rank 2 is job 101 (score 0.65)
    assert reranked[1]["job_id"] == 101
    assert reranked[1]["rank"] == 2
    assert reranked[1]["suitability_score"] == 0.65
    assert len(reranked[1]["experience_gaps"]) > 0

    # Check rank 3 is job 103 (score 0.20)
    assert reranked[2]["job_id"] == 103
    assert reranked[2]["rank"] == 3
    assert reranked[2]["suitability_score"] == 0.20

    # Ensure all original job details are preserved
    for r in reranked:
        assert "job_title" in r
        assert "company" in r
        assert "location" in r
        assert "skills" in r
        assert "description" in r
        assert "similarity_score" in r
        assert "suitability_reasoning" in r


def test_rerank_partial_evaluation_fallback(sample_candidate_profile, sample_qualifying_jobs):
    """Test safe fallback when LLM output omits an evaluation for one job."""
    # Only return evaluation for job 101
    partial_evaluations = [
        JobEvaluation(
            job_id=101,
            suitability_score=0.95,
            suitability_reasoning="Excellent fit.",
            matching_skills=["Python"],
            missing_skills=[],
            experience_gaps=[],
        )
    ]

    mock_response = MagicMock()
    mock_response.text = RerankResponse(evaluations=partial_evaluations).model_dump_json()

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    reranked = rerank_jobs(
        candidate_profile=sample_candidate_profile,
        qualifying_jobs=sample_qualifying_jobs,
        gemini_client=mock_client,
    )

    assert len(reranked) == 3
    # Job 101 has 0.95 and is rank 1
    assert reranked[0]["job_id"] == 101
    assert reranked[0]["suitability_score"] == 0.95

    # Missing jobs should fall back gracefully without crashing
    job_ids = [r["job_id"] for r in reranked]
    assert set(job_ids) == {101, 102, 103}


def test_rerank_retry_on_transient_error(sample_candidate_profile, sample_qualifying_jobs):
    """Test that transient errors (e.g. 503) trigger retry and succeed."""
    mock_evaluations = [
        JobEvaluation(
            job_id=101,
            suitability_score=0.80,
            suitability_reasoning="Good match.",
            matching_skills=["Python"],
            missing_skills=[],
            experience_gaps=[],
        ),
        JobEvaluation(
            job_id=102,
            suitability_score=0.85,
            suitability_reasoning="Great match.",
            matching_skills=["Python", "FastAPI"],
            missing_skills=[],
            experience_gaps=[],
        ),
        JobEvaluation(
            job_id=103,
            suitability_score=0.30,
            suitability_reasoning="Weak match.",
            matching_skills=[],
            missing_skills=["React"],
            experience_gaps=[],
        ),
    ]

    mock_response = MagicMock()
    mock_response.text = RerankResponse(evaluations=mock_evaluations).model_dump_json()

    mock_error = RuntimeError("503 Service Unavailable")
    setattr(mock_error, "code", 503)

    mock_client = MagicMock()
    # Fails first attempt, succeeds on second attempt
    mock_client.models.generate_content.side_effect = [mock_error, mock_response]

    reranked = rerank_jobs(
        candidate_profile=sample_candidate_profile,
        qualifying_jobs=sample_qualifying_jobs,
        gemini_client=mock_client,
        max_retries=2,
    )

    assert len(reranked) == 3
    assert mock_client.models.generate_content.call_count == 2
    assert reranked[0]["job_id"] == 102


def test_rerank_fresher_strictly_penalizes_senior_roles():
    """Verify that fresher candidate receives penalized scores on Senior/Architect roles."""
    fresher_profile = {
        "name": "Alex Student",
        "target_role": "Python Developer Intern",
        "skills": ["Python", "Git", "SQL"],
        "experience": [],  # 0 years / student
        "projects": [{"title": "Coursework App", "technologies": ["Python"]}],
    }

    jobs = [
        {"job_id": 1, "job_title": "Senior Python Architect", "company": "BigCorp", "similarity_score": 0.85},
        {"job_id": 2, "job_title": "Python Developer Intern", "company": "StartupX", "similarity_score": 0.70},
    ]

    mock_evaluations = [
        JobEvaluation(job_id=1, suitability_score=0.90, suitability_reasoning="Good skills", matching_skills=["Python"]),
        JobEvaluation(job_id=2, suitability_score=0.85, suitability_reasoning="Good match", matching_skills=["Python"]),
    ]
    mock_response = MagicMock()
    mock_response.text = RerankResponse(evaluations=mock_evaluations).model_dump_json()

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    results = rerank_jobs(
        candidate_profile=fresher_profile,
        qualifying_jobs=jobs,
        candidate_level="fresher",
        gemini_client=mock_client,
    )

    # Job 2 (Intern) must be ranked #1
    assert results[0]["job_id"] == 2
    assert results[0]["job_title"] == "Python Developer Intern"
    assert results[0]["suitability_score"] == 0.85

    # Job 1 (Senior Architect) must be penalized below 0.30
    assert results[1]["job_id"] == 1
    assert results[1]["suitability_score"] <= 0.30
    assert any("Senior" in gap for gap in results[1]["experience_gaps"])


def test_rerank_experienced_strictly_penalizes_internships():
    """Verify that experienced candidate receives penalized scores on Internships."""
    senior_profile = {
        "name": "Marcus Vance",
        "target_role": "Senior Backend Architect",
        "skills": ["Python", "FastAPI", "Kubernetes", "AWS"],
        "experience": [{"title": "Senior Engineer", "dates": "2018 - Present", "description": "6 years exp"}],
    }

    jobs = [
        {"job_id": 1, "job_title": "Backend Engineering Intern", "company": "StartupX", "similarity_score": 0.80},
        {"job_id": 2, "job_title": "Senior Python Engineer", "company": "Enterprise", "similarity_score": 0.78},
    ]

    mock_evaluations = [
        JobEvaluation(job_id=1, suitability_score=0.85, suitability_reasoning="Has all skills", matching_skills=["Python"]),
        JobEvaluation(job_id=2, suitability_score=0.88, suitability_reasoning="Matches seniority", matching_skills=["Python", "FastAPI"]),
    ]
    mock_response = MagicMock()
    mock_response.text = RerankResponse(evaluations=mock_evaluations).model_dump_json()

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    results = rerank_jobs(
        candidate_profile=senior_profile,
        qualifying_jobs=jobs,
        candidate_level="senior",
        gemini_client=mock_client,
    )

    # Job 2 (Senior Engineer) must be ranked #1
    assert results[0]["job_id"] == 2
    assert results[0]["job_title"] == "Senior Python Engineer"
    assert results[0]["suitability_score"] == 0.88

    # Job 1 (Intern) must be penalized below 0.30 for senior
    assert results[1]["job_id"] == 1
    assert results[1]["suitability_score"] <= 0.30

