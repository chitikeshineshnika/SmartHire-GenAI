"""
SmartHire Job Matcher Module.

Orchestrates the end-to-end job matching workflow:
1. Understands candidate profile & generates role-aligned search query.
2. Performs efficient semantic retrieval via FAISS vector store.
3. Applies relevance thresholding to exclude clearly irrelevant jobs.
4. Evaluates qualifying jobs against candidate credentials using Gemini LLM.
5. Returns final matches with suitability scores, reasoning, and skill/experience gap analysis
   without enforcing rigid job counts.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_TOP_N_JOBS, MIN_RETRIEVAL_RESULTS
from src.search.candidate_analysis import analyze_candidate
from src.search.embed import embed_text
from src.search.job_search import search_jobs
from src.search.rerank import rerank_jobs

logger = logging.getLogger(__name__)

DEFAULT_DATASET_LEVELS = ["fresher", "junior", "mid", "senior"]
DEFAULT_DATASET_LOCATIONS = [
    "Bengaluru", "Chennai", "Delhi", "Gurugram",
    "Hyderabad", "Kolkata", "Mumbai", "Noida", "Pune", "Remote"
]


def match_jobs_for_candidate(
    candidate_profile: Dict[str, Any],
    candidate_analysis: Optional[Dict[str, Any]] = None,
    similarity_threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD,
    max_retrieval_candidates: int = DEFAULT_TOP_N_JOBS,
    min_results: int = MIN_RETRIEVAL_RESULTS,
    min_suitability_threshold: Optional[float] = 0.50,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    rerank_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute the full SmartHire job-matching workflow for a candidate.

    Args:
        candidate_profile: Structured resume profile dict (from resume_parser.parse_resume).
        candidate_analysis: Pre-computed analysis dict containing 'search_text' and 'level' (optional).
        similarity_threshold: FAISS cosine similarity threshold (default: 0.75).
        max_retrieval_candidates: Upper limit for initial FAISS retrieval pool (default: 25).
        min_suitability_threshold: Minimum LLM suitability score for final matches (default: 0.50).
                                   Jobs with suitability below this are excluded.
        embed_fn: Optional embedding function override for dependency injection.
        search_fn: Optional FAISS search function override for dependency injection.
        rerank_fn: Optional LLM rerank function override for dependency injection.

    Returns:
        List of final matched job records, ordered from highest to lowest suitability,
        complete with suitability_score, suitability_reasoning, matching_skills,
        missing_skills, experience_gaps, and original job metadata.
    """
    if not isinstance(candidate_profile, dict):
        raise ValueError(
            f"candidate_profile must be a dictionary, got {type(candidate_profile).__name__}"
        )

    # ---------------------------------------------------------
    # Stage 1: Candidate Understanding & Search Query Generation
    # ---------------------------------------------------------
    analysis = candidate_analysis
    if not analysis or not analysis.get("search_text"):
        logger.info("[matcher] Candidate analysis not provided or incomplete; running analyze_candidate()...")
        analysis = analyze_candidate(
            profile=candidate_profile,
            allowed_levels=DEFAULT_DATASET_LEVELS,
            allowed_locations=DEFAULT_DATASET_LOCATIONS,
        )

    search_text = analysis.get("search_text", "").strip()
    if not search_text:
        # Fallback to summary / skills if search_text is unexpectedly empty
        skills_str = ", ".join(candidate_profile.get("skills", []))
        summary = candidate_profile.get("professional_summary") or candidate_profile.get("target_role") or ""
        search_text = f"{summary} {skills_str}".strip()
        if not search_text:
            raise ValueError("Candidate profile does not contain enough textual content for job search.")

    logger.info(f"[matcher] Generated candidate search text ({len(search_text.split())} words).")

    # ---------------------------------------------------------
    # Stage 2 & 3: FAISS Semantic Retrieval with Relevance Filtering
    # ---------------------------------------------------------
    embedder = embed_fn or embed_text
    searcher = search_fn or search_jobs

    logger.info("[matcher] Generating candidate embedding vector...")
    candidate_vector = embedder(search_text)

    logger.info(
        f"[matcher] Retrieving up to {max_retrieval_candidates} jobs from FAISS (threshold: {similarity_threshold}, soft floor min: {min_results})..."
    )
    qualifying_jobs = searcher(
        query=candidate_vector,
        top_k=max_retrieval_candidates,
        similarity_threshold=similarity_threshold,
        min_results=min_results,
    )

    logger.info(f"[matcher] FAISS retrieved {len(qualifying_jobs)} qualifying job(s) meeting similarity threshold.")

    # Short-circuit: If no jobs exist in store, return empty
    if not qualifying_jobs:
        logger.info("[matcher] No jobs returned from search. Returning empty matches without calling LLM.")
        return []

    # ---------------------------------------------------------
    # Stage 4: LLM Suitability Evaluation & Reranking
    # ---------------------------------------------------------
    reranker = rerank_fn or rerank_jobs
    cand_level = analysis.get("level") if analysis else None
    logger.info(
        f"[matcher] Passing {len(qualifying_jobs)} job(s) to Gemini LLM for holistic suitability evaluation (level: {cand_level})..."
    )
    try:
        evaluated_jobs = reranker(
            candidate_profile=candidate_profile,
            qualifying_jobs=qualifying_jobs,
            candidate_level=cand_level,
        )
    except TypeError:
        evaluated_jobs = reranker(
            candidate_profile=candidate_profile,
            qualifying_jobs=qualifying_jobs,
        )


    # ---------------------------------------------------------
    # Stage 5: Final Matches with Non-Rigid Counts & Soft Floor
    # ---------------------------------------------------------
    # Retain only genuinely suitable jobs (>= min_suitability_threshold)
    if min_suitability_threshold is not None:
        final_matches = [
            job for job in evaluated_jobs
            if job.get("suitability_score", 0.0) >= min_suitability_threshold
        ]
        # Soft floor: ensure at least min_results are returned if evaluated jobs exist
        if len(final_matches) < min_results and evaluated_jobs:
            final_matches = evaluated_jobs[:min(min_results, len(evaluated_jobs))]
        logger.info(
            f"[matcher] Filtered to {len(final_matches)} job(s) meeting suitability criteria."
        )
    else:
        final_matches = evaluated_jobs

    # Update final rank order for the qualifying set
    for rank_idx, match in enumerate(final_matches, start=1):
        match["rank"] = rank_idx

    logger.info(f"[matcher] Completed job matching workflow: returning {len(final_matches)} final match(es).")
    return final_matches
