"""
LLM Reranking Module for SmartHire GenAI.

Evaluates qualifying jobs returned by FAISS vector retrieval against the
candidate's structured resume profile using Gemini.
Generates suitability scores, qualitative reasoning, matching skills,
missing skills, and experience gap analysis, then reranks jobs by suitability.
"""

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from google.genai import types
from pydantic import BaseModel, Field

from src.config import MODEL_NAME
from src.parsing.resume_parser import client

logger = logging.getLogger(__name__)

RERANK_CACHE_DIR = Path("data/.rerank_cache")
RERANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MEM_RERANK_CACHE: dict[str, dict] = {}


# ---------- Pydantic Schemas for Structured LLM Output ----------

class JobEvaluation(BaseModel):
    job_id: Union[int, str] = Field(
        description="The unique identifier of the job being evaluated."
    )
    suitability_score: float = Field(
        description="Holistic suitability score between 0.0 (completely unsuitable) and 1.0 (perfect fit) evaluated by the LLM."
    )
    suitability_reasoning: str = Field(
        description="Detailed, objective explanation of why this job matches or does not match the candidate."
    )
    matching_skills: List[str] = Field(
        default_factory=list,
        description="Candidate's demonstrated skills that directly satisfy the job requirements."
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Skills or competencies required or preferred by the job that the candidate lacks."
    )
    experience_gaps: List[str] = Field(
        default_factory=list,
        description="Gaps in seniority, years of experience, domain depth, or responsibilities between candidate and job."
    )


class RerankResponse(BaseModel):
    evaluations: List[JobEvaluation] = Field(
        default_factory=list,
        description="List of job evaluations for each provided job record."
    )


# ---------- Prompt Definition ----------

RERANK_SYSTEM_PROMPT = """You are an expert talent acquisition and technical hiring specialist evaluating job fit.
Your task is to evaluate a candidate's structured resume profile against a list of qualifying job descriptions retrieved via semantic search.

Evaluation Guidelines:
1. Holistic Assessment: Judge suitability on the complete picture—core technical competencies, hands-on tool usage, professional experience depth, project relevance, and education.
2. Evidence Over Claims: Prioritize demonstrated work (deployed software, production systems, verified employment) over bare keyword lists.
3. Strict Experience Level & Seniority Gating (CRITICAL RULE):
   - You MUST strictly enforce candidate experience level alignment:
   - If candidate is a FRESHER (0-1 year of professional experience, college student, or recent graduate):
     * ONLY match them with entry-level, fresher, junior, or internship postings.
     * Any job demanding 2+, 3+, 5+, or senior/lead/manager years of industry experience is a SEVERE SENIORITY MISMATCH.
     * You MUST grade any multi-year or senior role as completely unsuitable (suitability_score < 0.40) and record the exact mismatch in experience_gaps. Academic, classroom, or personal projects do NOT count as multi-year professional industry experience.
   - If candidate is EXPERIENCED (mid, senior, 3+ years professional experience):
     * Match them with roles that correspond to their verified industry seniority (Mid, Senior, Lead).
     * Internships and entry-level fresher roles are under-level mismatches and must be graded down (suitability_score < 0.45).
4. No Fixed Python Math: Do not apply rigid arithmetic weighting. Evaluate contextual fitness intelligently as a seasoned engineering hiring manager.
5. Skill & Gap Precision:
   - matching_skills: verified skills the candidate genuinely possesses that the job asks for.
   - missing_skills: required or strongly preferred technologies/skills the candidate does not show evidence of.
   - experience_gaps: explicit mismatches in years of experience, role seniority, or technical complexity.
6. Score Calibration:
   - 0.85 - 1.00: Outstanding fit; matches core stack, seniority level, and responsibilities closely.
   - 0.70 - 0.84: Strong fit; possesses core capabilities with minor non-blocking skill or experience gaps.
   - 0.50 - 0.69: Moderate / Stretch fit; transferable skills but notable gaps in critical stack or seniority.
   - 0.00 - 0.49: Poor fit; significant skill disconnect or major seniority/experience mismatch.
7. Semantic Similarity Reinforcement:
   - Each job record includes `initial_semantic_score` computed from dense embedding similarity.
   - High semantic similarity indicates strong technical, domain, and stack alignment with the candidate.
   - Your suitability score must reflect and reinforce this semantic alignment: a job with high semantic similarity that satisfies the candidate's seniority level and core competencies should receive a top suitability score (0.85 - 1.00), ranking above jobs with lower semantic overlap.

Return a JSON object matching the requested schema with an evaluation for every provided job.
"""


def _prepare_rerank_prompt_content(
    profile: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    candidate_level: Optional[str] = None,
) -> str:
    """Format candidate profile and qualifying jobs for the LLM prompt."""
    inferred_level = candidate_level or (
        "fresher" if not profile.get("experience") or len(profile.get("experience", [])) == 0 else "experienced"
    )

    profile_summary = {
        "candidate_name": profile.get("name"),
        "candidate_level": inferred_level,
        "target_role": profile.get("target_role"),
        "professional_summary": profile.get("professional_summary"),
        "skills": profile.get("skills", []),
        "experience": profile.get("experience", []),
        "projects": profile.get("projects", []),
        "education": profile.get("education", []),
        "certifications": profile.get("certifications", []),
    }

    jobs_summary = [
        {
            "job_id": job.get("job_id"),
            "job_title": job.get("job_title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "skills": job.get("skills"),
            "description": job.get("description"),
            "initial_semantic_score": job.get("similarity_score"),
        }
        for job in jobs
    ]

    return (
        f"### CANDIDATE PROFILE (Level: {inferred_level.upper()}) ###\n"
        f"{json.dumps(profile_summary, ensure_ascii=False, indent=2)}\n\n"
        f"### JOBS TO EVALUATE ({len(jobs)} jobs) ###\n"
        f"{json.dumps(jobs_summary, ensure_ascii=False, indent=2)}"
    )


def rerank_jobs(
    candidate_profile: Dict[str, Any],
    qualifying_jobs: List[Dict[str, Any]],
    candidate_level: Optional[str] = None,
    gemini_client: Optional[Any] = None,
    model_name: Optional[str] = None,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Rerank FAISS-retrieved qualifying jobs against a candidate's structured profile.

    Args:
        candidate_profile: Structured resume profile dict (from parse_resume).
        qualifying_jobs: List of job records returned by search_jobs.
        candidate_level: Optional candidate seniority level ('fresher', 'junior', 'mid', 'senior').
        gemini_client: Optional custom Gemini client (defaults to shared client).
        model_name: Optional Gemini model name override (defaults to MODEL_NAME).
        max_retries: Maximum number of retry attempts on transient API errors.


    Returns:
        List of reranked job dictionaries, each augmented with:
        'suitability_score', 'suitability_reasoning', 'matching_skills',
        'missing_skills', 'experience_gaps', and sorted by 'suitability_score' descending.

    Raises:
        ValueError: If candidate_profile is not a valid dict.
        RuntimeError: If the Gemini API call fails after retries.
    """
    if not isinstance(candidate_profile, dict):
        raise ValueError(
            f"candidate_profile must be a dictionary, got {type(candidate_profile).__name__}"
        )

    # Short-circuit on empty retrieval: Avoid calling the LLM unnecessarily
    if not qualifying_jobs:
        logger.info("[rerank_jobs] qualifying_jobs is empty; returning empty list without LLM call.")
        return []

    active_client = gemini_client or client
    active_model = model_name or MODEL_NAME

    user_content = _prepare_rerank_prompt_content(
        profile=candidate_profile,
        jobs=qualifying_jobs,
        candidate_level=candidate_level,
    )

    # Check cache when using default client (preserves mock clients in unit tests)
    use_cache = gemini_client is None
    cache_key = None
    rerank_output = None

    if use_cache:
        job_ids = [str(j.get("job_id")) for j in qualifying_jobs]
        cache_payload = {
            "candidate": candidate_profile,
            "level": candidate_level,
            "job_ids": sorted(job_ids),
        }
        cache_key = hashlib.md5(
            json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        if cache_key in _MEM_RERANK_CACHE:
            rerank_output = RerankResponse.model_validate(_MEM_RERANK_CACHE[cache_key])
        else:
            cache_file = RERANK_CACHE_DIR / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    cached_json = cache_file.read_text(encoding="utf-8", errors="replace")
                    if cached_json and cached_json.strip():
                        cached_data = json.loads(cached_json)
                        _MEM_RERANK_CACHE[cache_key] = cached_data
                        rerank_output = RerankResponse.model_validate(cached_data)
                except Exception as cache_err:
                    logger.warning("[rerank_jobs] Could not read rerank cache %s: %s", cache_file.name, cache_err)

    if rerank_output is None:
        delay = 1.0
        response = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = active_client.models.generate_content(
                    model=active_model,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=RERANK_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=RerankResponse,
                    ),
                )
                break
            except Exception as e:
                last_error = e
                code = getattr(e, "code", None)
                logger.warning(
                    f"[rerank_jobs] Attempt {attempt}/{max_retries} failed (Status: {code}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                else:
                    break

        if response is None:
            if last_error:
                raise RuntimeError(f"Gemini API failure during job reranking: {last_error}") from last_error
            raise RuntimeError("Failed to obtain response from Gemini API during job reranking.")

        response_text = getattr(response, "text", None)
        if not response_text or not response_text.strip():
            raise ValueError("Gemini returned an empty reranking response.")

        try:
            rerank_output = RerankResponse.model_validate_json(response_text)
        except Exception as parse_err:
            raise ValueError(
                f"Failed to parse LLM rerank output into RerankResponse schema: {parse_err}"
            ) from parse_err

        # Save to disk and memory cache
        if use_cache and cache_key:
            try:
                RERANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file = RERANK_CACHE_DIR / f"{cache_key}.json"
                cache_data = rerank_output.model_dump()
                cache_file.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8")
                _MEM_RERANK_CACHE[cache_key] = cache_data
            except Exception as save_err:
                logger.warning("[rerank_jobs] Could not write rerank cache: %s", save_err)

    # Map evaluations by job_id (handling string vs int keys safely)
    eval_map: Dict[str, JobEvaluation] = {
        str(ev.job_id): ev for ev in rerank_output.evaluations
    }

    # Merge evaluation into original job records
    reranked_results: List[Dict[str, Any]] = []

    # Determine effective candidate level for seniority safeguards
    eff_level = (
        candidate_level
        or ("fresher" if not candidate_profile.get("experience") or len(candidate_profile.get("experience", [])) == 0 else "experienced")
    ).lower()

    for job in qualifying_jobs:
        jid_str = str(job.get("job_id"))
        ev = eval_map.get(jid_str)

        merged_job = dict(job)
        if ev:
            merged_job["suitability_score"] = round(float(ev.suitability_score), 4)
            merged_job["suitability_reasoning"] = ev.suitability_reasoning.strip()
            merged_job["matching_skills"] = ev.matching_skills
            merged_job["missing_skills"] = ev.missing_skills
            merged_job["experience_gaps"] = ev.experience_gaps
        else:
            # Fallback for any job un-evaluated by the LLM
            logger.warning(f"[rerank_jobs] Job ID {jid_str} was not found in LLM evaluation output.")
            merged_job["suitability_score"] = float(job.get("similarity_score", 0.0))
            merged_job["suitability_reasoning"] = "Evaluated via semantic similarity baseline."
            merged_job["matching_skills"] = []
            merged_job["missing_skills"] = []
            merged_job["experience_gaps"] = []

        # Seniority Alignment Safeguard
        title_lower = merged_job.get("job_title", "").lower()
        if eff_level == "fresher":
            senior_keywords = ["senior", "lead", "architect", "principal", "staff", "manager", "head of", "director"]
            if any(k in title_lower for k in senior_keywords):
                merged_job["suitability_score"] = min(merged_job["suitability_score"], 0.25)
                gap_msg = "Senior title requires industry professional experience beyond fresher level."
                if gap_msg not in merged_job.get("experience_gaps", []):
                    merged_job.setdefault("experience_gaps", []).append(gap_msg)
        elif eff_level in ("mid", "senior"):
            if "intern" in title_lower or "internship" in title_lower:
                merged_job["suitability_score"] = min(merged_job["suitability_score"], 0.25)
                gap_msg = "Internship position is under-level for an experienced professional."
                if gap_msg not in merged_job.get("experience_gaps", []):
                    merged_job.setdefault("experience_gaps", []).append(gap_msg)

        reranked_results.append(merged_job)


    # Sort primarily by suitability_score descending, with FAISS similarity_score as secondary tie-breaker
    reranked_results.sort(
        key=lambda x: (
            round(float(x.get("suitability_score", 0.0)), 4),
            round(float(x.get("similarity_score", 0.0)), 4),
        ),
        reverse=True,
    )

    # Update rank to reflect the final LLM-guided ordering
    for idx, job_record in enumerate(reranked_results, start=1):
        job_record["rank"] = idx

    logger.info(
        f"[rerank_jobs] Successfully reranked {len(reranked_results)} jobs by LLM suitability."
    )
    return reranked_results
