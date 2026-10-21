"""
Knowledge base ingestion, chunking, embedding, and indexing for SmartHire AI Career Mentor.

Handles:
1. Loading all career notes (19 roles) from data/career_notes/*.md.
2. Loading representative job postings from data/jobs/cleaned_jobs.csv.
3. Cleaning and chunking with context preservation (headings, roles, job metadata).
4. Generating embeddings and building a persisted FAISS index (IndexFlatIP) with rich source metadata.
5. Providing evidence retrieval for RAG-grounded career guidance.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CAREER_NOTES_DIR,
    CLEANED_JOBS_CSV,
    EMBED_BATCH_SIZE,
    EMBED_SLEEP_SECONDS,
    MENTOR_EMBEDDINGS_CACHE,
    MENTOR_INDEX_PATH,
    MENTOR_METADATA_PATH,
    MENTOR_VECTORSTORE_DIR,
    VECTORSTORE_DIR,
)
from src.search.embed import embed_text, embed_texts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Document Loading and Cleaning
# ---------------------------------------------------------------------------

def extract_role_title_from_markdown(content: str, fallback: str = "Career Note") -> str:
    """Extract the main H1 role title from markdown content."""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_career_notes(notes_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load all career guidance notes from markdown files.
    
    Returns:
        List of dicts: [{'source': filename, 'title': role_title, 'content': text}, ...]
    """
    cdir = Path(notes_dir or CAREER_NOTES_DIR)
    if not cdir.exists():
        raise FileNotFoundError(f"Career notes directory does not exist: {cdir}")

    note_files = sorted(cdir.glob("*.md"))
    if not note_files:
        raise ValueError(f"No markdown career notes found in {cdir}")

    documents = []
    for filepath in note_files:
        try:
            content = filepath.read_text(encoding="utf-8").strip()
            if not content:
                continue
            title = extract_role_title_from_markdown(content, fallback=filepath.stem.replace("_", " ").title())
            documents.append({
                "source": filepath.name,
                "title": title,
                "content": content,
            })
        except Exception as e:
            logger.warning(f"Error loading career note {filepath.name}: {e}")

    logger.info(f"Loaded {len(documents)} career guidance notes from {cdir}")
    return documents


def load_relevant_jobs(
    jobs_csv_path: Optional[Path] = None,
    max_per_title: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load job descriptions from jobs_metadata.csv or cleaned_jobs.csv.
    Connects all job entries that have been used for semantic search.
    
    Returns:
        List of job records with standard keys: job_id, job_title, company, location, skills, description.
    """
    default_path = VECTORSTORE_DIR / "jobs_metadata.csv"
    if not default_path.exists():
        default_path = CLEANED_JOBS_CSV
    path = Path(jobs_csv_path or default_path)
    if not path.exists():
        raise FileNotFoundError(f"Jobs CSV not found: {path}")

    df = pd.read_csv(path)
    
    # Ensure expected columns
    required_cols = ["job_id", "job_title", "company", "location", "skills", "description"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Drop entries with empty descriptions
    df = df[df["description"].str.len() >= 30]

    # Sample up to max_per_title per job title if requested, else connect all
    if max_per_title is not None and max_per_title > 0:
        sampled_df = df.groupby("job_title", as_index=False).head(max_per_title).reset_index(drop=True)
    else:
        sampled_df = df.reset_index(drop=True)

    job_records = sampled_df.to_dict(orient="records")
    logger.info(f"Loaded {len(job_records)} job records for mentor knowledge base from {path}")
    return job_records


# ---------------------------------------------------------------------------
# 2. Chunking with Context Preservation and Rich Metadata
# ---------------------------------------------------------------------------

def chunk_career_notes(
    notes: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Chunk markdown career notes while preserving section context and headings.
    
    Prepends contextual role and section information to each chunk:
    "[Career Note: {title} | Section: {section_title}]\n..."
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_chunks: List[Dict[str, Any]] = []

    for note in notes:
        filename = note["source"]
        title = note["title"]
        content = note["content"]

        # Split content into major sections by markdown headers (## or ###)
        # Regex captures headings like '## Role guide' or '### 1. Prerequisites'
        section_pattern = re.compile(r"(^#{2,3}\s+[^\n]+)", re.MULTILINE)
        parts = section_pattern.split(content)

        sections: List[Tuple[str, str]] = []
        current_section = "Overview"

        if parts and not parts[0].startswith("##"):
            preamble = parts[0].strip()
            if preamble:
                sections.append((current_section, preamble))
            parts = parts[1:]

        for i in range(0, len(parts), 2):
            if i < len(parts):
                sec_header = parts[i].strip().lstrip("#").strip()
                sec_body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                current_section = sec_header
                if sec_body:
                    sections.append((current_section, sec_body))

        # If no markdown headings were detected, treat whole content as one section
        if not sections:
            sections = [("Overview", content)]

        chunk_counter = 0
        for sec_name, sec_text in sections:
            split_texts = splitter.split_text(sec_text)
            for text_chunk in split_texts:
                cleaned_text = text_chunk.strip()
                if len(cleaned_text) < 30:
                    continue

                # Add context prefix to preserve semantic anchor during retrieval
                contextual_text = f"[Career Note: {title} | Section: {sec_name}]\n{cleaned_text}"
                chunk_id = f"note_{Path(filename).stem}_chunk_{chunk_counter}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": contextual_text,
                    "source_type": "career_note",
                    "source": filename,
                    "title": title,
                    "section": sec_name,
                    "job_id": None,
                    "company": None,
                    "location": None,
                    "skills": None,
                })
                chunk_counter += 1

    logger.info(f"Created {len(all_chunks)} chunks from {len(notes)} career notes.")
    return all_chunks


def chunk_job_descriptions(
    jobs: List[Dict[str, Any]],
    chunk_size: int = 850,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Format and chunk job postings with complete metadata.
    
    Each job is formatted as:
    "[Job Posting: {title} at {company} | Location: {location} | Job ID: {id}]
     Skills: {skills}
     Description: {description}"
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_chunks: List[Dict[str, Any]] = []

    for job in jobs:
        job_id = int(job.get("job_id", 0))
        title = job.get("job_title", "").strip()
        company = job.get("company", "").strip()
        location = job.get("location", "").strip()
        skills = job.get("skills", "").strip()
        desc = job.get("description", "").strip()

        source_label = f"Job Posting #{job_id}: {title} at {company}"
        header = f"[Job Posting: {title} at {company} | Location: {location} | Job ID: {job_id}]"
        body = f"Skills: {skills}\nJob Description: {desc}"
        full_text = f"{header}\n{body}"

        # If job posting is within size limit, keep it as single intact chunk
        if len(full_text) <= chunk_size:
            chunk_texts = [full_text]
        else:
            sub_chunks = splitter.split_text(body)
            chunk_texts = [f"{header}\n{sc}" for sc in sub_chunks]

        for idx, text_chunk in enumerate(chunk_texts):
            chunk_id = f"job_{job_id}_chunk_{idx}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": text_chunk.strip(),
                "source_type": "job_description",
                "source": source_label,
                "title": title,
                "section": "Job Requirements",
                "job_id": job_id,
                "company": company,
                "location": location,
                "skills": skills,
            })

    logger.info(f"Created {len(all_chunks)} chunks from {len(jobs)} job postings.")
    return all_chunks


def prepare_mentor_knowledge_base() -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Load, clean, and chunk all career notes and representative job postings.
    
    Returns:
        texts: List of text strings ready for embedding.
        metadata: List of metadata dicts corresponding 1-to-1 with texts.
    """
    notes = load_career_notes()
    note_chunks = chunk_career_notes(notes)

    jobs = load_relevant_jobs()
    job_chunks = chunk_job_descriptions(jobs)

    all_chunks = note_chunks + job_chunks
    texts = [c["text"] for c in all_chunks]
    
    # Metadata dictionary without the redundant 'text' field to save disk space
    metadata = []
    for c in all_chunks:
        meta = {k: v for k, v in c.items() if k != "text"}
        # Keep snippet or text in metadata for exact retrieval reconstruction
        meta["text"] = c["text"]
        metadata.append(meta)

    logger.info(f"Prepared total {len(texts)} chunks ({len(note_chunks)} notes, {len(job_chunks)} jobs).")
    return texts, metadata


# ---------------------------------------------------------------------------
# 3. Vector Normalization, Indexing, and Persistence
# ---------------------------------------------------------------------------

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity search with IndexFlatIP."""
    v_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v_f32, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v_f32 / norms


def build_mentor_index(
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, np.ndarray, List[Dict[str, Any]]]:
    """
    Build or load the FAISS vector index for the AI Career Mentor.
    
    Persists:
    - vectorstore/mentor/mentor.faiss
    - vectorstore/mentor/mentor_metadata.json
    - vectorstore/mentor/mentor_embeddings.npy
    """
    MENTOR_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    if (
        not force_rebuild
        and MENTOR_INDEX_PATH.exists()
        and MENTOR_METADATA_PATH.exists()
        and MENTOR_EMBEDDINGS_CACHE.exists()
    ):
        logger.info(f"Loading existing mentor index from {MENTOR_VECTORSTORE_DIR}...")
        index = faiss.read_index(str(MENTOR_INDEX_PATH))
        embeddings = np.load(str(MENTOR_EMBEDDINGS_CACHE))
        with open(MENTOR_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, embeddings, metadata

    logger.info("Building new mentor FAISS index from scratch...")
    texts, metadata = prepare_mentor_knowledge_base()

    if not texts:
        raise ValueError("No documents or chunks available to embed.")

    logger.info(f"Generating embeddings for {len(texts)} chunks using Gemini...")
    raw_embeddings = embed_texts(texts, batch_size=EMBED_BATCH_SIZE, sleep_seconds=EMBED_SLEEP_SECONDS)
    
    embeddings = np.array(raw_embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    # Normalize vectors for cosine similarity
    normed_vectors = normalize_vectors(embeddings)

    # Build IndexFlatIP
    index = faiss.IndexFlatIP(dim)
    index.add(normed_vectors)

    # Persist artifacts
    logger.info(f"Persisting mentor index to {MENTOR_INDEX_PATH}...")
    faiss.write_index(index, str(MENTOR_INDEX_PATH))
    np.save(str(MENTOR_EMBEDDINGS_CACHE), embeddings)
    with open(MENTOR_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully built and persisted mentor index with {index.ntotal} vectors.")
    return index, embeddings, metadata


def load_mentor_index() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load the persisted mentor FAISS index and metadata."""
    if not MENTOR_INDEX_PATH.exists() or not MENTOR_METADATA_PATH.exists():
        index, _, metadata = build_mentor_index(force_rebuild=False)
        return index, metadata

    index = faiss.read_index(str(MENTOR_INDEX_PATH))
    with open(MENTOR_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def ingest_career_notes_pdf(
    pdf_path: Optional[Union[str, Path]] = None,
) -> Tuple[int, int]:
    """
    Ingest, chunk, embed, and index new career notes from Career Notes 2.pdf
    into the AI Career Mentor FAISS vector store.
    Preserves all existing indexed work and appends only new unique chunks.

    Returns:
        Tuple of (new_chunks_added, total_vectors_in_index)
    """
    from src.parsing.loader import load_document

    target_pdf = Path(pdf_path or CAREER_NOTES_DIR / "Career Notes 2.pdf")
    if not target_pdf.exists():
        alt_pdf = Path("data") / "career_notes 2.pdf"
        if alt_pdf.exists():
            target_pdf = alt_pdf
        else:
            raise FileNotFoundError(f"Career notes PDF not found: {target_pdf}")

    logger.info(f"Extracting career notes from {target_pdf}...")
    doc = load_document(target_pdf)
    text = doc["text"]

    # Locate TOC boundary if present
    toc_pos = text.find("1. tech_interview_preparation.md\n")
    if toc_pos == -1:
        toc_pos = text.find("1. tech_interview_prep")
    body = text[toc_pos:] if toc_pos != -1 else text

    pattern = re.compile(r'(?:\A|\n)(?=(\d+)\.\s+([a-zA-Z0-9_\-]+\.md))')
    parts = pattern.split(body)

    notes = []
    for i in range(1, len(parts), 3):
        filename = parts[i+1].strip()
        content = parts[i+2].strip()
        lines = content.split('\n')
        if lines[0].strip().endswith('.md'):
            content = '\n'.join(lines[1:]).strip()
        title = extract_role_title_from_markdown(
            content,
            fallback=filename.replace('.md', '').replace('_', ' ').title()
        )
        notes.append({'source': filename, 'title': title, 'content': content})

    if not notes:
        notes.append({
            'source': target_pdf.name,
            'title': "Career Notes 2",
            'content': text
        })

    # Chunk the notes
    all_chunks = chunk_career_notes(notes)

    # Load existing mentor index, embeddings, and metadata
    index, embeddings, metadata = build_mentor_index(force_rebuild=False)
    existing_ids = {m.get("chunk_id") for m in metadata}

    # Filter to only new chunks not yet embedded
    new_chunks = [c for c in all_chunks if c.get("chunk_id") not in existing_ids]

    if not new_chunks:
        logger.info(f"All {len(all_chunks)} chunks from {target_pdf.name} are already indexed.")
        return 0, index.ntotal

    logger.info(f"Found {len(new_chunks)} new chunks to embed and add to Career Mentor.")
    new_texts = [c["text"] for c in new_chunks]

    # Generate embeddings for only new chunks
    raw_new_embeddings = embed_texts(
        new_texts,
        batch_size=EMBED_BATCH_SIZE,
        sleep_seconds=EMBED_SLEEP_SECONDS
    )
    new_embed_arr = np.array(raw_new_embeddings, dtype=np.float32)

    # L2-normalize new embeddings and add to FAISS index
    normed_new = normalize_vectors(new_embed_arr)
    index.add(normed_new)

    # Combine with existing cached embeddings and metadata
    combined_embeddings = np.vstack([embeddings, new_embed_arr])
    for c in new_chunks:
        meta = {k: v for k, v in c.items() if k != "text"}
        meta["text"] = c["text"]
        metadata.append(meta)

    # Persist updated artifacts
    faiss.write_index(index, str(MENTOR_INDEX_PATH))
    np.save(str(MENTOR_EMBEDDINGS_CACHE), combined_embeddings)
    with open(MENTOR_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Successfully embedded {len(new_chunks)} new chunks. Total indexed vectors: {index.ntotal}"
    )
    return len(new_chunks), index.ntotal


# ---------------------------------------------------------------------------
# 4. Evidence Retrieval
# ---------------------------------------------------------------------------

def _format_evidence_item(score: float, item: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to structure evidence dict consistently."""
    return {
        "score": float(score),
        "text": item.get("text", ""),
        "source_type": item.get("source_type"),
        "source": item.get("source"),
        "title": item.get("title"),
        "section": item.get("section"),
        "job_id": item.get("job_id"),
        "company": item.get("company"),
        "location": item.get("location"),
        "skills": item.get("skills"),
        "chunk_id": item.get("chunk_id"),
    }


def retrieve_mentor_evidence(
    query: str,
    top_k: int = 6,
    min_score: float = 0.35,
    source_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant grounded evidence chunks for a user career question.
    Ensures balanced multi-company job postings and prevents single document starvation.
    
    Args:
        query: User's career question or topic.
        top_k: Number of chunks to retrieve.
        min_score: Minimum cosine similarity score threshold.
        source_type_filter: Optional filter ('career_note' or 'job_description').
        
    Returns:
        List of evidence dicts containing score, text, and source metadata.
    """
    if not query or not query.strip():
        return []

    index, metadata = load_mentor_index()
    if index.ntotal == 0 or not metadata:
        return []

    # Embed query
    query_vector = embed_text(query.strip())
    q_arr = np.array([query_vector], dtype=np.float32)
    q_normed = normalize_vectors(q_arr)

    # Search a wider candidate pool to allow source diversity and company sampling
    fetch_k = min(max(top_k * 6, 40), index.ntotal)
    scores, indices = index.search(q_normed, fetch_k)

    # Check if user specifically asks about job postings/descriptions/dataset
    q_lower = query.lower()
    asks_for_jobs = any(
        term in q_lower
        for term in [
            "job description", "job posting", "job postings", "job dataset",
            "retrieved job", "jobs available", "in your job", "in the job",
            "sample job", "job requirements", "jobs require", "postings"
        ]
    )

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        if float(score) < min_score:
            continue
        item = metadata[idx]
        if source_type_filter and item.get("source_type") != source_type_filter:
            continue
        candidates.append((float(score), item))

    if not candidates:
        return []

    results = []
    source_counts: Dict[str, int] = {}
    seen_companies = set()

    # Dual-source retrieval: search both job corpus and career notes for every relevant question
    if not source_type_filter:
        job_candidates = [c for c in candidates if c[1].get("source_type") == "job_description"]
        note_candidates = [c for c in candidates if c[1].get("source_type") == "career_note"]

        # If user explicitly asked for job descriptions/postings, skew towards jobs:
        if asks_for_jobs:
            target_jobs = max(3, min(len(job_candidates), top_k - 2))
            target_notes = max(1, min(len(note_candidates), top_k - target_jobs))
        else:
            # Balanced dual-source target: job postings for market requirements + career notes for learning guidance
            target_notes = max(2, min(len(note_candidates), top_k // 2))
            target_jobs = max(2, min(len(job_candidates), top_k - target_notes))

        # 1. Retrieve job postings from distinct companies
        for score, item in job_candidates:
            comp = item.get("company")
            if comp and comp in seen_companies and len(job_candidates) > target_jobs:
                continue
            if comp:
                seen_companies.add(comp)
            results.append(_format_evidence_item(score, item))
            if len([r for r in results if r["source_type"] == "job_description"]) >= target_jobs:
                break

        # 2. Retrieve career notes (max 2 chunks per note to ensure broad guidance)
        for score, item in note_candidates:
            src = item.get("source") or item.get("title") or "note"
            if source_counts.get(src, 0) >= 2 and len(note_candidates) > target_notes:
                continue
            source_counts[src] = source_counts.get(src, 0) + 1
            results.append(_format_evidence_item(score, item))
            if len([r for r in results if r["source_type"] == "career_note"]) >= target_notes:
                break

        # 3. Fill any remaining slots up to top_k from all remaining candidates
        if len(results) < top_k:
            existing_ids = {r["chunk_id"] for r in results}
            for score, item in candidates:
                if item.get("chunk_id") not in existing_ids:
                    results.append(_format_evidence_item(score, item))
                    if len(results) >= top_k:
                        break

    else:
        # Filtered retrieval when source_type_filter is explicitly specified
        for score, item in candidates:
            src = item.get("source") or item.get("title") or "doc"
            if source_counts.get(src, 0) >= 3 and len(candidates) > top_k:
                continue
            source_counts[src] = source_counts.get(src, 0) + 1
            results.append(_format_evidence_item(score, item))
            if len(results) >= top_k:
                break

    return results
