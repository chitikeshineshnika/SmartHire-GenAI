"""
SmartHire AI Career Mentor module.
"""

from src.mentor.knowledge_base import (
    build_mentor_index,
    chunk_career_notes,
    chunk_job_descriptions,
    ingest_career_notes_pdf,
    load_career_notes,
    load_mentor_index,
    load_relevant_jobs,
    prepare_mentor_knowledge_base,
    retrieve_mentor_evidence,
)
from src.mentor.memory import ChatMessage, ConversationMemory
from src.mentor.rag_chain import (
    MENTOR_SYSTEM_PROMPT,
    UNSUPPORTED_FALLBACK_MESSAGE,
    MentorRetriever,
    ask_career_mentor,
    create_mentor_lcel_chain,
    format_citations_block,
    strip_source_citations,
    validate_and_align_citations,
)

__all__ = [
    "ChatMessage",
    "ConversationMemory",
    "MENTOR_SYSTEM_PROMPT",
    "MentorRetriever",
    "UNSUPPORTED_FALLBACK_MESSAGE",
    "ask_career_mentor",
    "build_mentor_index",
    "chunk_career_notes",
    "chunk_job_descriptions",
    "create_mentor_lcel_chain",
    "format_citations_block",
    "ingest_career_notes_pdf",
    "load_career_notes",
    "load_mentor_index",
    "load_relevant_jobs",
    "prepare_mentor_knowledge_base",
    "retrieve_mentor_evidence",
    "strip_source_citations",
    "validate_and_align_citations",
]
