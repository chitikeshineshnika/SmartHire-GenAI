"""
SmartHire AI Career Mentor — LangChain RAG Chain & Conversational Guidance Engine.

Orchestrates the complete Career Mentor RAG workflow using LangChain Expression Language (LCEL):
1. Input Guardrails: Blocks unsafe/adversarial attacks and off-topic requests at application level.
2. MentorRetriever: Custom LangChain BaseRetriever querying the FAISS vector index with relevance thresholds.
3. LangChain Document & ChatPromptTemplate: Structures retrieved context, system prompt, and chat history.
4. Gemini API Integration: Generates factual, grounded responses via Google GenAI SDK with low temperature.
5. LangChain StrOutputParser: Parses and streams clean text.
6. Citation Validation: Enforces exact source metadata from retrieved LangChain Documents.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from google.genai import errors, types
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import Field

from src.config import MODEL_NAME
from src.mentor.knowledge_base import retrieve_mentor_evidence
from src.parsing.resume_parser import client
from src.safety.guardrails import (
    MENTOR_OFF_TOPIC_REJECTION_MESSAGE,
    MENTOR_UNSAFE_REJECTION_MESSAGE,
    validate_mentor_input,
)

logger = logging.getLogger(__name__)

UNSUPPORTED_FALLBACK_MESSAGE = (
    "I don't know based on the available documents. "
    "Please provide a relevant document or ask a question covered by the available career notes or job dataset."
)

MENTOR_SYSTEM_PROMPT = """You are the SmartHire AI Career Mentor, an expert career advisor integrated into the SmartHire GenAI platform.

Your purpose is to provide students, freshers, and experienced candidates with sharp, highly intelligent, personalized, and practical career guidance strictly grounded in the verified knowledge base and job dataset.

CORE CAPABILITIES & PILLARS:
1. DUAL-SOURCE SYNTHESIS:
   - You receive dual-source evidence: verified job postings from the market corpus and learning guidance from career notes.
   - Use job postings to identify actual, live job-market requirements, in-demand tool stacks, and company expectations.
   - Use career notes to guide structured learning progression, prerequisites, foundational concepts, and project ideas.
   - Do NOT display, quote, or dump raw evidence blocks or citations; retrieve answers strictly from the evidence and present them naturally.

2. CANDIDATE PERSONALIZATION & EXPERIENCE LEVEL:
   - When candidate profile context (resume & matched jobs) is provided, personalize your guidance directly to the candidate's background, target role, experience level, and matched jobs. Address their specific career trajectory rather than giving generic advice.

3. SKILL GAP COMPARISON (DEMONSTRATED VS. NOT DEMONSTRATED):
   - When comparing candidate skills with job requirements:
     * Identify which skills the candidate already demonstrates through listed skills, projects, or work experience.
     * Identify which job requirements are not evident in the candidate's resume.
     * CRITICAL: Treat missing resume evidence strictly as "not demonstrated", rather than proof that the candidate completely lacks the skill or ability to learn.
     * Prioritize which missing skills the candidate should acquire first based on market demand across the retrieved job postings.

4. ACTIONABLE LEARNING ROADMAPS & RECOMMENDATIONS:
   - NEVER give superficial or generic advice like "learn SQL" or "study Python". Explain why a recommendation fits the user's situation and prioritize practical next steps.
   - Suggest a concrete, structured sequence grounded in career notes and job requirements (Foundations, Practical Exercises, Relevant Projects, and Interview Preparation).

5. STRICT GROUNDING & RELIABLE REFUSAL:
   - Base all facts, requirements, tools, and roadmaps strictly on the evidence provided to you. Never invent facts, job requirements, salaries, or citations.
   - If the available evidence is insufficient to answer the question, or if the question is outside career mentoring scope, state:
      "I don't know based on the available documents. Please provide a relevant document or ask a question covered by the available career notes or job dataset."

6. CONVERSATION CONTINUITY & HISTORY RECALL:
   - You receive the full conversation history of the current session as chat messages. This is your memory of everything discussed so far.
   - When the user explicitly asks about a PREVIOUS question, BEFORE conversation, EARLIER discussion, PAST answer, or anything they or you said earlier in this session, answer DIRECTLY and ACCURATELY from the conversation history. Do NOT say "I don't know based on the available documents" for these questions — the answer is in the conversation history you already have.
   - For all other career queries (roadmaps, skills, jobs, interview prep, etc.), continue using retrieved documents as your primary source exactly as before. Conversation history is secondary context for these queries.
   - If a question about prior conversation also relates to career topics covered in the retrieved evidence, use BOTH the conversation history AND the retrieved documents together to give a comprehensive answer.
   - Maintain natural conversational flow: if the user refers to something discussed earlier (e.g., "the roadmap you gave", "what you said about Python", "my first question"), recall it from the conversation history and respond naturally.

RESPONSE STYLE & FORMATTING RULES:
- **Adapt the format to the question**:
  * Use numbered steps for how-to questions.
  * Use clean bullet points for lists.
  * Use structured comparisons or markdown tables for alternatives.
  * Use code/workflow examples for programming questions.
  * Use personalized roadmaps for career-planning questions.
  * Do NOT force a rigid or repetitive template. Make every response feel thoughtful, context-aware, and naturally structured for that specific question.
- **Answer directly**: Put the most useful information first; avoid unnecessary introductory fluff, conversational throat-clearing, and repetition.
- **Vary the response length**: Keep simple questions brief and punchy; provide deeper, structured explanations for complex queries.
- **ZERO META-TALK & NO RAW EVIDENCE DISPLAY**: NEVER mention background retrieval processes, documents, chunk numbers, or knowledge base boundaries. Do NOT start answers with robotic phrases such as "Based on the retrieved documents...", "According to document [1]...", "From the provided notes...", or "Looking at the context...". Deliver authoritative, question-relevant career advice grounded strictly in the provided content without displaying the raw evidence.
"""


# ===========================================================================
# 1. LANGCHAIN RETRIEVER IMPLEMENTATION
# ===========================================================================

class MentorRetriever(BaseRetriever):
    """
    LangChain BaseRetriever querying the SmartHire FAISS mentor knowledge base.
    
    Converts raw chunk dictionaries into LangChain Document instances with
    complete source metadata (source, title, section, job_id, company).
    """
    top_k: int = 6
    min_score: float = 0.35

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> List[Document]:
        evidence = retrieve_mentor_evidence(query, top_k=self.top_k, min_score=self.min_score)
        documents = []
        for e in evidence:
            doc = Document(
                page_content=e.get("text", ""),
                metadata={
                    "score": e.get("score"),
                    "source": e.get("source"),
                    "source_type": e.get("source_type"),
                    "title": e.get("title"),
                    "section": e.get("section"),
                    "job_id": e.get("job_id"),
                    "company": e.get("company"),
                    "skills": e.get("skills"),
                    "chunk_id": e.get("chunk_id"),
                },
            )
            documents.append(doc)
        return documents


# ===========================================================================
# 2. CITATION ALIGNMENT & PROMPT HELPERS
# ===========================================================================

def format_citations_block(sources: List[Dict[str, Any]]) -> str:
    """Construct a validated Sources markdown block from actual retrieved evidence."""
    if not sources:
        return ""

    seen = set()
    lines = ["\n\n**Sources**"]
    for src in sources:
        label = src.get("source") or src.get("title") or ""
        if label and label not in seen:
            seen.add(label)
            lines.append(f"- {label}")

    return "\n".join(lines)


def validate_and_align_citations(
    response_text: str,
    retrieved_sources: List[Dict[str, Any]],
) -> str:
    """Ensure response citations match application metadata and remove hallucinations."""
    if not retrieved_sources:
        return re.sub(r"(?i)\*\*Sources\*\*[\s\S]*$", "", response_text).strip()

    if "i don't know based on the available documents" in response_text.lower():
        return re.sub(r"(?i)\*\*Sources\*\*[\s\S]*$", "", response_text).strip()

    validated_block = format_citations_block(retrieved_sources)
    sources_match = re.search(r"(?i)\*\*Sources\*\*[\s\S]*$", response_text)
    if sources_match:
        cleaned_text = response_text[:sources_match.start()].strip()
        return f"{cleaned_text}{validated_block}"
    else:
        return f"{response_text.strip()}{validated_block}"


def strip_source_citations(text: str) -> str:
    """Remove any Sources blocks, document citations, or file names from mentor answer."""
    if not text:
        return ""
    # 1. Split on any line matching 'Sources', '**Sources**', 'Sources:', 'References'
    parts = re.split(r"(?im)^\s*(?:\*{0,2}sources\*{0,2}|\*{0,2}references\*{0,2})\s*:?\s*$", text)
    cleaned = parts[0].strip()

    # 2. Also strip any lines containing markdown filenames (e.g. backend_developer.md)
    lines = []
    for line in cleaned.splitlines():
        if re.search(r"\b[a-z0-9_]+\.md\b", line, re.IGNORECASE):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def format_langchain_documents(docs: List[Document]) -> str:
    """Format LangChain Document objects into clean text for prompt ingestion."""
    blocks = []
    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata
        stype = meta.get("source_type", "document")
        source = meta.get("source", "unknown")
        title = meta.get("title", "")
        section = meta.get("section", "Content")

        header = f"--- RETRIEVED DOCUMENT [{idx}] (Source: {source} | Type: {stype} | Title: {title} | Section: {section}) ---"
        blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(blocks)


def build_evidence_context(evidence_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved evidence chunks for context string representation."""
    blocks = []
    for idx, e in enumerate(evidence_chunks, 1):
        stype = e.get("source_type", "document")
        source = e.get("source", "unknown")
        title = e.get("title", "")
        section = e.get("section", "Content")
        text = e.get("text", "")
        header = f"--- RETRIEVED DOCUMENT [{idx}] (Source: {source} | Type: {stype} | Title: {title} | Section: {section}) ---"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_conversation_history_context(history: Optional[List[Dict[str, str]]]) -> str:
    """Format session conversation history into context string representation."""
    if not history:
        return ""
    lines = ["CONVERSATION HISTORY (Current Session):"]
    for msg in history[-6:]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def contextualize_retrieval_query(
    query: str,
    history: Optional[List[Dict[str, str]]],
    selected_target_job: Optional[Dict[str, Any]] = None,
) -> str:
    """Enrich relative follow-up queries with previous conversation context and selected job details for FAISS retrieval."""
    enriched_query = query
    q_lower = query.lower()

    # 1. Company / selected target job resolution
    if selected_target_job:
        comp = str(selected_target_job.get("company") or "").strip()
        role = str(selected_target_job.get("job_title") or selected_target_job.get("title") or "").strip()
        skills = str(selected_target_job.get("skills") or "").strip()

        company_terms = [
            "this company", "that company", "the company", "this org",
            "this organization", "this employer", "here", "hired here", "get job here", "get a job here",
            "this job", "that job", "this role", "that role", "the job", "the role",
        ]
        if any(term in q_lower for term in company_terms) and (comp or role):
            job_context = f"{role} at {comp}".strip() if comp else role
            if skills:
                job_context = f"{job_context} {skills}"
            enriched_query = f"{job_context} {enriched_query}"

    # 2. Conversation history resolution for relative follow-up queries
    if history:
        referential_terms = [
            "this role", "that role", "this job", "that job", "this position",
            "that position", "the first role", "the second role", "this career",
            "for it", "about it", "with it", "for this", "about this", "level 1", "level 2", "level 3",
            "this company", "that company", "the company", "before question", "previous question",
            "earlier question", "what i asked", "what you said", "you mentioned", "that point",
            "only skills", "skills to learn", "what skills", "which skills", "roadmap",
            "more details", "elaborate", "clarify", "explain", "tell me more",
        ]
        has_reference = any(term in q_lower for term in referential_terms)
        is_short = len(query.split()) < 10

        followup_words = [
            "only", "also", "and", "more", "why", "how", "what about", "which",
            "mention", "list", "first", "second", "third", "step", "steps",
            "skills", "skill", "learn", "salary", "prep", "prepare",
            "before", "previous", "earlier", "last", "prior", "again", "repeat",
        ]
        has_followup_word = any(w in q_lower for w in followup_words)

        if has_reference or is_short or has_followup_word:
            for msg in reversed(history):
                if msg.get("role") == "user":
                    last_content = msg.get("content", "").strip()
                    if last_content and last_content.lower() not in enriched_query.lower():
                        enriched_query = f"{last_content} {enriched_query}"
                        break

    return enriched_query


def build_candidate_context(
    candidate_profile: Optional[Dict[str, Any]] = None,
    matched_jobs: Optional[List[Dict[str, Any]]] = None,
    selected_target_job: Optional[Dict[str, Any]] = None,
) -> str:
    """Format candidate profile, active target job, and matched jobs for deep mentor personalization."""
    lines = []

    # Active Target Job Chosen by Candidate
    if selected_target_job:
        t_comp = selected_target_job.get("company") or "Selected Company"
        t_role = selected_target_job.get("job_title") or selected_target_job.get("title") or "Selected Target Role"
        t_loc = selected_target_job.get("location") or "Location not specified"
        t_skills = selected_target_job.get("skills") or ""
        t_level = selected_target_job.get("level") or ""
        t_exp = selected_target_job.get("experience") or ""
        t_desc = selected_target_job.get("job_description") or selected_target_job.get("description") or ""

        lines.append("ACTIVE TARGET JOB SELECTED BY CANDIDATE (PRIMARY FOCUS):")
        lines.append(f"- Target Company: {t_comp}")
        lines.append(f"- Target Role Title: {t_role}")
        lines.append(f"- Location: {t_loc}")
        if t_level or t_exp:
            lines.append(f"- Required Seniority / Experience: {t_level or t_exp}")
        if t_skills:
            lines.append(f"- Required Technical Skills: {t_skills}")
        if t_desc:
            lines.append(f"- Job Description Snippet: {t_desc[:350]}")
        lines.append("")

    if candidate_profile:
        lines.append("CANDIDATE RESUME PROFILE (PERSONALIZATION & SKILL GAP CONTEXT):")
        name = candidate_profile.get("name") or "Candidate"
        target_role = candidate_profile.get("target_role") or "Target role not specified"
        lines.append(f"- Name: {name}")
        lines.append(f"- Target Role: {target_role}")

        # Skills demonstrated in resume
        tech_skills = candidate_profile.get("technical_skills") or candidate_profile.get("skills") or []
        if tech_skills:
            lines.append(f"- Demonstrated Technical Skills in Resume: {', '.join(str(s) for s in tech_skills)}")

        soft_skills = candidate_profile.get("soft_skills") or []
        if soft_skills:
            lines.append(f"- Soft Skills: {', '.join(str(s) for s in soft_skills)}")

        # Summary
        summary = candidate_profile.get("professional_summary")
        if summary:
            lines.append(f"- Professional Summary: {summary}")

        # Experience
        exp_list = candidate_profile.get("work_experience") or candidate_profile.get("experience") or []
        if exp_list:
            lines.append("- Work Experience & Internships:")
            for exp in exp_list[:3]:
                if isinstance(exp, dict):
                    title = exp.get("title") or "Role"
                    company = exp.get("company") or "Company"
                    techs = exp.get("technologies") or []
                    t_str = f" (Tech: {', '.join(techs)})" if techs else ""
                    lines.append(f"    * {title} at {company}{t_str}")

        # Projects
        proj_list = candidate_profile.get("projects") or []
        if proj_list:
            lines.append("- Projects Evidenced in Resume:")
            for proj in proj_list[:3]:
                if isinstance(proj, dict):
                    p_name = proj.get("name") or "Project"
                    p_desc = proj.get("description") or ""
                    p_techs = proj.get("technologies") or []
                    t_str = f" [Tech: {', '.join(p_techs)}]" if p_techs else ""
                    lines.append(f"    * {p_name}{t_str}: {p_desc[:120]}")

        # Education
        edu_list = candidate_profile.get("education") or []
        if edu_list:
            lines.append("- Education:")
            for edu in edu_list[:2]:
                if isinstance(edu, dict):
                    deg = edu.get("degree") or "Degree"
                    inst = edu.get("institution") or "Institution"
                    lines.append(f"    * {deg} from {inst}")

    # Matched Jobs from Current Session
    if matched_jobs:
        lines.append("\nTop Matched Jobs from Current Job Search Session:")
        for idx, job in enumerate(matched_jobs[:3], 1):
            if isinstance(job, dict):
                j_title = job.get("job_title") or job.get("title") or "Job"
                j_comp = job.get("company") or "Company"
                j_skills = job.get("skills") or ""
                lines.append(f"    [{idx}] {j_title} at {j_comp} | Required Skills: {j_skills}")

    return "\n".join(lines).strip()


# ===========================================================================
# 3. LANGCHAIN RAG CHAIN BUILDER (LCEL)
# ===========================================================================

def is_conversational_history_query(query: str) -> bool:
    """Detect if a user query is asking about the chat history, prior questions, past answers, or conversation recap."""
    q = query.lower().strip()
    history_patterns = [
        r"\b(?:before|previous|earlier|last|prior|past)\s+(?:conversation|chat|discussion|question|query|prompt|ask|asked|topic|message)\b",
        r"\b(?:about|regarding|in)\s+(?:our|the|my)?\s*(?:before|previous|earlier|last|prior|past)\s+(?:conversation|chat|discussion|question|query|message)\b",
        r"\bwhat\s+did\s+(?:i|you|we)\s+(?:ask|say|query|mention|answer|talk\s+about|discuss)\b",
        r"\bwhat\s+was\s+(?:my|our|the)\s+(?:before|previous|earlier|last|prior)\b",
        r"\bwhat\s+we\s+(?:talked|discussed|said)\b",
        r"\bwhat\s+was\s+discussed\b",
        r"\brepeat\s+(?:your\s+)?(?:last|previous|earlier)\s+(?:answer|response|message)\b",
        r"\bwhat\s+(?:were\s+the\s+previous\s+questions|questions\s+did\s+i\s+ask)\b",
        r"\bcan\s+you\s+remind\s+me\s+what\b",
        r"\babout\s+(?:my\s+)?(?:before|previous|earlier|last)\s+(?:conversation|question|discussion)\b",
        r"\b(?:summarize|recap)\s+(?:the|our)?\s*(?:conversation|chat|discussion|history)\b",
        r"\b(?:chat\s+history|conversation\s+history)\b",
    ]
    return any(re.search(p, q) for p in history_patterns)


def build_langchain_mentor_prompt() -> ChatPromptTemplate:
    """Build the LangChain ChatPromptTemplate with system instructions and chat history."""
    return ChatPromptTemplate.from_messages([
        ("system", MENTOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "{candidate_context_section}"
            "RETRIEVED DUAL-SOURCE EVIDENCE (Job Market Postings & Career Guidance Notes):\n{context}\n\n"
            "USER CAREER QUESTION:\n{question}\n\n"
            "Guidelines:\n"
            "- Answer directly, putting the most useful information first without generic filler or repetitive introductions.\n"
            "- Adapt the format to the question: use numbered steps for how-to, bullet points for lists, tables or comparisons for alternatives, and roadmaps for career planning.\n"
            "- If an active selected target job or company is present, prioritize and focus your advice directly on that specific role and company (e.g. when asked about 'this company' or 'this job', answer precisely for that selected company and role).\n"
            "- If the user asks about prior questions, previous messages, before conversation, or anything discussed earlier — answer DIRECTLY and ACCURATELY from the conversation history in the chat messages above. Do NOT refuse or say 'I don't know' when the answer is available in the conversation history.\n"
            "- If the user asks a follow-up, refinement, or filtering question (such as asking for 'only skills to learn', 'roadmap only', 'clarify step 2', etc.) regarding the job or topic discussed in the conversation — use the conversation history and evidence to fulfill their request directly.\n"
            "- For questions about prior conversation that also need career knowledge, combine your conversation memory with the retrieved evidence.\n"
            "- If candidate context is present, personalize your answer, compare demonstrated skills against target job requirements, and treat missing resume evidence strictly as 'not demonstrated'.\n"
            "- Ground all factual claims in the retrieved evidence, but do NOT display, quote, or dump raw evidence chunks.\n"
            "- Do NOT mention background processes, retrieval mechanics, or phrases like 'Based on the retrieved documents'. Speak naturally as a sharp AI Career Mentor.\n"
            "- If the evidence does not provide enough information to answer the question AND the answer is NOT available in conversation history, state: 'I don't know based on the available documents.'"
        ),
    ])


def create_gemini_runnable(client_override: Optional[Any] = None, model_name: Optional[str] = None):
    """Wrap Gemini API model invocation inside a LangChain RunnableLambda."""
    active_client = client_override or client
    active_model = model_name or MODEL_NAME

    def _invoke_gemini(prompt_value: Any) -> str:
        # Convert LangChain ChatPromptValue messages into content string
        messages = prompt_value.to_messages()
        full_prompt = "\n\n".join(f"{msg.type.upper()}: {msg.content}" for msg in messages)

        response = active_client.models.generate_content(
            model=active_model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )
        return getattr(response, "text", "") or ""

    return RunnableLambda(_invoke_gemini)


def create_mentor_lcel_chain(client_override: Optional[Any] = None, model_name: Optional[str] = None):
    """
    Construct the full LangChain Expression Language (LCEL) RAG pipeline:
    Inputs -> Context Formatter -> ChatPromptTemplate -> Gemini Runnable -> StrOutputParser
    """
    prompt_template = build_langchain_mentor_prompt()
    gemini_runnable = create_gemini_runnable(client_override=client_override, model_name=model_name)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_langchain_documents(x["documents"]),
            candidate_context_section=lambda x: f"{x['candidate_context']}\n\n" if x.get("candidate_context") else "",
        )
        | prompt_template
        | gemini_runnable
        | StrOutputParser()
    )
    return chain


# ===========================================================================
# 4. MAIN ENTRY POINT: ASK CAREER MENTOR
# ===========================================================================

def ask_career_mentor(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    candidate_profile: Optional[Dict[str, Any]] = None,
    matched_jobs: Optional[List[Dict[str, Any]]] = None,
    selected_target_job: Optional[Dict[str, Any]] = None,
    top_k: int = 6,
    min_score: float = 0.35,
    client_override: Optional[Any] = None,
    model_name: Optional[str] = None,
    include_sources_in_answer: bool = False,
) -> Dict[str, Any]:
    """
    Execute the SmartHire AI Career Mentor LangChain RAG pipeline.

    Workflow:
    1. Input Guardrail: Validates input, rejects unsafe/exploit or off-topic requests.
    2. LangChain MentorRetriever: Searches FAISS for dual-source grounded evidence (jobs + career notes).
    3. Retrieval Quality Gate: If no documents meet threshold, triggers fallback refusal.
    4. LangChain LCEL Chain: Executes with candidate personalization, target job grounding, and memory.
    5. Citation Alignment: Enforces verified source metadata from retrieved LangChain Documents.
    """
    # -----------------------------------------------------------------------
    # Step 1: Input Guardrail Check
    # -----------------------------------------------------------------------
    guardrail_result = validate_mentor_input(query, conversation_history=conversation_history)
    if not guardrail_result["is_valid"]:
        status = guardrail_result["status"]
        rejection_msg = guardrail_result["rejection_message"]
        logger.info(f"[career_mentor] Query rejected by guardrails: status={status}")
        return {
            "answer": rejection_msg,
            "status": status,
            "sources": [],
            "evidence": [],
            "is_grounded": True,
        }

    # -----------------------------------------------------------------------
    # Step 2: LangChain Retriever Invocation & Extra Grounded Docs
    # -----------------------------------------------------------------------
    retriever = MentorRetriever(top_k=top_k, min_score=min_score)
    retrieval_query = contextualize_retrieval_query(
        query=query,
        history=conversation_history,
        selected_target_job=selected_target_job,
    )

    try:
        documents: List[Document] = retriever.invoke(retrieval_query)
    except Exception as e:
        logger.error(f"[career_mentor] Retriever error: {e}")
        documents = []

    # Synthesize grounded Document for selected target job if present
    extra_docs: List[Document] = []
    if selected_target_job:
        t_comp = selected_target_job.get("company", "Selected Company")
        t_role = selected_target_job.get("job_title") or selected_target_job.get("title", "Selected Target Role")
        t_skills = selected_target_job.get("skills", "")
        t_loc = selected_target_job.get("location", "")
        t_desc = selected_target_job.get("job_description") or selected_target_job.get("description", "")
        t_level = selected_target_job.get("level") or selected_target_job.get("experience") or ""
        t_content = (
            f"SELECTED TARGET JOB POSTING DETAILS (CANDIDATE PRIMARY CHOICE):\n"
            f"Company: {t_comp}\n"
            f"Target Role: {t_role}\n"
            f"Location: {t_loc}\n"
            f"Experience Level: {t_level}\n"
            f"Required Skills: {t_skills}\n"
            f"Job Description:\n{t_desc}"
        )
        extra_docs.append(
            Document(
                page_content=t_content,
                metadata={
                    "source": f"Selected Target Job: {t_role} at {t_comp}",
                    "source_type": "selected_target_job",
                    "title": t_role,
                    "company": t_comp,
                    "skills": t_skills,
                },
            )
        )

    # Synthesize grounded Document for conversational history query
    if conversation_history:
        history_lines = []
        for idx, msg in enumerate(conversation_history, 1):
            r = msg.get("role", "user").capitalize()
            c = msg.get("content", "").strip()
            if c:
                history_lines.append(f"Turn [{idx}] {r}: {c}")
        extra_docs.append(
            Document(
                page_content="CURRENT SESSION CONVERSATION HISTORY & PRIOR QUESTIONS:\n" + "\n".join(history_lines),
                metadata={
                    "source": "Session Conversation History",
                    "source_type": "conversation_history",
                    "title": "Chat History & Prior Questions",
                },
            )
        )

    # Combine extra docs with retrieved documents (avoiding duplicates)
    if extra_docs:
        seen_texts = {d.page_content for d in extra_docs}
        remaining_docs = [d for d in documents if d.page_content not in seen_texts]
        documents = extra_docs + remaining_docs[:max(1, top_k - len(extra_docs))]

    # -----------------------------------------------------------------------
    # Step 3: Retrieval Quality Gate (Calibrated Sufficiency)
    # -----------------------------------------------------------------------
    if not documents and not conversation_history:
        logger.info(f"[career_mentor] Quality gate failed (no evidence >= {min_score}) for query: '{query[:50]}'")
        return {
            "answer": UNSUPPORTED_FALLBACK_MESSAGE,
            "status": "unsupported",
            "sources": [],
            "evidence": [],
            "is_grounded": True,
        }

    # Extract distinct source metadata from LangChain Document objects
    retrieved_sources = []
    seen_sources = set()
    for doc in documents:
        src = doc.metadata.get("source")
        if src and src not in seen_sources and doc.metadata.get("source_type") != "conversation_history":
            seen_sources.add(src)
            retrieved_sources.append(doc.metadata)

    evidence_list = [
        {"text": doc.page_content, **doc.metadata} for doc in documents
    ]

    # -----------------------------------------------------------------------
    # Step 4: Execute LangChain LCEL RAG Chain
    # -----------------------------------------------------------------------
    chat_history_messages = []
    if conversation_history:
        for msg in conversation_history[-8:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                chat_history_messages.append(("human", content))
            else:
                chat_history_messages.append(("ai", content))

    lcel_chain = create_mentor_lcel_chain(client_override=client_override, model_name=model_name)
    candidate_context_str = build_candidate_context(
        candidate_profile=candidate_profile,
        matched_jobs=matched_jobs,
        selected_target_job=selected_target_job,
    )

    try:
        raw_answer = lcel_chain.invoke({
            "question": query,
            "documents": documents,
            "candidate_context": candidate_context_str,
            "chat_history": chat_history_messages,
        })
    except Exception as e:
        logger.error(f"[career_mentor] LangChain execution error: {e}")
        return {
            "answer": "An error occurred while generating career guidance. Please try again.",
            "status": "error",
            "sources": [],
            "evidence": evidence_list,
            "is_grounded": False,
        }

    # -----------------------------------------------------------------------
    # Step 5: Citation Validation & Alignment
    # -----------------------------------------------------------------------
    if "i don't know based on the available documents" in raw_answer.lower():
        final_status = "unsupported"
        final_sources = []
    else:
        final_status = "valid"
        final_sources = retrieved_sources

    if include_sources_in_answer:
        display_answer = validate_and_align_citations(raw_answer, retrieved_sources)
    else:
        display_answer = strip_source_citations(raw_answer)

    return {
        "answer": display_answer,
        "status": final_status,
        "sources": final_sources,
        "evidence": evidence_list,
        "is_grounded": True,
    }

