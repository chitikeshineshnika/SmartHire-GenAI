
"""
Guardrails and validation for SmartHire GenAI.

Provides:
1. Basic input validation (legacy chat / general queries).
2. CV generator input validation with structural and safety checks.
3. Prompt injection detection for untrusted candidate and job data.
4. Output verification & hallucination checks:
   - Bullet point metric and grounding verification.
   - Professional summary credential and experience grounding.
   - End-to-end tailored resume structure and entity preservation.
"""

import logging
import re
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ===========================================================================
# LEGACY & GENERAL INPUT VALIDATION
# ===========================================================================

MAX_INPUT_LENGTH = 4000

ALLOWED_TOPICS = (
    "resume",
    "career",
    "job",
    "skill",
    "interview",
    "education",
    "employment",
    "professional development",
)


def validate_input(user_input: str) -> tuple[bool, str]:
    """Validate a user's message before sending it to an AI model."""
    if not isinstance(user_input, str):
        return False, "Input must be text."

    cleaned_input = user_input.strip()

    if not cleaned_input:
        return False, "Input cannot be empty."

    if len(cleaned_input) > MAX_INPUT_LENGTH:
        return False, "Input is too long."

    if not any(topic in cleaned_input.lower() for topic in ALLOWED_TOPICS):
        return False, "Please ask a question related to jobs or careers."

    return True, "Input accepted."


# ===========================================================================
# PROMPT INJECTION DETECTION
# ===========================================================================

# Compiled regex patterns targeting common adversarial injection techniques
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b"),
    re.compile(r"(?i)\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b"),
    re.compile(r"(?i)\b(?:system\s+prompt|system\s+message)\b"),
    re.compile(r"(?i)\bact\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken|DAN)\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+(?:an?\s+)?(?:unrestricted|jailbroken|DAN)\b"),
    re.compile(r"(?i)\b(?:jailbreak|DAN\s+mode|developer\s+mode)\b"),
    re.compile(r"(?i)\b(?:bypass|override)\s+(?:safety|rules|guardrails|filters|guidelines)\b"),
    re.compile(r"(?i)\bforget\s+(?:all\s+)?your\s+(?:rules|instructions|guidelines)\b"),
    re.compile(r"(?i)\bdo\s+anything\s+now\b"),
    re.compile(r"(?i)<\s*(?:system|instruction|prompt)\s*>"),
    re.compile(r"(?i)\[\s*(?:system|instruction|prompt)\s*\]"),
]


def detect_prompt_injection(text: str) -> bool:
    """
    Check whether a text string contains prompt injection or jailbreak patterns.

    Args:
        text: The string to inspect (e.g. from resume text or job description).

    Returns:
        True if an injection pattern is detected, False otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("[guardrails] Prompt injection detected matching: %s", pattern.pattern)
            return True

    return False


# ===========================================================================
# CV GENERATOR INPUT VALIDATION
# ===========================================================================

MAX_JOB_INPUT_LENGTH = 50000
MIN_JOB_INPUT_LENGTH = 10


def validate_cv_generator_inputs(
    resume_profile: Any,
    job_input: Any,
) -> tuple[bool, str]:
    """
    Validate candidate resume profile and target job inputs before generation.

    Performs structural integrity checks, completeness checks, length constraints,
    and prompt injection scans.

    Args:
        resume_profile: Parsed resume profile dictionary.
        job_input: Target job as a dictionary or raw job description string.

    Returns:
        tuple[bool, str]: (is_valid, error_or_success_message).
    """
    # ------------------------------------------------------------------
    # 1. Validate Resume Profile
    # ------------------------------------------------------------------
    if not isinstance(resume_profile, dict):
        return False, f"Resume profile must be a dictionary, got {type(resume_profile).__name__}."

    if not resume_profile:
        return False, "Resume profile cannot be empty."

    has_skills = bool(resume_profile.get("skills"))
    has_experience = bool(resume_profile.get("experience"))
    has_projects = bool(resume_profile.get("projects"))
    has_education = bool(resume_profile.get("education"))

    if not any([has_skills, has_experience, has_projects, has_education]):
        return (
            False,
            "Resume profile has no substantive content (no skills, experience, "
            "projects, or education). Cannot generate CV improvements.",
        )

    # Check for prompt injection in resume text fields
    resume_text_fields: list[str] = [
        str(resume_profile.get("name", "")),
        str(resume_profile.get("target_role", "")),
        str(resume_profile.get("professional_summary", "")),
    ]

    for exp in resume_profile.get("experience", []):
        if isinstance(exp, dict):
            resume_text_fields.append(str(exp.get("title", "")))
            resume_text_fields.append(str(exp.get("company", "")))
            desc = exp.get("description", [])
            if isinstance(desc, list):
                resume_text_fields.extend(str(b) for b in desc)
            elif isinstance(desc, str):
                resume_text_fields.append(desc)

    for proj in resume_profile.get("projects", []):
        if isinstance(proj, dict):
            resume_text_fields.append(str(proj.get("title", "")))
            desc = proj.get("description", [])
            if isinstance(desc, list):
                resume_text_fields.extend(str(b) for b in desc)
            elif isinstance(desc, str):
                resume_text_fields.append(desc)

    for field_text in resume_text_fields:
        if field_text and detect_prompt_injection(field_text):
            return (
                False,
                "Resume profile contains potentially unsafe instructions or prompt injection patterns.",
            )

    # ------------------------------------------------------------------
    # 2. Validate Target Job Input
    # ------------------------------------------------------------------
    if job_input is None:
        return False, "Target job input cannot be None."

    if isinstance(job_input, str):
        cleaned_job = job_input.strip()
        if not cleaned_job:
            return False, "Target job description cannot be empty."
        if len(cleaned_job) < MIN_JOB_INPUT_LENGTH:
            return (
                False,
                f"Target job description is too short (minimum {MIN_JOB_INPUT_LENGTH} characters).",
            )
        if len(cleaned_job) > MAX_JOB_INPUT_LENGTH:
            return (
                False,
                f"Target job description is too long (maximum {MAX_JOB_INPUT_LENGTH} characters).",
            )
        if detect_prompt_injection(cleaned_job):
            return (
                False,
                "Target job description contains potentially unsafe instructions or prompt injection patterns.",
            )

    elif isinstance(job_input, dict):
        if not job_input:
            return False, "Target job dictionary cannot be empty."

        has_title = bool(job_input.get("job_title") or job_input.get("title"))
        has_desc = bool(job_input.get("description"))
        has_skills_field = bool(job_input.get("skills") or job_input.get("required_skills"))
        has_raw = bool(job_input.get("raw_text"))

        if not any([has_title, has_desc, has_skills_field, has_raw]):
            return (
                False,
                "Target job dictionary must include at least a title, description, skills, or raw_text.",
            )

        # Check dictionary text values for prompt injection
        for key, val in job_input.items():
            if isinstance(val, str) and detect_prompt_injection(val):
                return (
                    False,
                    f"Target job field '{key}' contains potentially unsafe instructions or prompt injection patterns.",
                )
    else:
        return (
            False,
            f"Target job must be a string or dictionary, got {type(job_input).__name__}.",
        )

    return True, "Inputs validated successfully."


# ===========================================================================
# OUTPUT VERIFICATION & GROUNDING CHECKS
# ===========================================================================

# Pattern to capture metric numbers and percentages (e.g. 40%, 150k, 2.5x, $500)
_METRIC_PATTERN = re.compile(
    r"(?:\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?x\b|\b\d+(?:,\d{3})+\b)"
)

# Placeholder patterns indicating candidate should fill in verified numbers
_PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:X|N|amount|number|percentage|metric|percent|\d+%)\]",
    re.IGNORECASE,
)


def verify_bullet_grounding(
    original_bullets: list[str],
    improved_bullets: list[Union[dict, Any]],
) -> list[str]:
    """
    Verify that improved resume bullet points do not hallucinate unevidenced metrics.

    If an improved bullet introduces new quantitative metrics or percentages that
    were not present in the original bullet, it should format them as placeholders
    (e.g., '[X%]', '[amount]') rather than fabricating specific values.

    Args:
        original_bullets: List of original bullet point strings.
        improved_bullets: List of improved bullet dicts or Pydantic models.

    Returns:
        List of warning messages for any ungrounded additions.
    """
    warnings: list[str] = []

    for item in improved_bullets:
        if isinstance(item, dict):
            orig = item.get("original_bullet", "")
            improved = item.get("improved_bullet", "")
        else:
            orig = getattr(item, "original_bullet", "")
            improved = getattr(item, "improved_bullet", "")

        if not orig or not improved:
            continue

        # Extract metrics from original and improved bullets
        orig_metrics = set(_METRIC_PATTERN.findall(orig))
        improved_metrics = set(_METRIC_PATTERN.findall(improved))

        new_metrics = improved_metrics - orig_metrics
        if new_metrics:
            # Check if placeholders were used
            has_placeholder = bool(_PLACEHOLDER_PATTERN.search(improved))
            if not has_placeholder:
                bullet_preview = orig[:50] + ("..." if len(orig) > 50 else "")
                warnings.append(
                    f"Bullet '{bullet_preview}' introduced new ungrounded metric(s): "
                    f"{', '.join(sorted(new_metrics))}. "
                    "Unverifiable figures should use placeholders like '[X%]'.",
                )

    return warnings


# Degrees and educational credentials
_CREDENTIAL_PATTERNS = [
    (re.compile(r"\b(?:ph\.?d|doctorate|doctoral)\b", re.IGNORECASE), "Ph.D./Doctorate"),
    (re.compile(r"\b(?:master'?s|m\.?s\.?|m\.?tech|m\.?sc)\b", re.IGNORECASE), "Master's"),
    (re.compile(r"\b(?:mba)\b", re.IGNORECASE), "MBA"),
    (re.compile(r"\b(?:bachelor'?s|b\.?s\.?|b\.?tech|b\.?e\.?)\b", re.IGNORECASE), "Bachelor's"),
]


def verify_summary_grounding(
    original_profile: dict,
    new_summary: str,
) -> list[str]:
    """
    Verify that a rewritten professional summary does not introduce unearned credentials
    or ungrounded claims not supported by the candidate's profile.

    Args:
        original_profile: Candidate resume profile dict.
        new_summary: Rewritten professional summary text.

    Returns:
        List of warning messages.
    """
    warnings: list[str] = []
    if not isinstance(new_summary, str) or not new_summary.strip():
        return warnings

    # 1. Check prompt artifact leakage
    prompt_leak_tokens = [
        "UNTRUSTED_RESUME_DATA",
        "UNTRUSTED_TARGET_JOB",
        "BASE_GROUNDING_INSTRUCTIONS",
        "RESUME_JOB_COMPARISON",
    ]
    for token in prompt_leak_tokens:
        if token in new_summary:
            warnings.append(f"Rewritten summary contains internal prompt artifact: '{token}'.")

    # 2. Check credentials / degrees against education section
    edu_text = ""
    for edu in original_profile.get("education", []):
        if isinstance(edu, dict):
            edu_text += f" {edu.get('degree', '')} {edu.get('institution', '')}"
        elif isinstance(edu, str):
            edu_text += f" {edu}"
    orig_summary = str(original_profile.get("professional_summary", ""))
    full_orig_edu_context = (edu_text + " " + orig_summary).lower()

    for pattern, name in _CREDENTIAL_PATTERNS:
        if pattern.search(new_summary):
            if not pattern.search(full_orig_edu_context):
                warnings.append(
                    f"Rewritten summary mentions degree/credential '{name}' which is "
                    "not evidenced in the candidate's education history.",
                )

    # 3. Check claimed years of experience
    exp_year_match = re.search(r"(\d+)\+?\s*years(?:\s+of)?\s+experience", new_summary, re.IGNORECASE)
    if exp_year_match:
        claimed_years = int(exp_year_match.group(1))
        # Estimate candidate experience from work history
        work_entries = original_profile.get("experience", [])
        # Also check if original summary mentioned years
        orig_year_match = re.search(r"(\d+)\+?\s*years(?:\s+of)?\s+experience", orig_summary, re.IGNORECASE)
        max_evidenced_years = int(orig_year_match.group(1)) if orig_year_match else len(work_entries) * 2

        if max_evidenced_years > 0 and claimed_years > (max_evidenced_years + 3):
            warnings.append(
                f"Rewritten summary claims {claimed_years} years of experience, but profile "
                f"evidences approximately {max_evidenced_years} years.",
            )

    return warnings


def verify_tailored_resume(
    original_profile: dict,
    tailored_resume_markdown: str,
) -> dict[str, Any]:
    """
    Validate a complete job-tailored resume markdown document for structural completeness,
    entity preservation, and absence of hallucinated employers or prompt leakage.

    Args:
        original_profile: Candidate resume profile dict.
        tailored_resume_markdown: Tailored resume in Markdown format.

    Returns:
        dict with keys:
            - is_grounded (bool): True if no severe hallucination flags detected.
            - warnings (list[str]): Advisory warnings for the candidate.
            - hallucination_flags (list[str]): Critical hallucination or safety violations.
            - preserved_sections (list[str]): Successfully verified sections.
            - missing_sections (list[str]): Expected sections not found in output.
    """
    warnings: list[str] = []
    hallucination_flags: list[str] = []
    preserved_sections: list[str] = []
    missing_sections: list[str] = []

    if not isinstance(tailored_resume_markdown, str) or not tailored_resume_markdown.strip():
        return {
            "is_grounded": False,
            "warnings": ["Tailored resume is empty."],
            "hallucination_flags": ["Empty resume generated."],
            "preserved_sections": [],
            "missing_sections": ["Summary", "Experience", "Skills", "Education"],
        }

    # 1. Check for prompt leakage
    prompt_tokens = [
        "UNTRUSTED_RESUME_DATA",
        "UNTRUSTED_TARGET_JOB",
        "BASE_GROUNDING_INSTRUCTIONS",
        "TAILORED_RESUME_PROMPT",
        "<instruction>",
        "</instruction>",
    ]
    for token in prompt_tokens:
        if token in tailored_resume_markdown:
            hallucination_flags.append(f"Prompt leakage detected: found token '{token}'.")

    # 2. Structural section checks
    expected_sections = [
        ("summary", "Summary / Profile"),
        ("experience", "Experience / Work History"),
        ("skills", "Skills"),
        ("education", "Education"),
    ]

    md_lower = tailored_resume_markdown.lower()
    for sec_key, sec_name in expected_sections:
        if sec_key in md_lower:
            preserved_sections.append(sec_name)
        else:
            missing_sections.append(sec_name)
            warnings.append(f"Expected section '{sec_name}' was not detected in tailored resume.")

    # 3. Preserved employers verification
    original_experiences = original_profile.get("experience", [])
    for exp in original_experiences:
        if isinstance(exp, dict):
            company = exp.get("company", "").strip()
            if company and len(company) > 2:
                if company.lower() not in md_lower:
                    warnings.append(
                        f"Original employer '{company}' was not explicitly found in the tailored resume.",
                    )

    # 4. Preserved education institution verification
    original_education = original_profile.get("education", [])
    for edu in original_education:
        if isinstance(edu, dict):
            institution = edu.get("institution", "").strip()
            if institution and len(institution) > 2:
                if institution.lower() not in md_lower:
                    warnings.append(
                        f"Original educational institution '{institution}' was not found in the tailored resume.",
                    )

    is_grounded = len(hallucination_flags) == 0

    return {
        "is_grounded": is_grounded,
        "warnings": warnings,
        "hallucination_flags": hallucination_flags,
        "preserved_sections": preserved_sections,
        "missing_sections": missing_sections,
    }


# ===========================================================================
# AI CAREER MENTOR INPUT GUARDRAILS & SCOPE VALIDATION
# ===========================================================================

MENTOR_MAX_INPUT_LENGTH = 4000

MENTOR_UNSAFE_REJECTION_MESSAGE = (
    "I can't assist with that request. I can help with appropriate career guidance and professional development."
)

MENTOR_OFF_TOPIC_REJECTION_MESSAGE = (
    "I'm designed to help with career guidance using the available SmartHire documents. Please ask a career-related question."
)

# Unsafe requests, prompt injections, attacks, exploits, and harmful intents
MENTOR_UNSAFE_PATTERNS: list[re.Pattern] = [
    # Prompt injection / system prompt extraction / jailbreak
    re.compile(r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b"),
    re.compile(r"(?i)\b(?:reveal|show|print|output|display|dump)\s+(?:the\s+)?(?:system\s+prompt|developer\s+prompt|system\s+message|system\s+instructions)\b"),
    re.compile(r"(?i)\b(?:jailbreak|dan\s+mode|developer\s+mode|unrestricted\s+mode)\b"),
    re.compile(r"(?i)\b(?:bypass|override)\s+(?:safety|rules|guardrails|filters|guidelines)\b"),
    re.compile(r"(?i)\bdo\s+anything\s+now\b"),
    re.compile(r"(?i)\bact\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken|DAN)\b"),
    re.compile(r"(?i)<\s*(?:system|instruction|prompt)\s*>"),
    re.compile(r"(?i)\[\s*(?:system|instruction|prompt)\s*\]"),
    # Harmful attacks, malicious code, exploits, and criminal requests
    re.compile(r"(?i)\b(?:write|create|build|generate|code)\s+(?:a\s+)?(?:malware|ransomware|keylogger|trojan|virus|spyware|exploit|payload|rootkit|worm|ddos\s+script|backdoor)\b"),
    re.compile(r"(?i)\b(?:hack\s+into|penetrate|attack|breach)\s+(?:a|an|the|my|this|their)?\s*(?:website|server|database|network|account|system|wifi|router|phone|camera|bank|email|password)\b"),
    re.compile(r"(?i)\b(?:crack|steal|sniff|dump)\s+(?:passwords?|credentials?|credit\s+cards?|auth\s+tokens?|private\s+keys?|session\s+keys?)\b"),
    re.compile(r"(?i)\b(?:sql\s+injection|xss|csrf|rce)\s+(?:attack|payload|script|exploit)\s+(?:against|on|to\s+hack)\b"),
    re.compile(r"(?i)\b(?:how\s+to\s+make|synthesize|build)\s+(?:a\s+)?(?:bomb|explosive|weapon|poison|illegal\s+drugs?)\b"),
    re.compile(r"(?i)\b(?:suicide|self-harm|kill\s+myself)\b"),
]

# Legitimate career topics, roles, skills, and questions
CAREER_SCOPE_KEYWORDS = {
    # Core career concepts
    "career", "careers", "job", "jobs", "role", "roles", "position", "positions",
    "profession", "professional", "professionals", "work", "workplace", "employment",
    "roadmap", "pathway", "transition", "switch", "switch career", "promotion",
    "salary", "salaries", "package", "compensation", "stipend", "pay",
    "fresher", "freshers", "entry-level", "junior", "mid-level", "senior", "lead",
    "intern", "interns", "internship", "internships", "hire", "hiring",
    "recruit", "recruiting", "recruitment", "recruiter", "vacancy", "vacancies",
    "opening", "openings", "placement", "placements", "campus placement", "off-campus",
    # Skills, learning & career guidance
    "skill", "skills", "tech stack", "technologies", "technology", "tools",
    "upskill", "upskilling", "reskill", "reskilling", "job requirements", "required skills",
    "learning path", "learning roadmap", "skill roadmap", "skills to learn",
    "syllabus", "curriculum", "coursework",
    # Resume & portfolio
    "resume", "resumes", "cv", "curriculum vitae", "portfolio", "cover letter",
    "experience", "work history", "formatting", "bullet points", "projects",
    # Interview
    "interview", "interviews", "technical round", "hr round", "behavioral round",
    "coding interview", "mock interview", "interview prep", "interview questions",
    "offer", "offer letter", "negotiation",
    # Education & credentials
    "degree", "degrees", "btech", "b.tech", "be", "b.e", "bsc", "b.sc",
    "mca", "mtech", "m.tech", "msc", "m.sc", "college", "university",
    "cgpa", "gpa", "qualification", "qualifications", "eligibility", "eligible",
    "certification", "certifications", "certificate", "course", "courses",
    # Specific Job Roles & Titles (matching KB)
    "developer", "developers", "engineer", "engineers", "analyst", "analysts",
    "architect", "architects", "tester", "testers", "scientist", "scientists",
    "backend", "frontend", "fullstack", "full stack", "full-stack",
    "python developer", "java developer", "software engineer",
    "devops", "devops engineer", "sre", "site reliability", "platform engineer",
    "cloud engineer", "cloud architect", "cloud solutions",
    "data analyst", "data scientist", "data engineer", "big data", "analytics engineer",
    "product analyst", "business intelligence", "bi analyst", "reporting analyst",
    "cybersecurity", "security engineer", "penetration tester", "ethical hacker",
    "ethical hacking", "soc analyst", "application security", "appsec",
    "ai engineer", "ai research", "generative ai", "genai", "nlp engineer",
    "machine learning", "ml engineer", "computer vision", "api developer",
    # Core technical skills & tools
    "python", "java", "javascript", "typescript", "sql", "nosql", "c++", "c#", "golang",
    "react", "angular", "vue", "node", "nodejs", "fastapi", "django", "flask", "spring boot",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "linux", "git", "github",
    "ci/cd", "kafka", "redis", "mongodb", "postgresql", "mysql", "microservices",
    "rest api", "graphql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "spark", "hadoop", "tableau", "power bi", "excel", "siem", "wireshark",
}

# Regex patterns matching common career-related question intents
CAREER_INTENT_PATTERNS = [
    re.compile(r"(?i)\bhow\s+(?:can|do|should)\s+i\s+(?:become|start|prepare|transition|switch|learn|grow)\b"),
    re.compile(r"(?i)\bwhat\s+(?:skills|tools|technologies|qualifications|certifications)\s+(?:do|are|should)\b"),
    re.compile(r"(?i)\b(?:day\s+in\s+the\s+life|roles?\s+and\s+responsibilities)\b"),
    re.compile(r"(?i)\b(?:learning\s+path|learning\s+roadmap|skill\s+roadmap|career\s+path|career\s+roadmap)\b"),
    re.compile(r"(?i)\b(?:difference\s+between|compare|vs|versus)\b.*\b(?:engineer|developer|analyst|role|job)\b"),
    re.compile(r"(?i)\b(?:requirements?|prerequisites?)\s+for\b"),
    re.compile(r"(?i)\b(?:advice|tips|guidance|recommendations?)\s+for\b"),
    re.compile(r"(?i)\bshould\s+i\s+learn\b"),
    re.compile(r"(?i)\bhow\s+to\s+get\s+(?:a|an)?\s*(?:job|internship|offer)\b"),
    re.compile(r"(?i)\bprepare\s+for\s+(?:a|an)?\s*.*interview\b"),
    re.compile(r"(?i)\b(?:only\s+skills|skills\s+to\s+learn|tech\s+stack\s+to\s+learn)\b"),
]


def is_unsafe_mentor_request(user_input: str) -> tuple[bool, str]:
    """
    Check if a user input to the AI Career Mentor is unsafe or adversarial.

    Distinguishes between genuine career questions about security/penetration testing
    and actual adversarial attacks or requests to generate exploits/malware.

    Returns:
        (is_unsafe, reason)
    """
    if not isinstance(user_input, str):
        return True, "Input is not a string."

    cleaned = user_input.strip()
    if not cleaned:
        return False, ""

    # Check against prompt injection and unsafe patterns
    for pattern in MENTOR_UNSAFE_PATTERNS:
        if pattern.search(cleaned):
            logger.warning("[mentor_guardrails] Unsafe pattern detected: %s", pattern.pattern)
            return True, f"Pattern matched: {pattern.pattern}"

    return False, ""


# Regex patterns matching conversational follow-up questions or references to prior questions/conversation
CONVERSATIONAL_FOLLOWUP_PATTERNS = [
    re.compile(r"(?i)\b(?:before|previous|earlier|last|prior|past)\s+(?:conversation|chat|discussion|question|query|prompt|ask|asked|response|answer|message|topic|points?)\b"),
    re.compile(r"(?i)\b(?:about|regarding|in)\s+(?:our|the|my)?\s*(?:before|previous|earlier|last|prior|past)\s+(?:conversation|chat|discussion|question|query|message)\b"),
    re.compile(r"(?i)\bwhat\s+did\s+(?:i|you|we)\s+(?:ask|say|query|mention|answer|talk\s+about|discuss)\b"),
    re.compile(r"(?i)\bwhat\s+was\s+(?:my|our|the)\s+(?:before|previous|earlier|last|prior)\b"),
    re.compile(r"(?i)\b(?:what\s+we\s+(?:talked|discussed|said)|what\s+was\s+discussed)\b"),
    re.compile(r"(?i)\b(?:tell\s+me\s+more|elaborate|explain\s+further|can\s+you\s+clarify|expand\s+on\s+that)\b"),
    re.compile(r"(?i)\b(?:why\s+so|why\s+is\s+that|why\s+did\s+you|why\s+not)\b"),
    re.compile(r"(?i)\b(?:in\s+this\s+company|for\s+this\s+company|at\s+this\s+company|this\s+company|that\s+company|the\s+company)\b"),
    re.compile(r"(?i)\b(?:what\s+about\s+(?:it|that|this|the\s+first|the\s+second|the\s+third))\b"),
    re.compile(r"(?i)\b(?:repeat\s+that|summarize\s+that|summarize\s+(?:the|our)\s+conversation|recap\s+(?:the|our)\s+conversation|what\s+else|what\s+were\s+the\s+questions)\b"),
    re.compile(r"(?i)\b(?:history|chat\s+history|conversation\s+history)\b"),
    re.compile(r"(?i)\b(?:only\s+skills|skills\s+to\s+learn|mention\s+only|list\s+only)\b"),
]


def is_career_related(
    user_input: str,
    conversation_history: Optional[list[dict[str, Any]]] = None,
) -> bool:
    """
    Check if a user input falls within the permitted career-guidance scope.

    Covers career planning, resumes, job requirements, skills, interviews,
    learning roadmaps, qualifications, role comparisons, and valid conversational
    follow-ups within an active career mentoring session.
    """
    if not isinstance(user_input, str):
        return False

    cleaned = user_input.strip().lower()
    if not cleaned:
        return False

    # 1. Check for career intent regex patterns
    for pattern in CAREER_INTENT_PATTERNS:
        if pattern.search(cleaned):
            return True

    # 2. Tokenize and check against career keywords
    # Split on non-alphanumeric characters, keeping phrases or single tokens
    tokens = set(re.findall(r"[a-z0-9_\-\.\#\+]+", cleaned))
    
    # Check single tokens
    if any(tok in CAREER_SCOPE_KEYWORDS for tok in tokens):
        return True

    # Check multi-word keywords
    for keyword in CAREER_SCOPE_KEYWORDS:
        if " " in keyword and keyword in cleaned:
            return True

    # 3. Check for conversational follow-ups / prior conversation references if in an active session
    if conversation_history:
        for pattern in CONVERSATIONAL_FOLLOWUP_PATTERNS:
            if pattern.search(cleaned):
                return True

        # In an active career mentoring session, follow-up questions, clarifications,
        # requests to filter/reformat (e.g. "can you mention only skills to learn",
        # "what about salary?", "explain step 2", "tell me more"), or inquiries
        # are continuing the existing conversation.
        # Ensure obvious off-topic trivia (weather, recipes, sports, capitals, etc.) is rejected.
        off_topic_trivia = [
            r"\b(?:weather|temperature|forecast|rain|climate)\b",
            r"\b(?:recipe|cook|baking|cookies|pizza|pasta|cake|dough)\b",
            r"\b(?:world cup|fifa|cricket|football|nba|olympics|sports)\b",
            r"\b(?:capital of|president of|prime minister of)\b",
            r"\b(?:movie|film|song|lyrics|actor|actress)\b",
            r"\b(?:quantum physics|astrophysics|black hole|mars)\b",
            r"\b(?:math equation|solve \d+x|\d+\s*[\+\-\*\/]\s*\d+)\b",
        ]
        if not any(re.search(p, cleaned) for p in off_topic_trivia):
            return True

    return False


def validate_mentor_input(
    user_input: str,
    conversation_history: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Validate a user's query for the SmartHire AI Career Mentor before RAG or LLM invocation.

    Enforces Section 4 (Mandatory Input Guardrails) of the AI Career Mentor specification:
    1. Checks input validity (non-empty string, length constraints).
    2. Detects prompt injection, attacks, exploits, and unsafe intents.
    3. Checks relevance to permitted career-guidance scope or active conversation.
    4. Returns exact required rejection messages for unsafe or off-topic queries.

    Returns:
        dict: {
            "is_valid": bool,                   # True only if input is valid, safe, and career-related
            "status": str,                      # "valid", "unsafe", "off_topic", "invalid_input"
            "rejection_message": str | None,    # Exact user-facing rejection message if rejected
            "reason": str                       # Diagnostic reason for logging
        }
    """
    if not isinstance(user_input, str):
        return {
            "is_valid": False,
            "status": "invalid_input",
            "rejection_message": "Input must be text.",
            "reason": "Non-string input provided.",
        }

    cleaned = user_input.strip()

    if not cleaned:
        return {
            "is_valid": False,
            "status": "invalid_input",
            "rejection_message": "Input cannot be empty.",
            "reason": "Empty input provided.",
        }

    if len(cleaned) > MENTOR_MAX_INPUT_LENGTH:
        return {
            "is_valid": False,
            "status": "invalid_input",
            "rejection_message": "Input is too long. Please shorten your question.",
            "reason": f"Input length ({len(cleaned)}) exceeded maximum ({MENTOR_MAX_INPUT_LENGTH}).",
        }

    # Safety check: Detect prompt injections, malicious exploits, or unsafe intents
    is_unsafe, unsafe_reason = is_unsafe_mentor_request(cleaned)
    if is_unsafe:
        return {
            "is_valid": False,
            "status": "unsafe",
            "rejection_message": MENTOR_UNSAFE_REJECTION_MESSAGE,
            "reason": f"Unsafe request rejected: {unsafe_reason}",
        }

    # Scope check: Verify question is related to career guidance or active session
    if not is_career_related(cleaned, conversation_history=conversation_history):
        return {
            "is_valid": False,
            "status": "off_topic",
            "rejection_message": MENTOR_OFF_TOPIC_REJECTION_MESSAGE,
            "reason": "Query outside permitted career-guidance scope.",
        }

    return {
        "is_valid": True,
        "status": "valid",
        "rejection_message": None,
        "reason": "Input accepted.",
    }


def check_mentor_input_guardrail(
    user_input: str,
    conversation_history: Optional[list[dict[str, Any]]] = None,
) -> tuple[bool, str]:
    """
    Convenience gatekeeper function for AI Career Mentor.

    Returns:
        (is_accepted, message)
        If accepted: (True, "Input accepted.")
        If rejected: (False, rejection_message)
    """
    result = validate_mentor_input(user_input, conversation_history=conversation_history)
    if result["is_valid"]:
        return True, "Input accepted."
    return False, result["rejection_message"] or "Invalid input."


