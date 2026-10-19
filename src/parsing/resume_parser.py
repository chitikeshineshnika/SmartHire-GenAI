"""
Resume Parser Module for SmartHire GenAI.

Analyzes resume text (from src.parsing.loader) and extracts structured
candidate information: name, contact, target role, professional summary,
skills, education, experience, projects, certifications, and achievements.
Uses the Gemini API with a Pydantic schema following strict grounding:
- Single-value missing fields are returned as null (None).
- List fields are returned as empty lists ([]).
"""

import hashlib
import logging
import os
from pathlib import Path
import re
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.config import MODEL_NAME

logger = logging.getLogger(__name__)

load_dotenv()  # must run before creating the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CACHE_DIR = Path("data/.parse_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MEM_CACHE: dict[str, dict] = {}


# ---------- Pydantic models ----------
class ContactDetails(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None

    @field_validator(
        "email", "phone", "location", "linkedin", "github", "portfolio",
        mode="before",
    )
    @classmethod
    def clean_contact_str(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ("n/a", "none", "null", "not specified"):
                return None
            return clean
        return v


class Education(BaseModel):
    degree: Optional[str] = None
    specialization: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    grade: Optional[str] = None

    @field_validator(
        "degree", "specialization", "institution", "year", "grade",
        mode="before",
    )
    @classmethod
    def clean_edu_str(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ("n/a", "none", "null", "not specified"):
                return None
            return clean
        return v


class Experience(BaseModel):
    type: Literal["job", "internship"] = "job"
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    description: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)

    @field_validator("title", "company", "duration", mode="before")
    @classmethod
    def clean_exp_str(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ("n/a", "none", "null", "not specified"):
                return None
            return clean
        return v

    @field_validator("description", "technologies", mode="before")
    @classmethod
    def clean_exp_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [
                s.strip() for s in v
                if isinstance(s, str) and s.strip()
                and s.strip().lower() not in ("n/a", "none", "null")
            ]
        return v


class Project(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: list[str] = Field(default_factory=list)
    url: Optional[str] = None

    @field_validator("name", "description", "url", mode="before")
    @classmethod
    def clean_proj_str(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ("n/a", "none", "null", "not specified"):
                return None
            return clean
        return v

    @field_validator("technologies", mode="before")
    @classmethod
    def clean_proj_tech(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            seen, out = set(), []
            for s in v:
                if isinstance(s, str):
                    s = s.strip()
                    if s and s.lower() not in seen and s.lower() not in ("n/a", "none", "null"):
                        seen.add(s.lower())
                        out.append(s)
            return out
        return v


class Certification(BaseModel):
    name: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None
    credential_id: Optional[str] = None
    url: Optional[str] = None

    @field_validator(
        "name", "issuer", "date", "credential_id", "url",
        mode="before",
    )
    @classmethod
    def clean_cert_str(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ("n/a", "none", "null", "not specified"):
                return None
            return clean
        return v


class ResumeProfile(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: Optional[str] = None
    target_role: Optional[str] = None
    professional_summary: Optional[str] = None
    education: list[Education] = Field(default_factory=list)
    technical_skills: list[str] = Field(
        default_factory=list,
        description="Comprehensive list of technical skills, programming languages, frameworks, libraries, databases, cloud platforms, and developer tools.",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Alias for technical_skills.",
    )
    work_experience: list[Experience] = Field(
        default_factory=list,
        description="Work experience and internships written in your own words with real context and technical skills used.",
    )
    experience: list[Experience] = Field(
        default_factory=list,
        description="Alias for work_experience.",
    )
    projects: list[Project] = Field(
        default_factory=list,
        description="Projects written in your own words with real context and technical skills used.",
    )

    # Optional legacy fields for backward compatibility with existing tests
    contact: Optional[ContactDetails] = None
    soft_skills: list[str] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_aliases(self):
        if self.technical_skills and not self.skills:
            self.skills = list(self.technical_skills)
        elif self.skills and not self.technical_skills:
            self.technical_skills = list(self.skills)

        if self.work_experience and not self.experience:
            self.experience = list(self.work_experience)
        elif self.experience and not self.work_experience:
            self.work_experience = list(self.experience)
        return self

    @field_validator("name", "professional_summary", mode="before")
    @classmethod
    def clean_profile_str(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ("n/a", "none", "null", "not specified"):
                return None
            return clean
        return v

    @field_validator("target_role", mode="before")
    @classmethod
    def validate_target_role(cls, v):
        """
        Validate target_role with strict no-assumptions rule:
        - Extract target_role ONLY when the resume explicitly and clearly states the candidate's intended career position.
        - If not explicitly specified, return None (null).
        - Never infer target_role from internships, previous job titles, work experience, projects, skills, certifications, degree, or academic background.
        - Do not confuse past/current roles with desired future role.
        - Reject vague phrases like 'Looking for a suitable opportunity in the IT industry'.
        """
        if v is None:
            return None
        if not isinstance(v, str):
            return None

        clean = v.strip()
        if not clean or clean.lower() in ("n/a", "none", "null", "not specified", "target role not specified", "unspecified", "undefined"):
            return None

        lower = clean.lower()

        # Disqualify if it refers to an internship / intern
        if re.search(r"\b(?:intern|internship|interns)\b", lower):
            return None

        # Clean leading objective/intent prefixes
        prefix_patterns = [
            r"^(?:objective\s*:\s*)?(?:to\s+(?:work|secure|obtain|get)\s+(?:as\s+(?:a\s+|an\s+)?|a\s+position\s+as\s+|a\s+role\s+as\s+|position\s+as\s+|role\s+as\s+)?|to\s+be\s+(?:a\s+|an\s+)?)",
            r"^(?:seeking\s+(?:a\s+|an\s+)?(?:full-time\s+|part-time\s+|challenging\s+|entry-level\s+)?(?:position|role|job|opportunity)?\s*(?:as\s+(?:a\s+|an\s+)?)?)",
            r"^(?:aspiring\s+(?:to\s+be\s+(?:a\s+|an\s+)?)?)",
            r"^(?:looking\s+for\s+(?:a\s+|an\s+)?(?:suitable\s+|challenging\s+)?(?:role|position|job|opportunity)\s*(?:as\s+(?:a\s+|an\s+)?|in\s+(?:the\s+)?)?)",
            r"^(?:aiming\s+for\s+(?:a\s+|an\s+)?(?:role|position|job)\s+(?:as\s+(?:a\s+|an\s+)?)?)",
        ]
        cleaned_role = clean
        for pat in prefix_patterns:
            cleaned_role = re.sub(pat, "", cleaned_role, flags=re.IGNORECASE).strip()

        cleaned_role = re.sub(r"^(?:a|an)\s+", "", cleaned_role, flags=re.IGNORECASE).strip()
        cleaned_role = re.sub(r"\s+(?:position|role|job|opportunity)$", "", cleaned_role, flags=re.IGNORECASE).strip()

        lower = cleaned_role.lower()

        # 1. Reject vague aspirations or industry mentions without a specific career job title
        vague_patterns = [
            r"^(?:it|information technology|software|tech|corporate)\s+industry$",
            r"^(?:it|software|tech)\s+sector$",
            r"^opportunity\s+in\s+.*$",
            r"^suitable\s+opportunity.*$",
            r"^challenging\s+role$",
            r"^growth\s+oriented.*$",
            r"^entry\s+level\s+position$",
        ]
        for vpat in vague_patterns:
            if re.match(vpat, lower):
                return None

        # 2. Educational qualifications, degree names, or student status terms disqualify target_role
        student_degree_terms = (
            "undergraduate", "undergrad", "undergrads", "postgraduate", "student", "students",
            "fresher", "freshers", "bachelor", "bachelors", "b.tech", "btech", "m.tech", "mtech",
            "b.e", "b.e.", "m.e", "m.e.", "b.s", "b.s.", "m.s", "m.s.",
            "bsc", "msc", "bca", "mca", "phd", "ph.d", "diploma", "degree",
            "intermediate", "matriculate", "studying", "pursuing"
        )
        for term in student_degree_terms:
            if re.search(r"\b" + re.escape(term) + r"\b", lower):
                return None

        # 3. Standalone graduate status
        if lower.endswith(("graduate", "graduates")):
            return None
        words = lower.split()
        if "graduate" in words and not any(kw in words for kw in ("engineer", "developer", "analyst", "trainee", "associate", "consultant")):
            return None

        # 4. Disqualify standalone branches of study / academic disciplines
        standalone_majors = (
            "computer science", "cs", "cse", "information technology", "it",
            "computer science and engineering", "computer science engineering",
            "software engineering", "computer engineering",
            "electrical engineering", "electronics and communication",
            "mechanical engineering", "civil engineering", "data science"
        )
        if lower in standalone_majors:
            return None

        # 5. Require at least one valid occupational/job title noun (excluding intern/internship)
        common_role_nouns = (
            "developer", "developers", "engineer", "engineers", "analyst", "analysts",
            "scientist", "scientists", "architect", "architects", "programmer", "programmers",
            "consultant", "consultants", "designer", "designers", "administrator", "administrators",
            "specialist", "specialists", "lead", "leads", "manager", "managers",
            "officer", "officers", "associate", "associates", "tester", "testers",
            "trainee", "trainees", "technician", "technicians", "coordinator", "coordinators",
            "executive", "executives", "director", "directors"
        )
        tokens = re.findall(r"[a-zA-Z]+", lower)
        if not any(t in common_role_nouns for t in tokens):
            return None

        return cleaned_role

    @field_validator("skills", "technical_skills", "soft_skills", "achievements", mode="before")
    @classmethod
    def clean_str_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            seen, out = set(), []
            for s in v:
                if isinstance(s, str):
                    s = s.strip()
                    if s and s.lower() not in seen and s.lower() not in ("n/a", "none", "null"):
                        seen.add(s.lower())
                        out.append(s)
            return out
        return v

    @field_validator(
        "education", "experience", "work_experience", "projects", "certifications",
        mode="before",
    )
    @classmethod
    def ensure_list_of_objects(cls, v):
        if v is None:
            return []
        return v


# ---------- Dedicated LLM Output Schema (No Phone, Email, LinkedIn, GitHub) ----------
class _LLMResumeSchema(BaseModel):
    """
    Schema passed to Gemini API for structured resume extraction.
    Strictly contains ONLY:
    - name
    - target_role
    - professional_summary
    - education
    - technical_skills
    - work_experience
    - projects
    Strictly excludes phone number, email, linkedin, github, and contact details.
    """
    name: Optional[str] = Field(
        default=None,
        description="Full candidate name.",
    )
    target_role: Optional[str] = Field(
        default=None,
        description="Explicit intended occupational target job role ONLY IF the candidate explicitly and clearly states their desired career position in the resume (e.g. in a dedicated target role section, career objective, or professional summary stating clear career intent, such as 'Seeking a full-time Java Developer position'). Return null if not explicitly stated, missing, or vague.",
    )
    professional_summary: Optional[str] = Field(
        default=None,
        description="Concise professional summary of the candidate's background, qualifications, and core competencies extracted from the resume text. Return null if absent.",
    )
    education: list[Education] = Field(
        default_factory=list,
        description="Educational qualifications: degree, specialization, institution, graduation year, and grade.",
    )
    technical_skills: list[str] = Field(
        default_factory=list,
        description="Comprehensive list of technical skills, programming languages, frameworks, libraries, databases, cloud platforms, and developer tools.",
    )
    work_experience: list[Experience] = Field(
        default_factory=list,
        description="List of work experience and internships. For each entry, write in your own words a comprehensive narrative of the real context (responsibilities, problems solved, system architecture, domain impact) tailored for semantic job search, and explicitly list what technical skills were used in 'technologies'.",
    )
    projects: list[Project] = Field(
        default_factory=list,
        description="List of projects. For each project, write in your own words a comprehensive narrative of the real context (problem solved, architecture, outcomes) tailored for semantic job search, and explicitly list what technical skills were used in 'technologies'.",
    )


# ---------- Prompt (Privacy-Preserving & Rich Semantic Context) ----------
SYSTEM_PROMPT = """You extract structured candidate profiles from resume text with high fidelity for semantic job search.

STRICT PRIVACY RULES (CRITICAL):
1. DO NOT EXTRACT CONTACT DETAILS:
   - NEVER extract phone numbers, email addresses, LinkedIn URLs, GitHub URLs, portfolio URLs, or physical addresses.
   - Strictly ignore all phone numbers, email addresses, LinkedIn links, and GitHub links.
   - Do NOT output any phone number, email, LinkedIn, or GitHub information under any circumstances.

GROUNDING & SYNTHESIS RULES:
2. Grounding: Use ONLY facts explicitly supported by the resume text. Do not invent employment history, fake degrees, or unmentioned skills.
3. Missing-Value Policy:
   - For single-value fields (name, target_role, professional_summary), return null if missing, vague, or not mentioned.
   - For list fields (education, technical_skills, work_experience, projects), return an empty list [] if missing or not mentioned.
   - NEVER use placeholder strings such as "N/A", "None", "null", "Not specified", or empty strings.
4. Name: Full candidate name.
5. Target Role (STRICT NO-ASSUMPTIONS RULE — CRITICAL):
   - Extract target_role ONLY when the resume explicitly and clearly states the candidate's intended career position (e.g. in a dedicated target role header, career objective, or professional summary with clear career intent such as "Seeking a full-time Java Developer position").
   - If the target role is missing, vague, contradictory, or ambiguous, return null. Never guess, predict, or generate a role to fill the gap.
   - Vague statements like "Looking for a suitable opportunity in the IT industry", "Seeking a challenging role", or "Seeking an entry-level position" DO NOT specify a target role — return null.
   - NEVER infer the target role from internships, previous job titles, work experience, projects, skills, certifications, degrees, or academic background.
   - NEVER confuse a candidate's past or current position (e.g. previous experience as a Software Engineer, or a Python Developer Intern) with their desired future target role.
   - NEVER treat a degree, branch of study, student status, or internship as a target job role.
6. Professional Summary:
   - Extract a concise professional summary of the candidate's stated background, competencies, and experience if present in the resume. Return null if absent.
7. Education:
   - List educational qualifications: degree, specialization, institution, graduation year/dates, and grade/CGPA when available.
8. Technical Skills:
   - Comprehensive list of technical skills, programming languages, frameworks, libraries, databases, cloud platforms, and developer tools as distinct individual list items (e.g. ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"]).
9. Work Experience & Internships (REAL CONTEXT FOR SEMANTIC JOB SEARCH):
   - For each job or internship, write in your own words a substantive, detailed description that explains the real context of the work: the business or technical domain, system architecture, responsibilities, scale, and problems solved. This real context is directly embedded and used for semantic job search matching.
   - In 'technologies', explicitly list all the technical skills, programming languages, frameworks, databases, and tools used in that work experience.
   - Include company name, role title, duration/dates, and type ("job" or "internship").
10. Projects (REAL CONTEXT FOR SEMANTIC JOB SEARCH):
   - For each academic, personal, or open-source project, write in your own words a substantive description explaining the real context: the problem solved, system architecture, technical implementation, and outcomes. This real context is directly embedded and used for semantic job search matching.
   - In 'technologies', explicitly list all the technical skills, programming languages, libraries, frameworks, and developer tools used in that project.
   - Include project name.
11. Preservation: Keep dates, durations, grades, credential details, and technical keywords accurate."""


# ---------- LLM calls ----------
def _call_llm(resume_text: str, error: Optional[str] = None) -> ResumeProfile:
    prompt = f"Resume text:\n\n{resume_text}"
    if error:
        prompt += (f"\n\nYour previous output was invalid: {error}. "
                   "Fix it and follow the rules exactly.")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=_LLMResumeSchema,
        ),
    )
    response_text = getattr(response, "text", None) or ""
    if not response_text.strip():
        raise ValueError("Google Gemini returned an empty response. Retrying...")

    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    parsed = _LLMResumeSchema.model_validate_json(cleaned)
    target_role_val = parsed.target_role
    if target_role_val is not None and str(target_role_val).strip().lower() in ("none", "null", "not specified", "target role not specified", "n/a", ""):
        target_role_val = None

    return ResumeProfile(
        name=parsed.name,
        target_role=target_role_val,
        professional_summary=parsed.professional_summary,
        education=parsed.education,
        technical_skills=parsed.technical_skills,
        skills=parsed.technical_skills,
        work_experience=parsed.work_experience,
        experience=parsed.work_experience,
        projects=parsed.projects,
        contact=None,
        soft_skills=[],
        certifications=[],
        achievements=[],
    )


def _call_llm_with_retry(
    resume_text: str,
    error: Optional[str] = None,
    attempts: int = 3,
    initial_delay: float = 1.0,
) -> ResumeProfile:
    """Retry when Gemini is busy (503) or rate-limited (429) with bounded delays."""
    delay = initial_delay
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            return _call_llm(resume_text, error=error)
        except errors.APIError as e:
            last_error = e
            if e.code in (429, 500, 503, 504) and attempt < attempts:
                logger.warning(
                    "Gemini API returned %s. Attempt %d/%d; retrying in %.1fs...",
                    e.code, attempt, attempts, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 4.0)
            else:
                break
        except Exception as e:
            last_error = e
            break

    if isinstance(last_error, errors.APIError):
        if last_error.code == 503:
            raise RuntimeError(
                f"Google Gemini is temporarily experiencing high demand (HTTP 503). "
                f"Please try again in a few moments."
            ) from last_error
        elif last_error.code == 429:
            raise RuntimeError(
                f"Google Gemini rate limit or quota exceeded (HTTP 429). "
                f"Please wait a moment before trying again."
            ) from last_error
        else:
            raise RuntimeError(
                f"Google Gemini API error ({last_error.code}): {last_error.message}"
            ) from last_error

    raise RuntimeError("Failed to generate response from Google Gemini.")


def build_profile_text(profile: dict) -> str:
    """
    Build a dense semantic text representation for candidate embedding & job matching (Module 2).
    Includes target role, technical skills,
    work experience with real contextual responsibilities and technical skills used,
    projects with real contextual problem descriptions and technical skills used,
    and education.
    Strictly excludes personal contact details (email, phone, URLs).
    Skips null values and empty lists.
    """
    parts = []

    # 1. Target Role (only include if explicitly specified, avoid adding 'Not specified' into embeddings)
    target_role = profile.get("target_role")
    if target_role and str(target_role).strip().lower() not in ("not specified", "none", "null", "n/a"):
        parts.append(str(target_role).strip())

    # 2. Professional Summary
    summary = profile.get("professional_summary") or profile.get("summary")
    if summary and str(summary).strip().lower() not in ("none", "null", "not specified", "n/a"):
        parts.append(str(summary).strip())

    # 3. Technical Skills & Soft Skills
    all_skills = list(profile.get("technical_skills") or profile.get("skills") or [])
    if profile.get("soft_skills"):
        all_skills.extend(profile["soft_skills"])
    if all_skills:
        seen = set()
        deduped = [s for s in all_skills if s and s.lower() not in seen and not seen.add(s.lower())]
        if deduped:
            parts.append(", ".join(deduped))

    # 4. Projects: Real Context and Technical Skills Used
    proj_entries = []
    project_tech = set()
    project_names = []
    for proj in (profile.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        p_name = (proj.get("name") or "").strip()
        if p_name:
            project_names.append(p_name)
        p_desc = (proj.get("description") or "").strip()
        p_techs = [t.strip() for t in (proj.get("technologies") or []) if t and str(t).strip()]
        for t in p_techs:
            project_tech.add(t)

        joined_p_techs = ", ".join(p_techs)
        tech_clause = f"(using {joined_p_techs})" if p_techs else ""
        p_components = []
        if p_name:
            p_components.append(p_name)
        if tech_clause:
            p_components.append(tech_clause)
        if p_desc:
            p_components.append(f": {p_desc}")
        if p_components:
            proj_entries.append(" ".join(p_components))

    if proj_entries:
        parts.append("Projects: " + "; ".join(proj_entries))
    elif project_names:
        parts.append("Projects: " + ", ".join(project_names))
    if project_tech:
        parts.append("Technologies: " + ", ".join(sorted(project_tech)))

    # 5. Work Experience: Real Context and Technical Skills Used
    exp_entries = []
    exp_tech = set()
    exp_items = []
    experience_list = profile.get("work_experience") or profile.get("experience") or []
    for exp in experience_list:
        if not isinstance(exp, dict):
            continue
        title = (exp.get("title") or "").strip()
        company = (exp.get("company") or "").strip()
        item = f"{title} at {company}".strip(" at")
        if item:
            exp_items.append(item)

        desc_items = exp.get("description") or []
        if isinstance(desc_items, list):
            desc_str = " ".join(str(d).strip() for d in desc_items if d and str(d).strip())
        else:
            desc_str = str(desc_items).strip()

        e_techs = [t.strip() for t in (exp.get("technologies") or []) if t and str(t).strip()]
        for t in e_techs:
            exp_tech.add(t)

        joined_e_techs = ", ".join(e_techs)
        tech_clause = f"(using {joined_e_techs})" if e_techs else ""
        e_components = []
        if item:
            e_components.append(item)
        if tech_clause:
            e_components.append(tech_clause)
        if desc_str:
            e_components.append(f": {desc_str}")
        if e_components:
            exp_entries.append(" ".join(e_components))

    if exp_entries:
        parts.append("Experience: " + "; ".join(exp_entries))
    elif exp_items:
        parts.append("Experience: " + "; ".join(exp_items))

    new_work_tech = exp_tech - project_tech
    if new_work_tech:
        parts.append("Work Tech: " + ", ".join(sorted(new_work_tech)))

    # 6. Education
    edu_items = []
    for edu in (profile.get("education") or []):
        if not isinstance(edu, dict):
            continue
        deg = (edu.get("degree") or "").strip()
        spec = (edu.get("specialization") or "").strip()
        inst = (edu.get("institution") or "").strip()
        deg_spec = " in ".join(x for x in (deg, spec) if x)
        entry = f"{deg_spec} ({inst})" if deg_spec and inst else (deg_spec or inst)
        if entry:
            edu_items.append(entry)
    if edu_items:
        parts.append("Education: " + "; ".join(edu_items))

    # 7. Certifications (if present)
    certs = [c.get("name").strip() for c in (profile.get("certifications") or []) if isinstance(c, dict) and c.get("name")]
    if certs:
        parts.append("Certifications: " + ", ".join(certs))

    # 8. Achievements (if present)
    achievements = [a.strip() for a in (profile.get("achievements") or []) if a and isinstance(a, str) and a.strip()]
    if achievements:
        parts.append("Achievements: " + ", ".join(achievements))

    return " | ".join(x for x in parts if x)


# ---------- Main function (called by the loader flow) ----------
def parse_resume(resume_text: str) -> dict:
    """
    Parse extracted resume text into structured candidate information.
    Strictly excludes phone number, email, linkedin, and github.
    Contains: name, target_role, education, technical_skills (and skills),
    work_experience (and experience), and projects, with rich real-world context
    and technical skills used for semantic job search.

    Args:
        resume_text: The plain text string extracted from a resume document
        (use doc["text"] from loader.load_document).

    Returns:
        dict: The validated profile with keys: name, target_role, education,
        technical_skills (and skills), work_experience (and experience), projects.
        Strictly excludes phone number, email, linkedin, github.
    """
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("resume_text must be a non-empty string.")

    logger.debug("Received resume text of length: %d characters", len(resume_text))

    # cache key includes model + prompt, so changing either triggers a fresh call
    key_source = f"{MODEL_NAME}\n{SYSTEM_PROMPT}\n{resume_text}"
    cache_key = hashlib.md5(key_source.encode()).hexdigest()

    # 1. Fast in-memory lookup (sub-millisecond, zero I/O)
    if cache_key in _MEM_CACHE:
        return _MEM_CACHE[cache_key]

    # 2. Disk cache lookup
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists() and cache_file.stat().st_size > 0:
        try:
            logger.debug("Cache hit: %s", cache_file.name)
            raw_cache = None
            for enc in ("utf-8", "cp1252", "latin-1"):
                try:
                    raw_cache = cache_file.read_text(encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if raw_cache is None:
                raw_cache = cache_file.read_text(encoding="utf-8", errors="replace")

            if raw_cache and raw_cache.strip():
                data = ResumeProfile.model_validate_json(raw_cache).model_dump()
                # Strictly remove contact details (contact, phone, email, linkedin, github, location)
                for contact_key in ("contact", "email", "phone", "location", "linkedin", "github", "portfolio"):
                    data.pop(contact_key, None)
                t_role = data.get("target_role")
                if t_role and str(t_role).strip().lower() in ("none", "null", "not specified", "target role not specified", "n/a", ""):
                    t_role = None
                data["target_role"] = t_role
                skills = data.get("technical_skills") or data.get("skills") or []
                data["technical_skills"] = skills
                data["skills"] = skills
                exp = data.get("work_experience") or data.get("experience") or []
                data["work_experience"] = exp
                data["experience"] = exp
                _MEM_CACHE[cache_key] = data
                return data
        except Exception as cache_err:
            logger.warning("Corrupt or invalid cache file %s (%s). Re-parsing...", cache_file.name, cache_err)
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        profile = _call_llm_with_retry(resume_text)
    except (ValidationError, ValueError) as e:
        logger.warning("First attempt invalid, retrying once: %s", e)
        profile = _call_llm_with_retry(resume_text, error=str(e))

    # Strip contact information to guarantee no phone, email, linkedin, or github
    profile.contact = None

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    result = profile.model_dump()
    # Strictly remove contact details (contact, phone, email, linkedin, github, location)
    for contact_key in ("contact", "email", "phone", "location", "linkedin", "github", "portfolio"):
        result.pop(contact_key, None)
    t_role = result.get("target_role")
    if t_role and str(t_role).strip().lower() in ("none", "null", "not specified", "target role not specified", "n/a", ""):
        t_role = None
    result["target_role"] = t_role
    skills = result.get("technical_skills") or result.get("skills") or []
    result["technical_skills"] = skills
    result["skills"] = skills
    exp = result.get("work_experience") or result.get("experience") or []
    result["work_experience"] = exp
    result["experience"] = exp
    _MEM_CACHE[cache_key] = result
    return result


# ---------- Quick test ----------
if __name__ == "__main__":
    import json
    from src.parsing.loader import load_document

    doc = load_document("tests/fixtures/sample_resume.pdf")
    result = parse_resume(doc["text"])        # pass only the text
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nProfile text:", build_profile_text(result))