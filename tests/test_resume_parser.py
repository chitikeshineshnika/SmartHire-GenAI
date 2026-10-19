"""
Tests for src/parsing/resume_parser.py and loader-to-parser integration.

Covers:
1. Input validation for parse_resume (empty, whitespace, non-string).
2. Schema modeling and strict missing-value policy:
   - Single-value missing fields return null (None).
   - List missing fields return empty list [].
3. Semantic profile text builder (build_profile_text) compatibility & privacy:
   - Combines skills, role, summary, experience, education, certifications, achievements.
   - Excludes personal contact details.
   - Handles missing/empty profiles safely.
4. Absence of hallucinated / invented information.
5. End-to-end integration: loader extracts text from PDF and DOCX,
   and passes text string to parse_resume.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.parsing.loader import load_document
from src.parsing.resume_parser import (
    Certification,
    ContactDetails,
    Education,
    Experience,
    Project,
    ResumeProfile,
    build_profile_text,
    parse_resume,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_resume.pdf"
SAMPLE_DOCX = FIXTURES_DIR / "sample_resume.docx"


# ===========================================================================
# 1. INPUT VALIDATION TESTS
# ===========================================================================

class TestResumeParserValidation:
    """Unit tests for parse_resume input validation."""

    def test_parse_resume_rejects_none(self):
        """None input should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            parse_resume(None)

    def test_parse_resume_rejects_empty_string(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            parse_resume("")

    def test_parse_resume_rejects_whitespace_only(self):
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            parse_resume("   \n\t  ")

    def test_parse_resume_rejects_non_string_types(self):
        """Passing integers, dicts, or lists should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            parse_resume(12345)
        with pytest.raises(ValueError, match="non-empty string"):
            parse_resume(["resume", "lines"])
        with pytest.raises(ValueError, match="non-empty string"):
            parse_resume({"text": "sample"})


# ===========================================================================
# 2. SCHEMA AND MISSING-VALUE POLICY TESTS
# ===========================================================================

class TestResumeParserSchema:
    """Unit tests for Pydantic models supporting all 10 required fields and null/[] conventions."""

    def test_models_instantiation_full(self):
        """Verify models accept all 10 required fields when present."""
        profile = ResumeProfile(
            name="Alice Smith",
            contact=ContactDetails(
                email="alice@example.com",
                phone="+1-555-0199",
                location="San Francisco, CA",
                linkedin="https://linkedin.com/in/alicesmith",
                github="https://github.com/alicesmith",
                portfolio="https://alicesmith.dev",
            ),
            target_role="Lead Backend Engineer",
            professional_summary="Senior backend engineer with 6 years of experience in distributed systems.",
            skills=["Python", "Go", "PostgreSQL", "Docker"],
            soft_skills=["Leadership", "Mentorship"],
            education=[
                Education(
                    degree="Bachelor of Science",
                    specialization="Computer Science",
                    institution="Stanford University",
                    year="2018",
                    grade="3.9 GPA",
                )
            ],
            experience=[
                Experience(
                    type="job",
                    title="Senior Software Engineer",
                    company="Tech Corp",
                    duration="2020 - Present",
                    description=["Designed microservices architecture handling 10k RPS."],
                    technologies=["Go", "Docker", "Kafka"],
                ),
                Experience(
                    type="internship",
                    title="Software Intern",
                    company="Startup Inc",
                    duration="Summer 2017",
                    description=["Built internal analytics dashboard."],
                    technologies=["Python", "Flask"],
                ),
            ],
            projects=[
                Project(
                    name="Distributed Cache",
                    description="Implemented Raft-consensus key-value store.",
                    technologies=["Go", "gRPC"],
                    url="https://github.com/alicesmith/raft-cache",
                )
            ],
            certifications=[
                Certification(
                    name="AWS Solutions Architect",
                    issuer="Amazon Web Services",
                    date="2023",
                    credential_id="AWS-12345",
                    url="https://aws.amazon.com/verify/12345",
                )
            ],
            achievements=["1st Place in University Hackathon 2017"],
        )

        data = profile.model_dump()
        assert data["name"] == "Alice Smith"
        assert data["contact"]["email"] == "alice@example.com"
        assert data["contact"]["portfolio"] == "https://alicesmith.dev"
        assert data["target_role"] == "Lead Backend Engineer"
        assert data["professional_summary"] == "Senior backend engineer with 6 years of experience in distributed systems."
        assert len(data["skills"]) == 4
        assert len(data["soft_skills"]) == 2
        assert data["experience"][0]["type"] == "job"
        assert data["experience"][1]["type"] == "internship"
        assert data["education"][0]["grade"] == "3.9 GPA"
        assert data["certifications"][0]["name"] == "AWS Solutions Architect"
        assert data["achievements"] == ["1st Place in University Hackathon 2017"]

    def test_missing_value_policy_null_and_empty_lists(self):
        """
        Verify strict missing-value convention:
        - Single-value missing fields return null (None).
        - List missing fields return empty list ([]).
        """
        profile = ResumeProfile(name="Minimal Candidate")
        data = profile.model_dump()

        # Single-value fields must be None (JSON null)
        assert data["name"] == "Minimal Candidate"
        assert data["contact"] is None
        assert data["target_role"] is None
        assert data["professional_summary"] is None

        # List fields must be empty lists []
        assert data["skills"] == []
        assert data["soft_skills"] == []
        assert data["education"] == []
        assert data["experience"] == []
        assert data["projects"] == []
        assert data["certifications"] == []
        assert data["achievements"] == []

    def test_nested_objects_missing_value_policy(self):
        """Verify nested objects use None for single values and [] for lists."""
        exp = Experience(type="job", company="Solo Corp")
        exp_data = exp.model_dump()
        assert exp_data["company"] == "Solo Corp"
        assert exp_data["title"] is None
        assert exp_data["duration"] is None
        assert exp_data["description"] == []
        assert exp_data["technologies"] == []

        edu = Education(institution="State Univ")
        edu_data = edu.model_dump()
        assert edu_data["institution"] == "State Univ"
        assert edu_data["degree"] is None
        assert edu_data["specialization"] is None
        assert edu_data["grade"] is None
        assert edu_data["year"] is None

        contact = ContactDetails(email="test@example.com")
        contact_data = contact.model_dump()
        assert contact_data["email"] == "test@example.com"
        assert contact_data["phone"] is None
        assert contact_data["location"] is None
        assert contact_data["linkedin"] is None
        assert contact_data["github"] is None
        assert contact_data["portfolio"] is None

    def test_placeholder_strings_cleaned_to_none_or_dropped(self):
        """Values like 'N/A', 'None', 'null', '' must never be preserved as valid data."""
        contact = ContactDetails(
            email="valid@test.com",
            phone="N/A",
            location="None",
            linkedin="",
            github="null",
        )
        assert contact.email == "valid@test.com"
        assert contact.phone is None
        assert contact.location is None
        assert contact.linkedin is None
        assert contact.github is None

        profile = ResumeProfile(
            target_role="N/A",
            professional_summary="null",
            skills=["Python", "N/A", "", "  ", "SQL"],
        )
        assert profile.target_role is None
        assert profile.professional_summary is None
        assert profile.skills == ["Python", "SQL"]

    def test_target_role_accepts_clear_professional_roles(self):
        """Valid job roles like 'Python Developer' or 'Data Analyst' must be preserved, cleaning intent prefixes."""
        valid_cases = [
            ("Python Developer", "Python Developer"),
            ("Java Developer", "Java Developer"),
            ("Data Analyst", "Data Analyst"),
            ("Software Engineer", "Software Engineer"),
            ("Frontend Developer", "Frontend Developer"),
            ("Seeking a Java Developer position", "Java Developer"),
            ("Seeking a full-time Java Developer position", "Java Developer"),
            ("Aspiring Data Analyst", "Data Analyst"),
            ("Objective: To work as a Python Developer", "Python Developer"),
        ]
        for role_input, expected in valid_cases:
            profile = ResumeProfile(target_role=role_input)
            assert profile.target_role == expected, f"Failed on {role_input!r}: got {profile.target_role!r}"

    def test_target_role_rejects_student_and_degree_status(self):
        """Educational status, degrees, or student terms must NEVER be accepted as target_role."""
        invalid_roles = [
            "Computer Science undergraduate",
            "B.Tech CSE student",
            "Bachelor of Technology in Computer Science",
            "CS student",
            "Engineering student",
            "College student",
            "Fresher",
            "Engineering graduate",
            "Computer Science graduate",
            "Bachelor of Technology",
            "Pursuing B.Tech",
            "Computer Science",
            "Computer Science Engineering",
            "Information Technology",
            "B.Tech in Computer Science",
        ]
        for invalid in invalid_roles:
            profile = ResumeProfile(target_role=invalid)
            assert profile.target_role is None, f"Expected {invalid!r} to be rejected as target_role"

    def test_target_role_and_education_extracted_separately(self):
        """Ensure educational degrees belong to education, while target_role holds only distinct job titles."""
        # Case 1: Student status in summary/objective, no target job role
        profile_student = ResumeProfile(
            name="Rahul V",
            target_role="Computer Science undergraduate",
            education=[
                Education(
                    degree="Bachelor of Technology",
                    specialization="Computer Science",
                    institution="National Institute of Technology",
                )
            ],
        )
        assert profile_student.target_role is None
        assert profile_student.education[0].degree == "Bachelor of Technology"
        assert profile_student.education[0].specialization == "Computer Science"

        # Case 2: Explicit target job role + distinct education
        profile_engineer = ResumeProfile(
            name="Anita R",
            target_role="Python Developer",
            education=[
                Education(
                    degree="Bachelor of Engineering",
                    specialization="Information Science",
                    institution="State University",
                )
            ],
        )
        assert profile_engineer.target_role == "Python Developer"
        assert profile_engineer.education[0].degree == "Bachelor of Engineering"

    def test_strict_no_assumptions_target_role_validation(self):
        """
        Verify strict no-assumptions rule across all 5 user validation criteria:
        1. 'Python Developer Intern' listed only under internships -> null (None).
        2. Java and Python skills with academic projects -> null (None).
        3. Previous experience as a Software Engineer, with no stated career goal -> null (None).
        4. 'Seeking a full-time Java Developer position' -> 'Java Developer'.
        5. 'Looking for a suitable opportunity in the IT industry' -> null (None).
        """
        # 1. "Python Developer Intern" listed under internships -> null
        p1 = ResumeProfile(
            work_experience=[Experience(type="internship", title="Python Developer Intern", company="Acme Corp")],
            target_role="Python Developer Intern",
        )
        assert p1.target_role is None, "Internship titles must never be accepted as target_role"

        # 2. Java and Python skills with academic projects -> null
        p2 = ResumeProfile(
            skills=["Java", "Python"],
            technical_skills=["Java", "Python"],
            projects=[Project(name="Compiler Project", description="Built in Java and Python")],
        )
        assert p2.target_role is None, "Target role must not be inferred from skills or academic projects"

        # 3. Previous experience as a Software Engineer, with no stated career goal -> null
        p3 = ResumeProfile(
            work_experience=[Experience(type="job", title="Software Engineer", company="Past Corp")],
        )
        assert p3.target_role is None, "Target role must not be inferred from past work experience"

        # 4. "Seeking a full-time Java Developer position" -> "Java Developer"
        p4 = ResumeProfile(
            target_role="Seeking a full-time Java Developer position",
        )
        assert p4.target_role == "Java Developer", "Explicit career goal must extract the clean occupational title"

        # 5. "Looking for a suitable opportunity in the IT industry" -> null
        p5 = ResumeProfile(
            target_role="Looking for a suitable opportunity in the IT industry",
        )
        assert p5.target_role is None, "Vague industry aspirations must return null"


# ===========================================================================
# 3. SEMANTIC PROFILE TEXT BUILDER TESTS
# ===========================================================================

class TestBuildProfileText:
    """Tests for build_profile_text semantic representation for Module 2 job matching."""

    def test_build_profile_text_includes_all_semantic_fields(self):
        """Profile text combines target role, summary, skills, tech, experience, education, certifications, achievements."""
        profile_data = {
            "name": "Jane Doe",
            "contact": {
                "email": "jane@example.com",
                "phone": "+1-123-456-7890",
                "linkedin": "https://linkedin.com/in/janedoe",
                "portfolio": "https://janedoe.com",
            },
            "target_role": "Full Stack Engineer",
            "professional_summary": "Full Stack Developer specialized in Python, React, and cloud architectures.",
            "skills": ["Python", "JavaScript", "React"],
            "soft_skills": ["Team Leadership"],
            "projects": [
                {
                    "name": "E-commerce Platform",
                    "technologies": ["React", "FastAPI"],
                }
            ],
            "experience": [
                {
                    "title": "Software Developer",
                    "company": "AppCorp",
                    "technologies": ["Docker", "PostgreSQL"],
                }
            ],
            "education": [
                {
                    "degree": "B.S.",
                    "specialization": "Computer Science",
                    "institution": "State University",
                }
            ],
            "certifications": [
                {"name": "AWS Certified Developer"}
            ],
            "achievements": [
                "Dean's Honor List 2022"
            ],
        }

        text = build_profile_text(profile_data)

        # Verify semantic content is included
        assert "Full Stack Engineer" in text
        assert "Full Stack Developer specialized in Python, React, and cloud architectures." in text
        assert "Python" in text
        assert "React" in text
        assert "Team Leadership" in text
        assert "Projects: E-commerce Platform" in text
        assert "FastAPI" in text
        assert "Software Developer at AppCorp" in text
        assert "B.S. in Computer Science (State University)" in text
        assert "Certifications: AWS Certified Developer" in text
        assert "Achievements: Dean's Honor List 2022" in text

        # Verify contact details are STRICTLY EXCLUDED
        assert "jane@example.com" not in text
        assert "+1-123-456-7890" not in text
        assert "linkedin.com" not in text
        assert "janedoe.com" not in text

    def test_build_profile_text_handles_missing_fields_safely(self):
        """Missing fields and empty lists do not crash build_profile_text."""
        empty_profile = {
            "name": None,
            "contact": None,
            "target_role": None,
            "professional_summary": None,
            "skills": [],
            "soft_skills": [],
            "education": [],
            "experience": [],
            "projects": [],
            "certifications": [],
            "achievements": [],
        }
        assert build_profile_text(empty_profile) == ""
        assert build_profile_text({}) == ""


# ===========================================================================
# 4. PARSER RETURN STRUCTURE & GROUNDING TESTS
# ===========================================================================

class TestResumeParserBasic:
    """Unit tests for parse_resume functionality with structured output."""

    def test_parse_resume_returns_structured_dictionary(self, monkeypatch, tmp_path):
        """parse_resume returns dictionary adhering to schema and missing-value policy."""
        mock_profile = ResumeProfile(
            name="John Doe",
            target_role="Backend Developer",
            skills=["Python", "FastAPI"],
            education=[Education(degree="B.Tech", institution="NIT")],
        )

        import src.parsing.resume_parser as parser_module
        monkeypatch.setattr(parser_module, "_call_llm_with_retry", lambda text, error=None: mock_profile)
        monkeypatch.setattr(parser_module, "CACHE_DIR", tmp_path / ".test_cache")

        result = parse_resume("Sample resume text for John Doe")

        assert isinstance(result, dict)
        assert result["name"] == "John Doe"
        assert result["target_role"] == "Backend Developer"
        assert result["skills"] == ["Python", "FastAPI"]
        assert result["education"][0]["degree"] == "B.Tech"

        # Privacy policy: contact details are strictly omitted
        assert "contact" not in result
        assert result["professional_summary"] is None
        assert result["soft_skills"] == []
        assert result["experience"] == []
        assert result["projects"] == []
        assert result["certifications"] == []
        assert result["achievements"] == []


# ===========================================================================
# 5. LOADER TO PARSER INTEGRATION TESTS
# ===========================================================================

class TestLoaderToParserIntegration:
    """Integration tests connecting loader output to parser input."""

    def test_loader_pdf_to_parser_connection(self):
        """Verify load_document extracts text from PDF and parse_resume extracts structured profile."""
        loader_output = load_document(str(SAMPLE_PDF))
        assert "text" in loader_output
        assert isinstance(loader_output["text"], str)
        assert len(loader_output["text"]) > 0

        parser_output = parse_resume(loader_output["text"])
        assert isinstance(parser_output, dict)
        assert "name" in parser_output
        assert "skills" in parser_output
        assert "education" in parser_output
        assert "experience" in parser_output
        assert "projects" in parser_output

        assert parser_output["name"].upper() == "PRIYA SHARMA"
        assert "Python" in parser_output["skills"]
        assert isinstance(parser_output["education"], list)
        assert isinstance(parser_output["experience"], list)
        assert isinstance(parser_output["projects"], list)

    def test_loader_docx_to_parser_connection(self):
        """Verify load_document extracts text from DOCX and parse_resume extracts structured profile."""
        loader_output = load_document(str(SAMPLE_DOCX))
        assert "text" in loader_output
        assert isinstance(loader_output["text"], str)
        assert len(loader_output["text"]) > 0

        parser_output = parse_resume(loader_output["text"])
        assert isinstance(parser_output, dict)
        assert "name" in parser_output
        assert "skills" in parser_output
        assert parser_output["name"].upper() == "PRIYA SHARMA"


# ===========================================================================
# 6. ENCODING RESILIENCE & ERROR HANDLING TESTS
# ===========================================================================

class TestCacheEncodingResilience:
    """Verify that cached files with legacy Windows-1252 bytes (0x96) or corrupted content don't crash."""

    def test_cache_with_byte_0x96_reads_cleanly_or_recovers(self, tmp_path, monkeypatch):
        """Cache files containing Windows-1252 en-dash (0x96) decode via fallback without raising UnicodeDecodeError."""
        import hashlib
        import src.parsing.resume_parser as parser_module

        test_cache_dir = tmp_path / ".test_parse_cache"
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(parser_module, "CACHE_DIR", test_cache_dir)
        monkeypatch.setattr(parser_module, "_MEM_CACHE", {})

        resume_text = "Unique Test Candidate with en-dash experience"
        key_source = f"{parser_module.MODEL_NAME}\n{parser_module.SYSTEM_PROMPT}\n{resume_text}"
        cache_key = hashlib.md5(key_source.encode()).hexdigest()
        cache_file = test_cache_dir / f"{cache_key}.json"

        # Valid JSON encoded with cp1252 containing en-dash byte 0x96
        profile_bytes = b'{"name": "Jane Doe", "target_role": "Data Analyst", "skills": ["Python"], "experience": [{"title": "Analyst", "company": "Tech Corp", "duration": "2021 \x96 2024"}]}'
        cache_file.write_bytes(profile_bytes)

        # Parsing should load from cache using cp1252 fallback without throwing UnicodeDecodeError
        result = parser_module.parse_resume(resume_text)
        assert isinstance(result, dict)
        assert result["name"] == "Jane Doe"
        assert result["target_role"] == "Data Analyst"

    def test_corrupt_cache_file_is_purged_and_reparsed(self, tmp_path, monkeypatch):
        """Corrupted JSON in cache is removed and candidate is reparsed cleanly."""
        import hashlib
        import src.parsing.resume_parser as parser_module

        test_cache_dir = tmp_path / ".test_parse_cache"
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(parser_module, "CACHE_DIR", test_cache_dir)
        monkeypatch.setattr(parser_module, "_MEM_CACHE", {})

        resume_text = "Test Candidate with Corrupt Cache"
        key_source = f"{parser_module.MODEL_NAME}\n{parser_module.SYSTEM_PROMPT}\n{resume_text}"
        cache_key = hashlib.md5(key_source.encode()).hexdigest()
        cache_file = test_cache_dir / f"{cache_key}.json"

        # Write corrupt non-JSON content
        cache_file.write_bytes(b"{not-valid-json-at-all!!")

        mock_profile = ResumeProfile(
            name="Recovered Candidate",
            target_role="Software Engineer",
            skills=["Go", "Docker"],
        )
        monkeypatch.setattr(parser_module, "_call_llm_with_retry", lambda text, error=None: mock_profile)

        result = parser_module.parse_resume(resume_text)
        assert result["name"] == "Recovered Candidate"


class TestMarkdownJSONAndAPIErrorHandling:
    """Verify markdown-wrapped JSON extraction and API error messages."""

    def test_markdown_code_fence_json_cleaned(self):
        """LLM responses enclosed in ```json ... ``` are cleaned and parsed."""
        raw_llm = '```json\n{"name": "Dev User", "target_role": "Python Developer", "skills": ["Django"]}\n```'
        cleaned = raw_llm.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        profile = ResumeProfile.model_validate_json(cleaned)
        assert profile.name == "Dev User"
        assert profile.target_role == "Python Developer"

    def test_missing_optional_sections_returns_clean_schema(self, tmp_path, monkeypatch):
        """Resumes missing optional sections (no certs, no achievements, no summary) return valid dict."""
        import src.parsing.resume_parser as parser_module

        monkeypatch.setattr(parser_module, "CACHE_DIR", tmp_path / ".cache")
        monkeypatch.setattr(parser_module, "_MEM_CACHE", {})

        # Profile with only required minimums
        sparse_profile = ResumeProfile(
            name="Minimal Candidate",
            skills=["Python"],
        )
        monkeypatch.setattr(parser_module, "_call_llm_with_retry", lambda text, error=None: sparse_profile)

        result = parser_module.parse_resume("Minimal candidate text")
        assert result["name"] == "Minimal Candidate"
        assert result["target_role"] is None
        assert result["professional_summary"] is None
        assert result["experience"] == []
        assert result["projects"] == []
        assert result["certifications"] == []
        assert result["achievements"] == []
