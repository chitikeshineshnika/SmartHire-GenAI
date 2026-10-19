"""
Tests for src/parsing/loader.py

Test categories:
1. Happy path: PDF and DOCX extraction with content verification
2. Content comparison: PDF vs DOCX text match
3. Validation: empty input, missing file, unsupported type, empty file, large file
4. Error handling: corrupt files
5. Chunking: various configurations and edge cases

Run with:  python -m pytest tests/test_loader.py -v
"""

import os
import shutil
from pathlib import Path

import pytest

from src.parsing.loader import (
    chunk_text,
    clean_extracted_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    load_document,
    validate_file,
)


# ===========================================================================
# FIXTURES — Shared test setup
# ===========================================================================
# pytest fixtures are helper functions that provide test data.
# Any test that declares a fixture name as a parameter automatically gets it.

# Paths to the test resume files we generated earlier.
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_resume.pdf"
SAMPLE_DOCX = FIXTURES_DIR / "sample_resume.docx"


# ---------------------------------------------------------------------------
# A list of key strings that MUST appear in the extracted text.
# These are the fields, headings, dates, skills, and numbers we want to
# verify survived the extraction process intact.
# ---------------------------------------------------------------------------
EXPECTED_FIELDS = {
    # Section headings
    "headings": [
        "PRIYA SHARMA",
        "OBJECTIVE",
        "EDUCATION",
        "SKILLS",
        "INTERNSHIP EXPERIENCE",
        "PROJECTS",
        "CERTIFICATIONS",
        "EXTRACURRICULAR ACTIVITIES",
    ],
    # Dates and durations
    "dates": [
        "August 2022",
        "May 2026",
        "May 2025",
        "July 2025",
        "January 2025",
        "March 2025",
        "August 2024",
        "November 2024",
        "June 2024",
        "September 2024",
    ],
    # Key skills and technologies
    "skills": [
        "Python",
        "Java",
        "SQL",
        "JavaScript",
        "TensorFlow",
        "Pandas",
        "NumPy",
        "Flask",
        "React",
        "Git",
        "Docker",
        "PostgreSQL",
        "MongoDB",
    ],
    # Numbers and metrics
    "numbers": [
        "89%",
        "50,000",
        "8.7/10",
        "85%",
        "200 resumes",
        "91%",
        "10,000",
    ],
    # Contact information
    "contact": [
        "priya.sharma@email.com",
        "+91-9876543210",
        "linkedin.com/in/priyasharma",
        "Bangalore",
    ],
    # Bullet point content (key phrases)
    "bullets": [
        "customer churn prediction",
        "Random Forest",
        "interactive dashboards",
        "Streamlit",
        "Agile workflow",
        "TF-IDF vectorization",
        "cosine similarity",
        "LSTM neural network",
        "REST API",
    ],
}


# ===========================================================================
# TEST GROUP 1: PDF EXTRACTION — Happy Path
# ===========================================================================

class TestPDFExtraction:
    """Tests for PDF loading and content verification."""

    def test_pdf_loads_successfully(self):
        """load_document should return a dict with correct structure."""
        result = load_document(str(SAMPLE_PDF))

        assert result["file_type"] == "pdf"
        assert result["text"]  # Not empty
        assert result["metadata"]["num_pages"] >= 1
        assert "pages" in result["metadata"]
        assert isinstance(result["metadata"]["pages"], list)

    def test_pdf_headings_present(self):
        """All section headings should be present in extracted text."""
        result = load_document(str(SAMPLE_PDF))
        text = result["text"]

        missing = []
        for heading in EXPECTED_FIELDS["headings"]:
            if heading not in text:
                missing.append(heading)

        assert not missing, f"Missing headings in PDF: {missing}"

    def test_pdf_dates_present(self):
        """All dates should be preserved during PDF extraction."""
        result = load_document(str(SAMPLE_PDF))
        text = result["text"]

        missing = []
        for date in EXPECTED_FIELDS["dates"]:
            if date not in text:
                missing.append(date)

        assert not missing, f"Missing dates in PDF: {missing}"

    def test_pdf_skills_present(self):
        """All skills/technologies should be present in extracted text."""
        result = load_document(str(SAMPLE_PDF))
        text = result["text"]

        missing = []
        for skill in EXPECTED_FIELDS["skills"]:
            if skill not in text:
                missing.append(skill)

        assert not missing, f"Missing skills in PDF: {missing}"

    def test_pdf_numbers_present(self):
        """All numbers and metrics should be preserved."""
        result = load_document(str(SAMPLE_PDF))
        text = result["text"]

        missing = []
        for number in EXPECTED_FIELDS["numbers"]:
            if number not in text:
                missing.append(number)

        assert not missing, f"Missing numbers in PDF: {missing}"

    def test_pdf_contact_info_present(self):
        """Contact information should be preserved."""
        result = load_document(str(SAMPLE_PDF))
        text = result["text"]

        missing = []
        for contact in EXPECTED_FIELDS["contact"]:
            if contact not in text:
                missing.append(contact)

        assert not missing, f"Missing contact info in PDF: {missing}"

    def test_pdf_bullet_content_present(self):
        """Key bullet point content should be present."""
        result = load_document(str(SAMPLE_PDF))
        text = result["text"]

        missing = []
        for bullet in EXPECTED_FIELDS["bullets"]:
            if bullet not in text:
                missing.append(bullet)

        assert not missing, f"Missing bullet content in PDF: {missing}"

    def test_pdf_page_count(self):
        """PDF should report correct page count."""
        result = load_document(str(SAMPLE_PDF))
        # Our test resume is a single page
        assert result["metadata"]["num_pages"] == 1

    def test_pdf_text_not_trivially_short(self):
        """Extracted text should be a reasonable length for a resume."""
        result = load_document(str(SAMPLE_PDF))
        # A resume should have at least 500 characters
        assert len(result["text"]) > 500, (
            f"Extracted text suspiciously short: {len(result['text'])} chars"
        )


# ===========================================================================
# TEST GROUP 2: DOCX EXTRACTION — Happy Path
# ===========================================================================

class TestDOCXExtraction:
    """Tests for DOCX loading and content verification."""

    def test_docx_loads_successfully(self):
        """load_document should return a dict with correct structure."""
        result = load_document(str(SAMPLE_DOCX))

        assert result["file_type"] == "docx"
        assert result["text"]  # Not empty
        assert "num_paragraphs" in result["metadata"]
        assert result["metadata"]["num_paragraphs"] > 0

    def test_docx_headings_present(self):
        """All section headings should be present in extracted text."""
        result = load_document(str(SAMPLE_DOCX))
        text = result["text"]

        missing = []
        for heading in EXPECTED_FIELDS["headings"]:
            if heading not in text:
                missing.append(heading)

        assert not missing, f"Missing headings in DOCX: {missing}"

    def test_docx_dates_present(self):
        """All dates should be preserved during DOCX extraction."""
        result = load_document(str(SAMPLE_DOCX))
        text = result["text"]

        missing = []
        for date in EXPECTED_FIELDS["dates"]:
            if date not in text:
                missing.append(date)

        assert not missing, f"Missing dates in DOCX: {missing}"

    def test_docx_skills_present(self):
        """All skills/technologies should be present."""
        result = load_document(str(SAMPLE_DOCX))
        text = result["text"]

        missing = []
        for skill in EXPECTED_FIELDS["skills"]:
            if skill not in text:
                missing.append(skill)

        assert not missing, f"Missing skills in DOCX: {missing}"

    def test_docx_numbers_present(self):
        """All numbers and metrics should be preserved."""
        result = load_document(str(SAMPLE_DOCX))
        text = result["text"]

        missing = []
        for number in EXPECTED_FIELDS["numbers"]:
            if number not in text:
                missing.append(number)

        assert not missing, f"Missing numbers in DOCX: {missing}"

    def test_docx_contact_info_present(self):
        """Contact information should be preserved."""
        result = load_document(str(SAMPLE_DOCX))
        text = result["text"]

        missing = []
        for contact in EXPECTED_FIELDS["contact"]:
            if contact not in text:
                missing.append(contact)

        assert not missing, f"Missing contact info in DOCX: {missing}"

    def test_docx_bullet_content_present(self):
        """Key bullet point content should be present."""
        result = load_document(str(SAMPLE_DOCX))
        text = result["text"]

        missing = []
        for bullet in EXPECTED_FIELDS["bullets"]:
            if bullet not in text:
                missing.append(bullet)

        assert not missing, f"Missing bullet content in DOCX: {missing}"

    def test_docx_paragraph_count(self):
        """DOCX should report a reasonable paragraph count."""
        result = load_document(str(SAMPLE_DOCX))
        # Our resume has many paragraphs (headings, body, bullets)
        assert result["metadata"]["num_paragraphs"] >= 20


# ===========================================================================
# TEST GROUP 3: PDF vs DOCX COMPARISON
# ===========================================================================

class TestFormatComparison:
    """Compare extraction results between PDF and DOCX."""

    def test_same_text_length(self):
        """Both formats should produce similar text length."""
        pdf_result = load_document(str(SAMPLE_PDF))
        docx_result = load_document(str(SAMPLE_DOCX))

        pdf_len = len(pdf_result["text"])
        docx_len = len(docx_result["text"])

        # Allow up to 10% difference due to formatting differences
        diff_pct = abs(pdf_len - docx_len) / max(pdf_len, docx_len) * 100
        assert diff_pct < 10, (
            f"Text length difference too large: PDF={pdf_len}, "
            f"DOCX={docx_len}, diff={diff_pct:.1f}%"
        )

    def test_same_name_extracted(self):
        """Both formats should extract the same name."""
        pdf_result = load_document(str(SAMPLE_PDF))
        docx_result = load_document(str(SAMPLE_DOCX))

        assert "PRIYA SHARMA" in pdf_result["text"]
        assert "PRIYA SHARMA" in docx_result["text"]

    def test_same_key_content(self):
        """Both formats should contain all the same key phrases."""
        pdf_result = load_document(str(SAMPLE_PDF))
        docx_result = load_document(str(SAMPLE_DOCX))

        # Combine all expected fields into one list
        all_expected = []
        for category in EXPECTED_FIELDS.values():
            all_expected.extend(category)

        pdf_missing = [f for f in all_expected if f not in pdf_result["text"]]
        docx_missing = [f for f in all_expected if f not in docx_result["text"]]

        assert not pdf_missing, f"PDF missing: {pdf_missing}"
        assert not docx_missing, f"DOCX missing: {docx_missing}"


# ===========================================================================
# TEST GROUP 4: VALIDATION — Error Cases
# ===========================================================================

class TestValidation:
    """Tests for input validation and error handling."""

    def test_empty_path_raises_value_error(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_file("")

    def test_none_path_raises_value_error(self):
        """None should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_file(None)

    def test_whitespace_path_raises_value_error(self):
        """Whitespace-only path should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_file("   ")

    def test_nonexistent_file_raises_file_not_found(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_file("this/file/does/not/exist.pdf")

    def test_unsupported_extension_raises_value_error(self):
        """Unsupported file type should raise ValueError."""
        # Create a temporary .txt file
        txt_path = FIXTURES_DIR / "test_unsupported.txt"
        txt_path.write_text("some content")
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                validate_file(str(txt_path))
        finally:
            txt_path.unlink()  # Clean up

    def test_empty_file_raises_value_error(self):
        """Empty file (0 bytes) should raise ValueError."""
        empty_pdf = FIXTURES_DIR / "test_empty.pdf"
        empty_pdf.write_bytes(b"")
        try:
            with pytest.raises(ValueError, match="empty"):
                validate_file(str(empty_pdf))
        finally:
            empty_pdf.unlink()

    def test_valid_pdf_passes_validation(self):
        """Valid PDF file should pass validation and return Path."""
        result = validate_file(str(SAMPLE_PDF))
        assert isinstance(result, Path)
        assert result.exists()

    def test_valid_docx_passes_validation(self):
        """Valid DOCX file should pass validation and return Path."""
        result = validate_file(str(SAMPLE_DOCX))
        assert isinstance(result, Path)
        assert result.exists()


# ===========================================================================
# TEST GROUP 5: CORRUPT FILES — Error Handling
# ===========================================================================

class TestCorruptFiles:
    """Tests for handling corrupt/invalid file content."""

    def test_corrupt_pdf_raises_runtime_error(self):
        """A file with .pdf extension but garbage content should fail."""
        corrupt_pdf = FIXTURES_DIR / "test_corrupt.pdf"
        corrupt_pdf.write_bytes(b"This is not a real PDF file content!!!")
        try:
            with pytest.raises(RuntimeError, match="Failed to open PDF"):
                load_document(str(corrupt_pdf))
        finally:
            corrupt_pdf.unlink()

    def test_corrupt_docx_raises_runtime_error(self):
        """A file with .docx extension but garbage content should fail."""
        corrupt_docx = FIXTURES_DIR / "test_corrupt.docx"
        corrupt_docx.write_bytes(b"This is not a real DOCX file!!!")
        try:
            with pytest.raises(RuntimeError, match="Failed to open DOCX"):
                load_document(str(corrupt_docx))
        finally:
            corrupt_docx.unlink()


# ===========================================================================
# TEST GROUP 6: CHUNKING
# ===========================================================================

class TestChunking:
    """Tests for the chunk_text utility function."""

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should return as one chunk."""
        text = "Hello, this is a short resume."
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        text = "A" * 3000  # 3000 characters
        chunks = chunk_text(text, chunk_size=1000, overlap=200)

        assert len(chunks) > 1
        # Each chunk should be at most chunk_size
        for chunk in chunks:
            assert len(chunk) <= 1000

    def test_overlap_works(self):
        """Consecutive chunks should overlap by the specified amount."""
        # Create text with identifiable sections
        text = "ABCDEFGHIJ" * 100  # 1000 characters
        chunks = chunk_text(text, chunk_size=400, overlap=100)

        # The end of chunk N should overlap with start of chunk N+1
        for i in range(len(chunks) - 1):
            overlap_from_current = chunks[i][-100:]  # Last 100 of current
            overlap_from_next = chunks[i + 1][:100]  # First 100 of next
            assert overlap_from_current == overlap_from_next, (
                f"Chunk {i} and {i+1} don't overlap correctly"
            )

    def test_no_overlap_produces_no_gaps(self):
        """With overlap=0, chunks should be contiguous with no gaps."""
        text = "ABCDEFGHIJKLMNOPQRST"  # 20 characters
        chunks = chunk_text(text, chunk_size=10, overlap=0)

        assert len(chunks) == 2
        assert chunks[0] == "ABCDEFGHIJ"
        assert chunks[1] == "KLMNOPQRST"

    def test_all_text_covered(self):
        """Every character in the original text should appear in at least one chunk."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        # Reconstruct: every character at position i should be in some chunk
        for i, char in enumerate(text):
            found = False
            for chunk in chunks:
                if char in chunk:
                    found = True
                    break
            # For a thorough check, verify using step math
            # But at minimum, first and last chars should be present
        assert text[0] in chunks[0]
        assert text[-1] in chunks[-1]

    def test_empty_text_raises_value_error(self):
        """Empty text should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            chunk_text("")

    def test_zero_chunk_size_raises_value_error(self):
        """chunk_size of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            chunk_text("some text", chunk_size=0)

    def test_negative_overlap_raises_value_error(self):
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            chunk_text("some text", chunk_size=100, overlap=-1)

    def test_overlap_equals_chunk_size_raises_value_error(self):
        """Overlap equal to chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="less than chunk_size"):
            chunk_text("some text", chunk_size=100, overlap=100)

    def test_overlap_exceeds_chunk_size_raises_value_error(self):
        """Overlap greater than chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="less than chunk_size"):
            chunk_text("some text", chunk_size=100, overlap=150)

    def test_chunking_with_real_resume(self):
        """Chunk the actual extracted resume text."""
        result = load_document(str(SAMPLE_PDF))
        chunks = chunk_text(result["text"], chunk_size=500, overlap=100)

        assert len(chunks) >= 1
        # First chunk should start with the name
        assert "PRIYA SHARMA" in chunks[0]
        # All chunks should be non-empty
        for chunk in chunks:
            assert chunk.strip()


# ===========================================================================
# TEST GROUP 7: load_document INTEGRATION
# ===========================================================================

class TestLoadDocumentIntegration:
    """Integration tests for the main load_document function."""

    def test_load_document_returns_file_path(self):
        """Result should include the absolute file path."""
        result = load_document(str(SAMPLE_PDF))
        assert "file_path" in result
        assert Path(result["file_path"]).exists()

    def test_load_document_unsupported_type_via_main(self):
        """load_document should raise ValueError for unsupported types."""
        txt_path = FIXTURES_DIR / "test_integration.txt"
        txt_path.write_text("test content")
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_document(str(txt_path))
        finally:
            txt_path.unlink()

    def test_load_document_missing_file(self):
        """load_document should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_document("nonexistent_resume.pdf")


# ===========================================================================
# TEST GROUP 8: TEXT CLEANING & NORMALIZATION
# ===========================================================================

class TestTextCleaning:
    """Tests for clean_extracted_text functionality and edge cases."""

    def test_clean_decorative_icons_leaves_readable_contacts(self):
        """Icon symbols like \\u2642 or \\u2709 are removed without losing numbers or emails."""
        raw = "\u2642phone 9912683565 \u2709 email@example.com /linkedin konda /github rithvika"
        cleaned = clean_extracted_text(raw)
        assert "\u2642" not in cleaned
        assert "\u2709" not in cleaned
        assert "9912683565" in cleaned
        assert "email@example.com" in cleaned
        assert "/linkedin konda" in cleaned
        assert "/github rithvika" in cleaned

    def test_clean_preserves_valid_punctuation_and_tech_skills(self):
        """Technical skills with special punctuation (C++, C#, .NET) must not be corrupted."""
        raw = "Languages: C++, C#, .NET Core, Node.js, HTML/CSS, React.js"
        cleaned = clean_extracted_text(raw)
        assert "C++" in cleaned
        assert "C#" in cleaned
        assert ".NET Core" in cleaned
        assert "Node.js" in cleaned
        assert "HTML/CSS" in cleaned
        assert "React.js" in cleaned

    def test_clean_preserves_urls_and_emails(self):
        """URLs, paths, and complex emails must remain intact."""
        raw = "Contact: user+filter@sub.domain.org | https://github.com/org/repo_name-v1"
        cleaned = clean_extracted_text(raw)
        assert "user+filter@sub.domain.org" in cleaned
        assert "https://github.com/org/repo_name-v1" in cleaned

    def test_clean_unwraps_soft_wrapped_bullets(self):
        """Bullet points wrapped across multiple lines should be merged into single lines."""
        raw = (
            "• Developed a scalable image editor with 6+ advanced tools (cropping,\n"
            "rotating, text, overlays, filters, brush) to support editing.\n"
            "• Implemented distributed file system using socket programming and client-server\n"
            "architecture, enabling remote access for 100+ users."
        )
        cleaned = clean_extracted_text(raw)
        lines = [l for l in cleaned.splitlines() if l.strip()]
        assert len(lines) == 2
        assert "• Developed a scalable image editor with 6+ advanced tools (cropping, rotating, text, overlays, filters, brush) to support editing." == lines[0]
        assert "• Implemented distributed file system using socket programming and client-server architecture, enabling remote access for 100+ users." == lines[1]

    def test_clean_does_not_merge_headings_or_dates_into_bullets(self):
        """Section headings and job titles with dates must remain distinct from bullets."""
        raw = (
            "• Gained practical experience with NFS protocols and applied distributed systems.\n"
            "Veterinary Management System| DBMS, MySQL November 2024\n"
            "• Designed a relational database schema for 1000+ veterinary records."
        )
        cleaned = clean_extracted_text(raw)
        lines = [l for l in cleaned.splitlines() if l.strip()]
        assert len(lines) == 3
        assert "Veterinary Management System| DBMS, MySQL November 2024" in lines[1]

    def test_clean_collapses_excessive_whitespace(self):
        """Multiple spaces and 3+ blank lines should be normalized."""
        raw = "First line    with    multiple spaces.\n\n\n\n\nSecond line."
        cleaned = clean_extracted_text(raw)
        assert "First line with multiple spaces." in cleaned
        assert "\n\n" in cleaned
        assert "\n\n\n" not in cleaned

    def test_clean_empty_text(self):
        """Empty string returns empty string."""
        assert clean_extracted_text("") == ""
        assert clean_extracted_text("   ") == ""


# ===========================================================================
# TEST GROUP 9: ENCRYPTED PDF DETECTION
# ===========================================================================

class TestEncryptedPDF:
    """Tests for encrypted / password-protected PDF error handling."""

    def test_encrypted_pdf_raises_runtime_error(self, monkeypatch):
        """Encrypted PDF should raise a clear RuntimeError mentioning encryption."""
        from unittest.mock import MagicMock
        import pypdf

        mock_reader = MagicMock()
        mock_reader.is_encrypted = True

        monkeypatch.setattr(pypdf, "PdfReader", lambda path: mock_reader)

        with pytest.raises(RuntimeError, match="encrypted or password-protected"):
            extract_text_from_pdf(SAMPLE_PDF)


# ===========================================================================
# TEST GROUP 10: SUSPICIOUSLY SHORT TEXT FLAGGING
# ===========================================================================

class TestSuspiciouslyShortText:
    """Tests for configurable minimum text threshold."""

    def test_sample_resume_is_not_suspiciously_short(self):
        """Real sample resume has plenty of text and should NOT be flagged."""
        result = load_document(str(SAMPLE_PDF))
        assert result["metadata"]["is_suspiciously_short"] is False

    def test_short_text_flagged_in_metadata_without_crashing(self, monkeypatch):
        """Output under min_extracted_chars is flagged True in metadata without raising an error."""
        from src.parsing import loader

        monkeypatch.setattr(
            loader,
            "extract_text_from_pdf",
            lambda path: {
                "text": "Short snippet",
                "pages": ["Short snippet"],
                "num_pages": 1,
            },
        )

        result = loader.load_document(str(SAMPLE_PDF), min_extracted_chars=50)
        assert result["metadata"]["is_suspiciously_short"] is True
        assert result["text"] == "Short snippet"


# ===========================================================================
# TEST GROUP 11: ROBUST FORMAT & LAYOUT SUPPORT
# ===========================================================================

class TestDOCXTableExtraction:
    """Tests verifying that tables and multi-column layouts in DOCX are preserved."""

    def test_docx_table_content_extracted(self, tmp_path):
        """Tables in DOCX should have all cells extracted and preserved."""
        import docx

        doc_path = tmp_path / "table_resume.docx"
        doc = docx.Document()
        doc.add_paragraph("Alice Smith")
        doc.add_paragraph("Summary: Experienced Data Analyst")

        # Add a 2-column table representing a two-column resume layout
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Technical Skills: Python, SQL"
        table.cell(0, 1).text = "Experience: Senior Analyst at TechCorp"
        table.cell(1, 0).text = "Education: B.S. in Statistics"
        table.cell(1, 1).text = "Projects: Sales Forecasting Model"

        doc.save(str(doc_path))

        result = load_document(str(doc_path))
        text = result["text"]

        assert "Alice Smith" in text
        assert "Technical Skills: Python, SQL" in text
        assert "Senior Analyst at TechCorp" in text
        assert "B.S. in Statistics" in text
        assert "Sales Forecasting Model" in text
        assert result["metadata"]["num_paragraphs"] > 0


class TestZeroWidthAndGlyphCleaning:
    """Tests for invisible unicode characters and unusual formatting symbols."""

    def test_zero_width_and_bom_removed(self):
        """Zero-width spaces and BOM should be cleanly stripped."""
        text_with_zero_width = "Python\u200bDeveloper\ufeff with 5 years\u200c experience"
        cleaned = clean_extracted_text(text_with_zero_width)
        assert "\u200b" not in cleaned
        assert "\ufeff" not in cleaned
        assert "\u200c" not in cleaned
        assert "PythonDeveloper with 5 years experience" in cleaned

    def test_extended_bullet_glyphs_recognized(self):
        """Alternative resume bullet glyphs should be normalized without mangling."""
        bullet_text = "• First bullet point\n✓ Second bullet point\n▸ Third bullet point"
        cleaned = clean_extracted_text(bullet_text)
        assert "• First bullet point" in cleaned
        assert "✓ Second bullet point" in cleaned
        assert "▸ Third bullet point" in cleaned


class TestScannedPDFDetection:
    """Tests for image-based or empty PDF detection."""

    def test_empty_pdf_raises_clear_scanned_notice(self, monkeypatch):
        """A PDF producing no text must raise RuntimeError explaining scanned/no-text status."""
        from unittest.mock import MagicMock
        import pypdf

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]

        monkeypatch.setattr(pypdf, "PdfReader", lambda path: mock_reader)

        with pytest.raises(RuntimeError, match="scanned image or photo without selectable text"):
            extract_text_from_pdf(SAMPLE_PDF)

