from src.parsing.resume_parser import parse_resume
from src.parsing.loader import load_document
from src.search.candidate_analysis import analyze_candidate


def test_candidate_analysis():
    doc = load_document("tests/fixtures/sample_resume.pdf")
    profile = parse_resume(doc["text"])

    result = analyze_candidate(
        profile=profile,
        allowed_levels=["fresher", "junior", "mid", "senior"],
        allowed_locations=["Hyderabad", "Bengaluru", "Remote"],
    )

    print(result)

    assert result["level"] in ["fresher", "junior", "mid", "senior"]
    assert result["search_text"].strip()
    assert "levels" in result["filters"]
    assert "locations" in result["filters"]