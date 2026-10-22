"""
Unit and integration tests for SmartHire Streamlit Web Interface.
Validates page initialization, tab rendering, session state management,
and integration of all 4 pillars:
(a) Parsed Profile View
(b) Recommended Job Matches
(c) CV Improvement Generator
(d) AI Career Mentor Chatbot
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app" / "streamlit_app.py")


def test_streamlit_app_loads_successfully():
    """Verify the Streamlit app loads and initializes without errors."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=25)
    assert not at.exception, f"App raised an exception: {at.exception}"

    # Verify 3 main tabs exist
    assert len(at.tabs) >= 3
    tab_labels = [t.label for t in at.tabs]
    assert any("Job Matches" in lbl for lbl in tab_labels)
    assert any("CV Improvement" in lbl for lbl in tab_labels)
    assert any("Career Mentor" in lbl for lbl in tab_labels)


def test_mentor_tab_initializes_memory_and_elements():
    """Verify that the AI Career Mentor tab initializes session state and UI components."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    # Check mentor memory initialized in session state
    assert "mentor_memory" in at.session_state
    mem = at.session_state["mentor_memory"]
    assert mem is not None
    assert len(mem.get_messages()) == 0

    # Check quick prompt buttons exist
    assert len(at.button) > 0


def test_candidate_profile_view_renders_when_profile_present():
    """Verify that structured candidate profile renders when parsed_profile is in session state."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    mock_profile = {
        "name": "Jane Doe",
        "target_role": "Backend Engineer",
        "contact_details": {
            "email": "jane@example.com",
            "phone": "+1-555-0199",
            "location": "Bengaluru",
        },
        "professional_summary": "Experienced backend engineer specializing in distributed systems.",
        "skills": {
            "technical": ["Python", "FastAPI", "PostgreSQL", "Docker"],
            "soft": ["Problem Solving", "Collaboration"],
        },
        "education": [
            {"degree": "B.Tech Computer Science", "institution": "IIT Madras", "year": "2023"}
        ],
        "experience": [
            {"title": "Software Engineer Intern", "company": "Tech Corp", "duration": "6 months"}
        ],
    }

    at.session_state["parsed_profile"] = mock_profile
    at.run(timeout=10)

    assert not at.exception
    # Check that the profile expander title contains the candidate's name
    assert any("Jane Doe" in exp.label for exp in at.expander)


def test_candidate_profile_view_with_list_skills():
    """Verify that structured candidate profile renders when skills is a list (standard schema)."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    mock_profile = {
        "name": "Alex Smith",
        "target_role": "Full Stack Developer",
        "contact": {
            "email": "alex@example.com",
            "phone": "+1-555-0100",
            "location": "Hyderabad",
        },
        "professional_summary": "Full stack engineer building scalable web applications.",
        "skills": ["JavaScript", "React", "Node.js", "MongoDB", "TypeScript"],
        "soft_skills": ["Agile", "Teamwork"],
        "education": [
            {"degree": "B.E. Information Technology", "institution": "BITS Pilani", "year": "2022"}
        ],
        "experience": [
            {"title": "Junior Developer", "company": "Acme Inc", "duration": "1 year"}
        ],
    }

    at.session_state["parsed_profile"] = mock_profile
    at.run(timeout=10)

    assert not at.exception
    assert any("Alex Smith" in exp.label for exp in at.expander)


def test_mentor_messages_do_not_render_sources_expander():
    """Verify that mentor conversation messages never display source expanders or file references."""
    from src.mentor.memory import ConversationMemory
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    mem = ConversationMemory()
    mem.add_user_message("What are backend developer skills?")
    mem.add_assistant_message(
        "**Answer**\nBackend developers build server-side logic and APIs.",
        sources=[{"source": "backend_developer.md", "source_type": "note"}],
    )
    at.session_state["mentor_memory"] = mem
    at.run(timeout=10)

    assert not at.exception
    # Ensure no expander displays 'Verified Evidence Sources' or document names
    expander_labels = [exp.label for exp in at.expander]
    assert not any("Sources" in lbl for lbl in expander_labels)
    assert not any("backend_developer.md" in lbl for lbl in expander_labels)


def test_bottom_model_selection_box_rendered_after_all_matches():
    """Verify that after job matches are rendered, the bottom model selection box appears and switches tabs."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    # Provide mock matched jobs in session state
    mock_jobs = [
        {
            "rank": 1,
            "job_id": "job_1",
            "job_title": "Python Developer",
            "company": "Tech Corp",
            "location": "Bengaluru",
            "suitability_reasoning": "Strong match for Python backend development.",
            "matching_skills": ["Python", "FastAPI"],
            "skills": "Python, FastAPI, SQL",
            "description": "Develop high performance services.",
        },
        {
            "rank": 2,
            "job_id": "job_2",
            "job_title": "Backend Engineer",
            "company": "Data Inc",
            "location": "Remote",
            "suitability_reasoning": "Good background in distributed services.",
            "matching_skills": ["Python", "Docker"],
            "skills": "Python, Docker, AWS",
            "description": "Maintain scalable APIs.",
        },
    ]

    at.session_state["matched_jobs"] = mock_jobs
    at.run(timeout=10)

    assert not at.exception
    # Ensure bottom model selection buttons are present
    button_keys = [btn.key for btn in at.button]
    assert "bottom_select_cv_btn" in button_keys
    assert "bottom_select_mentor_btn" in button_keys

    # Click the CV Improvement Generator button and verify tab switch
    at.button(key="bottom_select_cv_btn").click()
    at.run(timeout=10)

    assert not at.exception
    assert at.session_state["main_active_tab"] == "✨ CV Improvement Generator"


def test_find_matches_btn_bottom_rendered_after_profile():
    """Verify that after parsing a resume, the bottom 'Find Matching Jobs' button is rendered."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=10)

    # When no profile is parsed, bottom match button is not present
    button_keys = [btn.key for btn in at.button]
    assert "find_matches_btn_bottom" not in button_keys

    # Set mock parsed profile
    at.session_state["parsed_profile"] = {
        "name": "David Miller",
        "target_role": "Data Analyst",
        "skills": ["Python", "SQL", "Tableau"],
    }
    at.run(timeout=10)

    assert not at.exception
    # Now the bottom Find Matching Jobs button is present below the parsed profile
    button_keys = [btn.key for btn in at.button]
    assert "find_matches_btn_bottom" in button_keys


def test_universal_back_navigation():
    """Verify universal page history navigation: Page A -> Page B -> Back to Page A."""
    at = AppTest.from_file(APP_PATH)
    # Set mock profile and jobs
    at.session_state["parsed_profile"] = {"name": "Test User", "target_role": "Engineer"}
    at.session_state["matched_jobs"] = [
        {"rank": 1, "job_title": "ML Engineer", "company": "AI Labs", "location": "Remote", "suitability_reasoning": "Fit"}
    ]
    at.run(timeout=10)

    # Initial page is Job Matches (Page A)
    assert at.session_state["main_active_tab"] == "🎯 Recommended Job Matches"

    # Navigate to CV Improvement Generator (Page B) via bottom button
    at.button(key="bottom_select_cv_btn").click()
    at.run(timeout=10)
    assert at.session_state["main_active_tab"] == "✨ CV Improvement Generator"

    # Verify dynamic back button appears on Page B
    button_keys = [btn.key for btn in at.button]
    assert "nav_back_btn_tab2_top" in button_keys

    # Click Back on Page B -> returns to Job Matches (Page A)
    at.button(key="nav_back_btn_tab2_top").click()
    at.run(timeout=10)
    assert at.session_state["main_active_tab"] == "🎯 Recommended Job Matches"

