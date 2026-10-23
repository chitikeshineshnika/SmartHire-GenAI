"""
SmartHire GenAI — Semantic Job Matching & CV Improvement Generator Interface.

Provides an executive, modern user interface where candidates can:
1. Upload and parse their resume.
2. Discover personalized, LLM-evaluated job recommendations ranked by fit.
3. Select any target job (from recommendations or custom text) to generate:
   - Resume-to-job gap analysis & match score.
   - Categorized missing skills with learning paths vs. missing evidence.
   - High-impact bullet point improvements with verified fact grounding.
   - Job-tailored professional summary.
   - Complete job-tailored resume with downloadable text output.
"""

import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path for Streamlit Cloud deployment
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.parsing.loader import load_document
from src.parsing.resume_parser import parse_resume
from src.search.candidate_analysis import analyze_candidate
from src.search.matcher import match_jobs_for_candidate
from src.generate.cv_suggestions import (
    CVImprovementResult,
    generate_cv_improvements,
    normalize_job_input,
)
from src.safety.guardrails import (
    detect_prompt_injection,
    validate_cv_generator_inputs,
)
from src.mentor.rag_chain import ask_career_mentor
from src.mentor.memory import ConversationMemory, ChatMessage

logger = logging.getLogger(__name__)

DATASET_LEVELS = ["fresher", "junior", "mid", "senior"]
DATASET_LOCATIONS = [
    "Bengaluru", "Chennai", "Delhi", "Gurugram",
    "Hyderabad", "Kolkata", "Mumbai", "Noida", "Pune", "Remote",
]

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartHire GenAI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hero Header Banner */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #31104b 100%);
    border-radius: 16px;
    padding: 36px 32px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-title {
    font-size: 2.85rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.025em;
    margin-bottom: 10px;
    color: #ffffff;
}

.hero-brand {
    background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-genai {
    background: linear-gradient(90deg, #38bdf8 0%, #60a5fa 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.hero-content {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 32px;
}

.hero-left {
    flex: 1 1 58%;
    min-width: 300px;
}

.hero-subtitle {
    font-size: 1.02rem;
    color: #94a3b8;
    max-width: 680px;
    line-height: 1.5;
    margin-bottom: 0;
}

.hero-features {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 18px;
    flex: 0 0 auto;
}

.hero-feature-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 10px;
    width: 125px;
}

.feature-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 12px;
}

.feature-icon svg {
    width: 24px;
    height: 24px;
}

.icon-cyan {
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.28);
}

.icon-teal {
    color: #14b8a6;
    background: rgba(20, 184, 166, 0.12);
    border: 1px solid rgba(20, 184, 166, 0.28);
}

.icon-purple {
    color: #a78bfa;
    background: rgba(167, 139, 250, 0.12);
    border: 1px solid rgba(167, 139, 250, 0.28);
}

.feature-divider {
    width: 1px;
    height: 54px;
    background: rgba(255, 255, 255, 0.15);
}

.feature-label {
    color: #ffffff;
    font-weight: 600;
    font-size: 0.82rem;
    line-height: 1.25;
    text-align: center;
}

@media (max-width: 960px) {
    .hero-content {
        flex-direction: column;
        align-items: flex-start;
    }
    .hero-features {
        justify-content: flex-start;
        margin-top: 18px;
    }
}

/* Job Result Card Container */
.job-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px;
    margin-top: 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    transition: all 0.2s ease-in-out;
}

.job-card:hover {
    border-color: #8b5cf6;
    box-shadow: 0 10px 20px -3px rgba(139, 92, 246, 0.14);
}

.card-header-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}

.rank-badge {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 5px 14px;
    border-radius: 9999px;
    letter-spacing: 0.03em;
    white-space: nowrap;
}

.rank-badge-top {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
}

.job-header-title {
    font-size: 1.28rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.35;
    margin: 0;
}

/* LLM Match Rationale Box */
.rationale-box {
    background: #f0fdf4;
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
    font-size: 0.94rem;
    color: #064e3b;
    line-height: 1.55;
}

.rationale-title {
    font-weight: 700;
    margin-bottom: 4px;
    color: #047857;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Matching Skills Section */
.skills-section {
    margin-bottom: 8px;
}

.skills-label {
    font-size: 0.85rem;
    font-weight: 700;
    color: #334155;
    margin-bottom: 8px;
    display: block;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.skill-tag {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 6px;
    margin-right: 6px;
    margin-bottom: 6px;
    border: 1px solid #c7d2fe;
}

/* CV Improvement Section Cards */
.cv-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px;
    margin-top: 18px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.cv-score-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    color: #ffffff;
    padding: 20px 24px;
    border-radius: 12px;
    margin-bottom: 24px;
}

.score-badge {
    font-size: 2rem;
    font-weight: 800;
    color: #34d399;
}

.bullet-comp-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
}

.bullet-orig {
    color: #64748b;
    font-size: 0.92rem;
    margin-bottom: 8px;
    padding-left: 8px;
    border-left: 3px solid #cbd5e1;
}

.bullet-improved {
    color: #0f172a;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 8px;
    padding-left: 8px;
    border-left: 3px solid #6366f1;
}

.change-tag {
    display: inline-block;
    background: #e0e7ff !important;
    color: #3730a3 !important;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
}

/* Confirmed Skill Gaps Card - High Contrast */
.gap-card-confirmed {
    background: #fff1f2 !important;
    border-left: 5px solid #e11d48 !important;
    border: 1px solid #fecdd3 !important;
    border-left-width: 5px !important;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
    color: #4c0519 !important;
}

.gap-card-confirmed * {
    color: #4c0519 !important;
}

.gap-card-confirmed strong {
    color: #9f1239 !important;
    font-weight: 700;
}

/* Missing Evidence Card - High Contrast */
.gap-card-evidence {
    background: #eff6ff !important;
    border-left: 5px solid #2563eb !important;
    border: 1px solid #bfdbfe !important;
    border-left-width: 5px !important;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
    color: #172554 !important;
}

.gap-card-evidence * {
    color: #172554 !important;
}

.gap-card-evidence strong {
    color: #1e40af !important;
    font-weight: 700;
}

.grounded-banner {
    background: #ecfdf5 !important;
    border: 1px solid #a7f3d0 !important;
    color: #065f46 !important;
    padding: 12px 16px;
    border-radius: 8px;
    font-weight: 600;
    margin-bottom: 16px;
}

.grounded-banner * {
    color: #065f46 !important;
}

/* Tailored Resume Paper Container - High Contrast & ATS Structured */
.resume-paper {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px;
    padding: 32px 36px;
    margin-top: 14px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a !important;
    line-height: 1.6;
}

.resume-paper * {
    color: #0f172a !important;
}

.resume-paper h1 {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
}

.resume-paper h2 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #1e3a8a !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 22px;
    margin-bottom: 12px;
}

.resume-paper h3 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    margin-top: 14px;
    margin-bottom: 6px;
}

.resume-paper p, .resume-paper li {
    font-size: 0.94rem !important;
    color: #334155 !important;
    line-height: 1.65;
}

.resume-paper strong {
    color: #0f172a !important;
    font-weight: 700;
}

.resume-paper ul {
    margin-top: 4px;
    margin-bottom: 12px;
    padding-left: 20px;
}

.resume-paper hr {
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
}

/* Candidate Extracted Profile Box */
.profile-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 22px 26px;
    margin-top: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
}

.profile-meta-chip {
    display: inline-flex;
    align-items: center;
    background: #f8fafc;
    color: #334155;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    margin-right: 8px;
    margin-bottom: 6px;
    border: 1px solid #e2e8f0;
}

/* AI Career Mentor Styling */
.mentor-hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    border-radius: 14px;
    padding: 24px 28px;
    color: #ffffff;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.mentor-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(8px);
    color: #e0e7ff;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-right: 8px;
    margin-top: 8px;
    border: 1px solid rgba(255, 255, 255, 0.22);
}

/* =========================================================================
   Executive Highlighting for the 3 Core AI Models (Tabs & Section Banners)
   ========================================================================= */

/* Highlighted Tab Navigation Bar */
div[data-baseweb="tab-list"] {
    gap: 12px !important;
    background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%) !important;
    padding: 10px 14px 0 14px !important;
    border-radius: 14px 14px 0 0 !important;
    border: 1px solid #e2e8f0 !important;
    border-bottom: 2px solid #cbd5e1 !important;
    margin-bottom: 24px !important;
}

button[data-baseweb="tab"],
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 1.20rem !important;
    font-weight: 800 !important;
    padding: 13px 26px !important;
    border-radius: 12px 12px 0 0 !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1.5px solid #cbd5e1 !important;
    border-bottom: none !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span,
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] span {
    font-size: 1.20rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    letter-spacing: -0.01em !important;
    margin: 0 !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"]:active,
div[data-testid="stTabs"] button[role="tab"]:active {
    transform: scale(0.97) !important;
}

/* Individual Hover States */
div[data-baseweb="tab-list"] button:nth-of-type(1):hover,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(1):hover {
    color: #065f46 !important;
    -webkit-text-fill-color: #065f46 !important;
    background: #ecfdf5 !important;
    border-color: #a7f3d0 !important;
}

div[data-baseweb="tab-list"] button:nth-of-type(2):hover,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(2):hover {
    color: #581c87 !important;
    -webkit-text-fill-color: #581c87 !important;
    background: #faf5ff !important;
    border-color: #e9d5ff !important;
}

div[data-baseweb="tab-list"] button:nth-of-type(3):hover,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(3):hover {
    color: #0891b2 !important;
    -webkit-text-fill-color: #0891b2 !important;
    background: #ecfeff !important;
    border-color: #a5f3fc !important;
}

/* Specific Striking Highlights When Clicked / Selected (aria-selected="true") */
/* Model 1: Recommended Job Matches (Emerald Highlight) */
div[data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(1)[aria-selected="true"] {
    background: linear-gradient(135deg, #065f46 0%, #059669 60%, #10b981 100%) !important;
    border: 2px solid #34d399 !important;
    border-bottom: 4px solid #047857 !important;
    box-shadow: 0 4px 18px rgba(16, 185, 129, 0.38) !important;
}
div[data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"] p,
div[data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"] span,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(1)[aria-selected="true"] p,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(1)[aria-selected="true"] span {
    color: #ffffff !important;
    font-size: 1.28rem !important;
    font-weight: 900 !important;
    -webkit-text-fill-color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35) !important;
}

/* Model 2: CV Improvement Generator (Royal Violet Highlight) */
div[data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(2)[aria-selected="true"] {
    background: linear-gradient(135deg, #581c87 0%, #7c3aed 60%, #9333ea 100%) !important;
    border: 2px solid #c084fc !important;
    border-bottom: 4px solid #6b21a8 !important;
    box-shadow: 0 4px 18px rgba(147, 51, 234, 0.38) !important;
}
div[data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"] p,
div[data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"] span,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(2)[aria-selected="true"] p,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(2)[aria-selected="true"] span {
    color: #ffffff !important;
    font-size: 1.28rem !important;
    font-weight: 900 !important;
    -webkit-text-fill-color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35) !important;
}

/* Model 3: AI Career Mentor (Electric Cyan & Midnight Teal Highlight) */
div[data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(3)[aria-selected="true"] {
    background: linear-gradient(135deg, #0e7490 0%, #0891b2 50%, #06b6d4 100%) !important;
    border: 2px solid #22d3ee !important;
    border-bottom: 4px solid #155e75 !important;
    box-shadow: 0 4px 18px rgba(6, 182, 212, 0.45) !important;
}
div[data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"] p,
div[data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"] span,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(3)[aria-selected="true"] p,
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(3)[aria-selected="true"] span {
    color: #ffffff !important;
    font-size: 1.28rem !important;
    font-weight: 900 !important;
    -webkit-text-fill-color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35) !important;
}

/* Model Section Highlight Banners */
.model-hero-card {
    border-radius: 14px;
    padding: 24px 28px;
    color: #ffffff;
    margin-bottom: 22px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.model-hero-job-matches {
    background: linear-gradient(135deg, #064e3b 0%, #0f172a 55%, #1e1b4b 100%);
    border-left: 6px solid #10b981;
}

.model-hero-cv-generator {
    background: linear-gradient(135deg, #7C2D12 0%, #431407 55%, #1c1917 100%);
    border-left: 6px solid #f97316;
    box-shadow: 0 10px 25px -5px rgba(124, 45, 18, 0.45);
}

.model-hero-career-mentor {
    background: linear-gradient(135deg, #022027 0%, #083344 45%, #0e7490 85%, #0891b2 100%);
    border-left: 6px solid #22d3ee;
    box-shadow: 0 10px 25px -5px rgba(6, 182, 212, 0.4);
}

.model-category-badge {
    display: inline-block;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.15);
    color: #e2e8f0;
    margin-bottom: 8px;
}

.model-hero-title {
    font-size: 1.85rem;
    font-weight: 900;
    letter-spacing: -0.01em;
    margin: 0 0 6px 0;
    color: #ffffff;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

.model-hero-desc {
    font-size: 0.98rem;
    color: #cbd5e1;
    max-width: 820px;
    line-height: 1.5;
    margin: 0;
}

/* Unified Elegant Model Selection Boxes at Very Bottom */
div.st-key-bottom_select_cv_btn button,
div.st-key-bottom_select_mentor_btn button {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 55%, #312e81 100%) !important;
    border: 1.5px solid rgba(165, 180, 252, 0.45) !important;
    color: #ffffff !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 13px 20px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div.st-key-bottom_select_cv_btn button p,
div.st-key-bottom_select_cv_btn button span,
div.st-key-bottom_select_mentor_btn button p,
div.st-key-bottom_select_mentor_btn button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}

div.st-key-bottom_select_cv_btn button:hover,
div.st-key-bottom_select_mentor_btn button:hover {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 55%, #4338ca 100%) !important;
    border-color: #c7d2fe !important;
    box-shadow: 0 8px 24px rgba(67, 56, 202, 0.38), inset 0 1px 1px rgba(255, 255, 255, 0.28) !important;
    transform: translateY(-2px) !important;
}

/* AI Career Mentor: Question Sent & Chat Message Primary Royal Blue (#2563E8) */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
div[aria-label="Chat message from user"],
.stChatMessageUser {
    background-color: #2563E8 !important;
    border: 1.5px solid #1d4ed8 !important;
    border-radius: 14px !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 232, 0.28) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div,
div[aria-label="Chat message from user"] p,
div[aria-label="Chat message from user"] span,
div[aria-label="Chat message from user"] div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 600 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"],
div[aria-label="Chat message from user"] [data-testid="stChatMessageAvatarUser"] {
    background-color: #1e40af !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) svg,
div[aria-label="Chat message from user"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* Spinner in Royal Blue #2563E8 */
div.stSpinner > div {
    border-top-color: #2563E8 !important;
}

div.stSpinner {
    color: #2563E8 !important;
}

/* Chat Input: Single Pure Royal Blue Border with ZERO orange */
div[data-testid="stChatInput"] {
    --primary-color: #2563E8 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

div[data-testid="stChatInput"]:focus-within {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* The actual chat input box (direct child container) */
div[data-testid="stChatInput"] > div {
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

div[data-testid="stChatInput"] > div:focus-within {
    border: 1.5px solid #2563E8 !important;
    border-color: #2563E8 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 232, 0.22) !important;
}

/* Completely strip borders and outlines from all inner children */
div[data-testid="stChatInput"] > div *,
div[data-testid="stChatInput"] div[data-baseweb="base-input"],
div[data-testid="stChatInput"] div[data-baseweb="textarea"],
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInputTextArea"] {
    border: none !important;
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

div[data-testid="stChatInput"] > div *:focus,
div[data-testid="stChatInput"] > div *:focus-within,
div[data-testid="stChatInput"] div[data-baseweb="base-input"]:focus-within,
div[data-testid="stChatInput"] div[data-baseweb="textarea"]:focus-within,
div[data-testid="stChatInput"] textarea:focus,
div[data-testid="stChatInputTextArea"]:focus {
    border: none !important;
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Send Button: Original Streamlit send button styled with Teal #14B8A6 instead of orange */
button[data-testid="stChatInputSubmitButton"] {
    background-color: #14B8A6 !important;
    border: none !important;
    color: #ffffff !important;
}

button[data-testid="stChatInputSubmitButton"]:hover {
    background-color: #0d9488 !important;
}

button[data-testid="stChatInputSubmitButton"]:disabled {
    background-color: transparent !important;
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}

button[data-testid="stChatInputSubmitButton"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

button[data-testid="stChatInputSubmitButton"]:disabled svg {
    fill: #94a3b8 !important;
    color: #94a3b8 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "mentor_memory" not in st.session_state or not hasattr(st.session_state.mentor_memory, "get_conversation_turns"):
    st.session_state.mentor_memory = ConversationMemory(max_turns=10)

# ---------------------------------------------------------------------------
# Header Hero Section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <div class="hero-left">
            <div class="hero-title"><span class="hero-brand">SmartHire</span> <span class="hero-genai">GenAI</span></div>
            <div class="hero-subtitle">
                Upload your resume to extract your structured profile, discover top matching jobs via semantic search,
                generate strictly grounded CV improvements for target roles, and consult the AI Career Mentor RAG chatbot.
            </div>
        </div>
        <div class="hero-features">
            <div class="hero-feature-item">
                <div class="feature-icon icon-cyan">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                </div>
                <div class="feature-label">Semantic Job Search</div>
            </div>
            <div class="feature-divider"></div>
            <div class="hero-feature-item">
                <div class="feature-icon icon-teal">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                </div>
                <div class="feature-label">CV Improvement Generator</div>
            </div>
            <div class="feature-divider"></div>
            <div class="hero-feature-item">
                <div class="feature-icon icon-purple">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                </div>
                <div class="feature-label">AI Career Mentor</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Resume Upload & Parsing Helper
# ---------------------------------------------------------------------------
def _extract_and_parse_resume(uploaded_file) -> Optional[dict]:
    """Extract and parse candidate resume, caching into session state."""
    suffix = Path(uploaded_file.name).suffix.lower()
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_file_path = tmp.name

        doc = load_document(temp_file_path)
        resume_text = doc.get("text", "").strip()
        if not resume_text:
            raise ValueError("Could not extract readable text from document.")

        profile = parse_resume(resume_text)
        st.session_state["resume_text"] = resume_text
        st.session_state["parsed_profile"] = profile
        st.session_state["resume_filename"] = uploaded_file.name
        st.session_state["resume_parsed_target_role"] = profile.get("target_role")
        st.session_state["selected_target_job"] = None
        st.session_state["selected_target_job_id"] = None
        for k in list(st.session_state.keys()):
            if k.startswith("target_role_chk_"):
                st.session_state[k] = False
        return profile
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass


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


def _extract_profile_skills(profile_dict: Optional[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    """
    Extract technical and soft skills cleanly whether profile stores skills
    as list (standard ResumeProfile schema) or dict ({'technical': [...], 'soft': [...]}).
    """
    if not profile_dict or not isinstance(profile_dict, dict):
        return [], []

    raw_skills = profile_dict.get("skills")
    raw_soft = profile_dict.get("soft_skills")

    tech_skills: List[str] = []
    soft_skills: List[str] = []

    if isinstance(raw_skills, dict):
        tech_skills = [str(s).strip() for s in (raw_skills.get("technical") or []) if s]
        soft_skills = [str(s).strip() for s in (raw_skills.get("soft") or []) if s]
    elif isinstance(raw_skills, list):
        tech_skills = [str(s).strip() for s in raw_skills if s]

    if isinstance(raw_soft, list):
        for s in raw_soft:
            if s and str(s).strip() not in soft_skills:
                soft_skills.append(str(s).strip())
    elif isinstance(raw_soft, dict):
        for s in (raw_soft.get("soft") or []):
            if s and str(s).strip() not in soft_skills:
                soft_skills.append(str(s).strip())

    return tech_skills, soft_skills




# ---------------------------------------------------------------------------
# Upload & Action Section
# ---------------------------------------------------------------------------
upload_col, action_col = st.columns([2.5, 1.5], vertical_alignment="bottom")

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload Candidate Resume (PDF or DOCX)",
        type=["pdf", "docx"],
        help="Upload standard digital PDF or DOCX format resume (max 10MB).",
        label_visibility="visible",
    )

with action_col:
    parse_col, match_col = st.columns(2)
    with parse_col:
        parse_btn = st.button("📄 Parse Resume", use_container_width=True)
    with match_col:
        find_matches_btn = st.button("🚀 Find Matching Jobs", type="primary", use_container_width=True)

# Parse on explicit button click
if parse_btn:
    if uploaded_file is None:
        st.warning("⚠️ Please select and upload a resume file first.")
    else:
        with st.spinner("📄 Reading and analyzing candidate resume with Gemini..."):
            try:
                _extract_and_parse_resume(uploaded_file)
                st.success("✅ Resume parsed successfully! Review the extracted candidate profile below.")
            except Exception as e:
                st.error(f"❌ Resume parsing failed: {e}. If the AI service is experiencing heavy load, please wait a moment and click 'Parse Resume' again.")

# Handle file replacement/removal without automatic parsing
if uploaded_file is not None:
    if st.session_state.get("uploaded_file_name") != uploaded_file.name:
        st.session_state["uploaded_file_name"] = uploaded_file.name
        # Clear previous parsed data so old resume data is not retained
        st.session_state["parsed_profile"] = None
        st.session_state["resume_text"] = None
        st.session_state["matched_jobs"] = None
        st.session_state["selected_target_job"] = None
        st.session_state["selected_target_job_id"] = None
else:
    if "uploaded_file_name" in st.session_state:
        st.session_state.pop("uploaded_file_name", None)
        st.session_state["parsed_profile"] = None
        st.session_state["resume_text"] = None
        st.session_state["matched_jobs"] = None
        st.session_state["selected_target_job"] = None
        st.session_state["selected_target_job_id"] = None

# ---------------------------------------------------------------------------
# Structured Candidate Profile View (Pillar a)
# ---------------------------------------------------------------------------
find_matches_btn_bottom = False
active_profile = st.session_state.get("parsed_profile")
if active_profile:
    cand_name = active_profile.get("name") or "Candidate"
    target_role = active_profile.get("target_role")
    has_role = bool(target_role and str(target_role).strip().lower() not in ("none", "null", "not specified", "target role not specified"))
    display_role = str(target_role).strip() if has_role else "Target role not specified"
    title_suffix = f" ({display_role})" if has_role else ""
    with st.expander(f"👤 Extracted Candidate Profile — {cand_name}{title_suffix}", expanded=True):
        st.markdown(f"**🎯 Target Role:** `{display_role}`")

        # Professional Summary
        summary = active_profile.get("professional_summary")
        if summary:
            st.markdown("##### 📝 Professional Summary")
            st.markdown(f"> {summary}")

        # Skills
        tech_skills, soft_skills = _extract_profile_skills(active_profile)
        if tech_skills or soft_skills:
            st.markdown("##### 🛠️ Extracted Skills")
            if tech_skills:
                tech_chips = "".join([f'<span class="skill-tag">{s}</span>' for s in tech_skills])
                st.markdown(f"**Technical:** {tech_chips}", unsafe_allow_html=True)
            if soft_skills:
                soft_chips = "".join([f'<span class="skill-tag" style="background:#f1f5f9; color:#475569; border-color:#cbd5e1;">{s}</span>' for s in soft_skills])
                st.markdown(f"**Soft Skills:** {soft_chips}", unsafe_allow_html=True)

        # Experience & Education in 2 columns
        exp_list = active_profile.get("experience") or []
        edu_list = active_profile.get("education") or []
        if exp_list or edu_list:
            c_exp, c_edu = st.columns([1.1, 0.9])
            with c_exp:
                st.markdown("##### 💼 Experience & Internships")
                if exp_list:
                    for exp in exp_list:
                        if isinstance(exp, dict):
                            t = exp.get("title") or "Role"
                            c = exp.get("company") or "Organization"
                            d = exp.get("duration") or ""
                            etype = exp.get("type", "job")
                            desc = exp.get("description") or []
                            techs = exp.get("technologies") or []
                        else:
                            t = getattr(exp, "title", "Role")
                            c = getattr(exp, "company", "Organization")
                            d = getattr(exp, "duration", "")
                            etype = getattr(exp, "type", "job")
                            desc = getattr(exp, "description", [])
                            techs = getattr(exp, "technologies", [])

                        type_badge = "Internship" if etype == "internship" else "Job"
                        st.markdown(f"- **{t}** at **{c}** *({type_badge})* {f'— {d}' if d else ''}")
                        if techs:
                            tech_badges = " ".join([f"`{tc}`" for tc in techs[:6]])
                            st.markdown(f"  <small>Tech: {tech_badges}</small>", unsafe_allow_html=True)
                        if isinstance(desc, list) and desc:
                            for bullet in desc[:3]:
                                st.markdown(f"  - {bullet}")
                else:
                    st.caption("No prior work experience listed.")

            with c_edu:
                st.markdown("##### 🎓 Education")
                if edu_list:
                    for edu in edu_list:
                        if isinstance(edu, dict):
                            deg = edu.get("degree") or "Degree"
                            spec = edu.get("specialization") or ""
                            inst = edu.get("institution") or "Institution"
                            yr = edu.get("year") or ""
                            grade = edu.get("grade") or ""
                        else:
                            deg = getattr(edu, "degree", "Degree")
                            spec = getattr(edu, "specialization", "")
                            inst = getattr(edu, "institution", "Institution")
                            yr = getattr(edu, "year", "")
                            grade = getattr(edu, "grade", "")

                        title = f"{deg} in {spec}" if spec else deg
                        grade_str = f" • Grade: {grade}" if grade else ""
                        st.markdown(f"- **{title}** — {inst} {f'({yr})' if yr else ''}{grade_str}")
                else:
                    st.caption("No formal degree explicitly listed.")

        # Projects, Certifications & Achievements
        proj_list = active_profile.get("projects") or []
        cert_list = active_profile.get("certifications") or []
        achieve_list = active_profile.get("achievements") or []
        if proj_list or cert_list or achieve_list:
            c_proj, c_cert = st.columns(2)
            with c_proj:
                if proj_list:
                    st.markdown("##### 🚀 Projects")
                    for proj in proj_list:
                        if isinstance(proj, dict):
                            pname = proj.get("name") or "Project"
                            pdesc = proj.get("description") or ""
                            ptech = proj.get("technologies") or []
                            purl = proj.get("url") or ""
                        else:
                            pname = getattr(proj, "name", "Project")
                            pdesc = getattr(proj, "description", "")
                            ptech = getattr(proj, "technologies", [])
                            purl = getattr(proj, "url", "")

                        name_display = f"[{pname}]({purl})" if purl else f"**{pname}**"
                        st.markdown(f"- {name_display}")
                        if ptech:
                            st.caption(f"Technologies: {', '.join(ptech)}")
                        if pdesc:
                            st.markdown(f"  {pdesc}")
            with c_cert:
                if cert_list:
                    st.markdown("##### 📜 Certifications")
                    for cert in cert_list:
                        if isinstance(cert, dict):
                            cname = cert.get("name") or "Certification"
                            cissuer = cert.get("issuer") or ""
                            cdate = cert.get("date") or ""
                            curl = cert.get("url") or ""
                        else:
                            cname = getattr(cert, "name", "Certification")
                            cissuer = getattr(cert, "issuer", "")
                            cdate = getattr(cert, "date", "")
                            curl = getattr(cert, "url", "")

                        cert_display = f"[{cname}]({curl})" if curl else f"**{cname}**"
                        issuer_str = f" by {cissuer}" if cissuer else ""
                        date_str = f" ({cdate})" if cdate else ""
                        st.markdown(f"- {cert_display}{issuer_str}{date_str}")

                if achieve_list:
                    st.markdown("##### 🏆 Key Achievements")
                    for ach in achieve_list:
                        st.markdown(f"- {ach}")

        # Enable the same find job matching button down of after parsing
        c_match_down, _ = st.columns([1.5, 2.5])
        with c_match_down:
            find_matches_btn_bottom = st.button(
                "🚀 Find Matching Jobs",
                type="primary",
                key="find_matches_btn_bottom",
                use_container_width=True,
                help="Discover and rank top matching job opportunities for this profile.",
            )

st.divider()

# ---------------------------------------------------------------------------
# Navigation State & Tab Switching Helpers
# ---------------------------------------------------------------------------
if "nav_history" not in st.session_state:
    st.session_state["nav_history"] = []

def switch_to_tab(tab_name: str) -> None:
    history = st.session_state.get("nav_history", [])
    if not history or history[-1] != tab_name:
        history.append(tab_name)
    st.session_state["nav_history"] = history
    st.session_state["main_active_tab"] = tab_name

# ---------------------------------------------------------------------------
# Matching Workflow Pipeline Execution
# ---------------------------------------------------------------------------
if find_matches_btn or find_matches_btn_bottom:
    if uploaded_file is None:
        st.warning("⚠️ Please select and upload a resume file first before searching.")
    else:
        with st.status("🔍 Analyzing Resume & Searching Jobs...", expanded=True) as status:
            try:
                profile = st.session_state.get("parsed_profile")
                if not profile:
                    status.update(label="📄 Extracting text and parsing resume...", state="running")
                    profile = _extract_and_parse_resume(uploaded_file)

                # Stage: Candidate intent analysis
                status.update(label="🧠 Analyzing candidate credentials with Gemini...", state="running")
                candidate_analysis = None
                try:
                    candidate_analysis = analyze_candidate(
                        profile,
                        allowed_levels=DATASET_LEVELS,
                        allowed_locations=DATASET_LOCATIONS,
                    )
                except Exception as ca_err:
                    logger.warning("Optional candidate analysis failed: %s", ca_err)

                # Stage: FAISS Semantic Retrieval + Gemini Reranking
                status.update(label="⚡ Searching FAISS vector store & evaluating suitability...", state="running")
                matched_jobs = match_jobs_for_candidate(
                    candidate_profile=profile,
                    candidate_analysis=candidate_analysis,
                )

                st.session_state["matched_jobs"] = matched_jobs
                status.update(label="✅ Recommendations generated successfully!", state="complete", expanded=False)
                switch_to_tab("🎯 Recommended Job Matches")

            except Exception as pipeline_err:
                status.update(label="❌ An error occurred during matching.", state="error", expanded=True)
                st.error(f"Error during job matching: {pipeline_err}")
                logger.exception("Job matching pipeline error: %s", pipeline_err)


# ---------------------------------------------------------------------------
# Main Application Tabs (Job Search, CV Improvement Generator, & Career Mentor)
# ---------------------------------------------------------------------------
DEFAULT_TAB = "🎯 Recommended Job Matches"
if "nav_history" not in st.session_state or not st.session_state["nav_history"]:
    st.session_state["nav_history"] = [DEFAULT_TAB]
if "main_active_tab" not in st.session_state:
    st.session_state["main_active_tab"] = DEFAULT_TAB

def navigate_back() -> None:
    history = st.session_state.get("nav_history", [])
    if len(history) > 1:
        history.pop()
        prev_tab = history[-1]
    else:
        prev_tab = DEFAULT_TAB
    st.session_state["nav_history"] = history
    st.session_state["main_active_tab"] = prev_tab

def on_main_tab_changed() -> None:
    curr_tab = st.session_state.get("main_active_tab")
    if curr_tab:
        history = st.session_state.get("nav_history", [])
        if not history or history[-1] != curr_tab:
            history.append(curr_tab)
            st.session_state["nav_history"] = history

def render_back_button(key_suffix: str) -> None:
    history = st.session_state.get("nav_history", [])
    if len(history) > 1:
        prev_page = history[-2]
        prev_label = prev_page
        for emoji in ["🎯 ", "✨ ", "🤖 "]:
            prev_label = prev_label.replace(emoji, "")
        col_back, _ = st.columns([0.8, 11.2])
        with col_back:
            st.button(
                "←",
                key=f"nav_back_btn_{key_suffix}",
                on_click=navigate_back,
                help=f"Back to {prev_label}",
            )
            st.write("")

main_tab_search, main_tab_cv, main_tab_mentor = st.tabs(
    [
        "🎯 Recommended Job Matches",
        "✨ CV Improvement Generator",
        "🤖 AI Career Mentor",
    ],
    key="main_active_tab",
    on_change=on_main_tab_changed,
)

# ===========================================================================
# TAB 1: JOB SEARCH RESULTS
# ===========================================================================
with main_tab_search:
    render_back_button("tab1_top")
    st.markdown("""
    <div class="model-hero-card model-hero-job-matches">
        <span class="model-category-badge">CORE AI MODEL 1 • RECOMMENDED CAREER OPPORTUNITIES</span>
        <h2 class="model-hero-title">🎯 Recommended Job Matches</h2>
        <p class="model-hero-desc">
            Personalized role recommendations and career opportunities evaluated against your resume qualifications, experience level, and target career goals.
        </p>
        <div style="margin-top: 10px;">
            <span class="mentor-badge">🎯 Top Matched Roles</span>
            <span class="mentor-badge">💼 Relevant Opportunities</span>
            <span class="mentor-badge">📊 Role Fit & Rationales</span>
            <span class="mentor-badge">🎯 Target Role Selection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    matched_jobs: Optional[List[Dict[str, Any]]] = st.session_state.get("matched_jobs")

    if matched_jobs is not None:
        if not matched_jobs:
            st.info("ℹ️ No active job postings met the relevance threshold for this resume.")
        else:
            st.markdown(f"#### 📋 Ranked Opportunities ({len(matched_jobs)} Roles Found)")
            st.caption("Ranked from most suitable to least suitable based on Gemini's evaluation of skills, experience, and requirements.")

            for job in matched_jobs:
                rank = job.get("rank", 1)
                title = job.get("job_title", "Untitled Role")
                company = job.get("company", "Undisclosed Company")
                location = job.get("location", "Location Not Specified")
                reasoning = job.get("suitability_reasoning", "").strip()
                matching_skills = job.get("matching_skills", [])
                description = job.get("description", "").strip()
                skills_req = job.get("skills", "").strip()

                rank_badge_class = "rank-badge rank-badge-top" if rank <= 3 else "rank-badge"
                job_header = f"#{rank} {title} at {company} ({location})"

                # Interactive Result Card
                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <div class="card-header-row">
                            <span class="{rank_badge_class}">Rank #{rank}</span>
                            <div class="job-header-title">{job_header}</div>
                        </div>
                        <div class="rationale-box">
                            <div class="rationale-title">💡 Why this job is a match</div>
                            {reasoning}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if matching_skills:
                        chips_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in matching_skills])
                        st.markdown(f"""
                        <div class="skills-section" style="margin-left: 12px; margin-bottom: 12px;">
                            <span class="skills-label">Matching Skills</span>
                            {chips_html}
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander(f"📄 Full Job Description — {title} at {company}", expanded=False):
                        if skills_req:
                            st.markdown(f"**Required Qualifications:** {skills_req}")
                        if description:
                            st.markdown(description)

                    # Directing button: Symbol and Target Role only (not a link, with toggle & switch support)
                    job_uid = str(job.get("job_id") or f"{title}_{company}_{rank}")
                    current_selected_id = st.session_state.get("selected_target_job_id")
                    is_current_target = (current_selected_id == job_uid)

                    c_btn, _ = st.columns([2.2, 7.8])
                    with c_btn:
                        btn_label = "✅ Target Role" if is_current_target else "🎯 Target Role"
                        btn_help = "Click to unselect this target role" if is_current_target else f"Set {title} as your target role"
                        if st.button(
                            btn_label,
                            key=f"target_role_btn_{rank}_{job_uid}",
                            type="primary" if is_current_target else "secondary",
                            help=btn_help,
                        ):
                            current_profile = st.session_state.get("parsed_profile")
                            if is_current_target:
                                # Mistakenly selected: erase selection and revert
                                st.session_state["selected_target_job"] = None
                                st.session_state["selected_target_job_id"] = None
                                default_role = st.session_state.get("resume_parsed_target_role")
                                if current_profile is not None:
                                    current_profile["target_role"] = default_role
                                if active_profile is not None:
                                    active_profile["target_role"] = default_role
                            else:
                                # Overwrite previous selection and set new target role (erases before selection)
                                st.session_state["selected_target_job"] = job
                                st.session_state["selected_target_job_id"] = job_uid
                                if current_profile is not None:
                                    current_profile["target_role"] = title
                                if active_profile is not None:
                                    active_profile["target_role"] = title
                            st.rerun()

                    st.write("")

            # -------------------------------------------------------------------
            # Small model selection boxes at the very bottom
            st.write("")
            col_bottom_cv, col_bottom_mentor = st.columns(2, gap="medium")

            with col_bottom_cv:
                st.button(
                    "✨ Select CV Improvement Generator",
                    key="bottom_select_cv_btn",
                    type="primary",
                    use_container_width=True,
                    on_click=switch_to_tab,
                    args=("✨ CV Improvement Generator",),
                    help="Switch directly to the CV Improvement Generator with your profile and selected role pre-loaded.",
                )

            with col_bottom_mentor:
                st.button(
                    "🤖 Select AI Career Mentor",
                    key="bottom_select_mentor_btn",
                    type="primary",
                    use_container_width=True,
                    on_click=switch_to_tab,
                    args=("🤖 AI Career Mentor",),
                    help="Switch directly to the AI Career Mentor tab.",
                )
    else:
        st.info("Upload your resume and click **Find Matching Jobs** to view ranked recommendations.")


# ===========================================================================
# TAB 2: CV IMPROVEMENT GENERATOR
# ===========================================================================
with main_tab_cv:
    render_back_button("tab2_top")

    st.markdown("""
    <div class="model-hero-card model-hero-cv-generator">
        <span class="model-category-badge">CORE AI MODEL 2 • RESUME TAILORING & GAP ANALYSIS</span>
        <h2 class="model-hero-title">✨ CV Improvement Generator</h2>
        <p class="model-hero-desc">
            5-stage Gemini CV engineering: Grounded gap analysis, categorized skill roadmaps, verified high-impact bullet improvements, and fully tailored resume.
        </p>
        <div style="margin-top: 10px;">
            <span class="mentor-badge">🔍 Grounded Gap Analysis</span>
            <span class="mentor-badge">📈 Missing Skill Learning Paths</span>
            <span class="mentor-badge">✍️ Impact Bullet Improvements</span>
            <span class="mentor-badge">📄 ATS Job-Tailored Resume</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    profile = st.session_state.get("parsed_profile")
    matched_jobs_list = st.session_state.get("matched_jobs") or []

    if not profile:
        st.warning("⚠️ Please upload a resume above to begin generating CV improvements.")
    else:
        # Candidate banner
        cand_name = profile.get("name") or "Candidate"
        cand_role = profile.get("target_role")
        has_cand_role = bool(cand_role and str(cand_role).strip().lower() not in ("none", "null", "not specified", "target role not specified"))
        display_cand_role = str(cand_role).strip() if has_cand_role else "Target role not specified"
        role_label = f" ({display_cand_role})" if has_cand_role else ""
        st.markdown(f"**Active Resume:** `{cand_name}`{role_label}")

        # -------------------------------------------------------------------
        # Target Job Application & Custom Description Option
        # -------------------------------------------------------------------
        selected_target_job = st.session_state.get("selected_target_job")
        has_picked_target = selected_target_job is not None

        if has_picked_target:
            t_title = selected_target_job.get("job_title", "Untitled Role")
            t_comp = selected_target_job.get("company", "Company")
            t_loc = selected_target_job.get("location", "")
            t_skills = selected_target_job.get("skills", "")
            t_rank = selected_target_job.get("rank", "")
            rank_str = f"Rank #{t_rank} • " if t_rank else ""

            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 18px; margin: 12px 0 14px 0;">
                <div style="font-weight: 700; color: #166534; font-size: 1.05rem;">
                    🎯 Selected Target Role: {t_title} at {t_comp}
                </div>
                <div style="font-size: 0.85rem; color: #15803d; margin-top: 4px;">
                    {rank_str}{t_loc} {f'| Required Skills: {t_skills}' if t_skills else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            enable_custom_paste = st.checkbox(
                "📝 Paste custom job description instead",
                value=False,
                help="Enable this if you wish to paste a different custom job description instead of using the selected target role.",
            )
        else:
            st.info("💡 You can select a target role directly from the **Job Matches** tab, or paste a custom job description below.")
            enable_custom_paste = True

        selected_job_input: Optional[Any] = None

        if enable_custom_paste:
            st.markdown("##### 📝 Paste Custom Job Description")
            custom_title = st.text_input(
                "Target Job Title (Optional)",
                placeholder="e.g. Senior Backend Engineer",
                key="cv_custom_job_title",
            )
            custom_desc = st.text_area(
                "Job Description / Requirements *",
                placeholder="Paste the full job description or key requirements here...",
                height=180,
                key="cv_custom_job_desc",
            )

            if custom_desc.strip():
                if custom_title.strip():
                    selected_job_input = {
                        "job_title": custom_title.strip(),
                        "description": custom_desc.strip(),
                    }
                else:
                    selected_job_input = custom_desc.strip()
        else:
            selected_job_input = selected_target_job

        gen_cv_btn = st.button("🚀 Generate CV Improvements & Tailored Resume", type="primary")

        # -------------------------------------------------------------------
        # Generation Execution
        # -------------------------------------------------------------------
        if gen_cv_btn:
            if not selected_job_input:
                st.error("Please provide or select a target job description.")
            else:
                with st.status("🛠️ Generating Tailored CV Improvements...", expanded=True) as cv_status:
                    try:
                        cv_status.update(label="🛡️ Validating inputs and safety guardrails...", state="running")
                        is_valid, val_msg = validate_cv_generator_inputs(profile, selected_job_input)
                        if not is_valid:
                            raise ValueError(val_msg)

                        cv_status.update(label="🤖 Running 5-step Gemini CV tailoring pipeline...", state="running")
                        cv_result: CVImprovementResult = generate_cv_improvements(
                            resume_profile=profile,
                            job_input=selected_job_input,
                        )

                        st.session_state["cv_improvement_result"] = cv_result
                        st.session_state["cv_target_job"] = cv_result.target_job

                        cv_status.update(
                            label=f"✅ CV Improvements generated successfully! (Status: {cv_result.status})",
                            state="complete",
                            expanded=False,
                        )

                    except Exception as gen_err:
                        cv_status.update(label="❌ CV generation error occurred.", state="error", expanded=True)
                        st.error(f"Generation error: {gen_err}")
                        logger.exception("CV generation pipeline error: %s", gen_err)

        # -------------------------------------------------------------------
        # Results Display (4 Dedicated Tabs)
        # -------------------------------------------------------------------
        cv_result: Optional[CVImprovementResult] = st.session_state.get("cv_improvement_result")

        if cv_result is not None:
            tj = cv_result.target_job
            job_name = f"{tj.title} at {tj.company}" if (tj.title and tj.company) else (tj.title or "Target Job")

            # Target Job & Score Banner
            match_pct = int((cv_result.comparison.overall_match_score if cv_result.comparison else 0.75) * 100)
            st.markdown(f"""
            <div class="cv-score-banner">
                <div>
                    <h3 style="margin: 0; color: #ffffff;">Target Role: {job_name}</h3>
                    <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.95rem;">
                        Pipeline Status: <strong>{cv_result.status.upper()}</strong> | Fact-Grounded Tailoring
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Fit Score</div>
                    <div class="score-badge">{match_pct}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Warnings Callout if any
            if cv_result.warnings:
                with st.expander("⚠️ Candidate Verification Advisory Notes", expanded=False):
                    for w in cv_result.warnings:
                        st.markdown(f"- {w}")

            # 4 Sub-Tabs for CV Improvement Results
            subtab_gap, subtab_bullets, subtab_summary, subtab_resume = st.tabs([
                "📊 Gap Analysis & Skills",
                "✍️ Bullet Point Enhancements",
                "📝 Tailored Summary",
                "📄 Complete Tailored Resume",
            ])

            # ---------------------------------------------------------------
            # SUB-TAB 1: Gap Analysis & Missing Skills
            # ---------------------------------------------------------------
            with subtab_gap:
                comp = cv_result.comparison
                skills = cv_result.missing_skills

                if comp:
                    st.markdown("#### Overall Alignment Assessment")
                    st.info(comp.overall_assessment)

                    col_matched, col_unmet = st.columns(2)
                    with col_matched:
                        st.markdown("##### ✅ Demonstrated Strengths & Matched Requirements")
                        if comp.matched_requirements:
                            for m in comp.matched_requirements:
                                st.markdown(f"""
                                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px; margin-bottom: 8px; color: #064e3b;">
                                    <div style="color: #14532d; font-weight: 700; font-size: 0.95rem;">
                                        {m.requirement} <span style="font-size: 0.72rem; background: #dcfce7; color: #166534; padding: 2px 6px; border-radius: 4px; font-weight: 800;">{m.match_strength.upper()}</span>
                                    </div>
                                    <div style="color: #166534; font-size: 0.88rem; margin-top: 4px;">
                                        <strong>Evidence:</strong> {m.evidence}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No direct matches identified.")

                    with col_unmet:
                        st.markdown("##### ⚠️ Missing Requirements / Gaps")
                        if comp.unmet_requirements:
                            for u in comp.unmet_requirements:
                                st.markdown(f"""
                                <div style="background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 12px; margin-bottom: 8px; color: #431407;">
                                    <div style="color: #9a3412; font-weight: 700; font-size: 0.95rem;">
                                        {u.requirement}
                                    </div>
                                    <div style="color: #c2410c; font-size: 0.8rem; font-weight: 700; margin-top: 2px;">
                                        Type: {u.gap_type.replace('_', ' ').title()}
                                    </div>
                                    <div style="color: #7c2d12; font-size: 0.88rem; margin-top: 4px;">
                                        {u.explanation}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No unmet requirements.")

                st.divider()

                if skills:
                    st.markdown("#### Skill Development Roadmap")
                    gap_col, evid_col = st.columns(2)

                    with gap_col:
                        st.markdown("##### 🎯 Confirmed Skill Gaps (Learning Paths)")
                        if skills.confirmed_skill_gaps:
                            for g in skills.confirmed_skill_gaps:
                                st.markdown(f"""
                                <div class="gap-card-confirmed" style="background: #fff1f2; border-left: 5px solid #e11d48; border: 1px solid #fecdd3; border-left-width: 5px; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; color: #4c0519;">
                                    <div style="font-size: 1.05rem; font-weight: 700; color: #9f1239; margin-bottom: 6px;">
                                        {g.skill} <span style="font-size: 0.72rem; background: #ffe4e6; color: #be123c; padding: 2px 8px; border-radius: 4px; font-weight: 800; vertical-align: middle;">{g.importance.upper()}</span>
                                    </div>
                                    <div style="color: #4c0519; font-size: 0.92rem; line-height: 1.5; margin-bottom: 6px;">
                                        <strong style="color: #881337;">Recommended Learning:</strong> {g.learning_recommendation}
                                    </div>
                                    <div style="color: #be123c; font-size: 0.85rem; font-weight: 700;">
                                        ⏱️ Estimated Effort: {g.estimated_effort}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No major confirmed technical gaps!")

                    with evid_col:
                        st.markdown("##### 💡 Missing Evidence (Resume Clarification)")
                        if skills.missing_evidence_items:
                            for e in skills.missing_evidence_items:
                                st.markdown(f"""
                                <div class="gap-card-evidence" style="background: #eff6ff; border-left: 5px solid #2563eb; border: 1px solid #bfdbfe; border-left-width: 5px; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; color: #172554;">
                                    <div style="font-size: 1.05rem; font-weight: 700; color: #1e40af; margin-bottom: 6px;">
                                        {e.skill}
                                    </div>
                                    <div style="color: #1e293b; font-size: 0.92rem; line-height: 1.5; margin-bottom: 6px;">
                                        <strong style="color: #1e3a8a;">Likely Context:</strong> {e.likely_context}
                                    </div>
                                    <div style="color: #1d4ed8; font-size: 0.92rem; line-height: 1.5;">
                                        <strong style="color: #1e40af;">How to Evidence:</strong> {e.resume_improvement_suggestion}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("All evidenced skills are clearly documented.")

            # ---------------------------------------------------------------
            # SUB-TAB 2: Bullet Point Enhancements
            # ---------------------------------------------------------------
            with subtab_bullets:
                bullets = cv_result.bullet_improvements
                if bullets and bullets.improved_bullets:
                    st.markdown("#### Side-by-Side Bullet Point Enhancements")
                    st.caption("Rewritten for maximum role relevance while strictly preserving verified achievements.")

                    for b in bullets.improved_bullets:
                        st.markdown(f"""
                        <div class="bullet-comp-card" style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 16px; margin-bottom: 16px; color: #0f172a;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <strong style="color: #0f172a; font-size: 0.95rem;">{b.entry_title} {f'at {b.entry_company}' if b.entry_company else ''}</strong>
                                <span class="change-tag" style="background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">{b.section.upper()}</span>
                            </div>
                            <div class="bullet-orig" style="color: #475569; font-size: 0.92rem; margin-bottom: 8px; padding-left: 8px; border-left: 3px solid #94a3b8;">
                                <strong style="color: #334155;">Original:</strong> {b.original_bullet}
                            </div>
                            <div class="bullet-improved" style="color: #0f172a; font-size: 0.95rem; font-weight: 600; margin-bottom: 8px; padding-left: 8px; border-left: 3px solid #6366f1;">
                                <strong style="color: #4338ca;">Improved:</strong> {b.improved_bullet}
                            </div>
                            <div style="font-size: 0.85rem; color: #475569; margin-top: 6px;">
                                <em>Why: {b.change_explanation}</em>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No bullet improvements generated or all original bullets met quality standards.")

            # ---------------------------------------------------------------
            # SUB-TAB 3: Rewritten Summary
            # ---------------------------------------------------------------
            with subtab_summary:
                summ = cv_result.summary_rewrite
                if summ and summ.rewritten_summary:
                    st.markdown("#### Tailored Professional Summary")

                    sum_col1, sum_col2 = st.columns(2)
                    with sum_col1:
                        st.markdown("##### 📄 Original Summary")
                        orig_s = summ.original_summary or profile.get("professional_summary") or "No original summary provided."
                        st.markdown(f"""
                        <div style="background: #f8fafc !important; border-left: 5px solid #94a3b8 !important; border: 1px solid #e2e8f0; border-left-width: 5px; padding: 18px; border-radius: 10px; color: #334155 !important; line-height: 1.65; font-size: 0.95rem;">
                            <p style="color: #334155 !important; margin: 0;">{orig_s}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with sum_col2:
                        st.markdown("##### 🚀 Tailored Executive Summary")
                        st.markdown(f"""
                        <div style="background: #ffffff !important; border-left: 5px solid #6366f1 !important; border: 1px solid #cbd5e1; border-left-width: 5px; padding: 18px; border-radius: 10px; color: #0f172a !important; line-height: 1.65; font-size: 0.98rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            <p style="color: #0f172a !important; margin: 0; font-weight: 500;">{summ.rewritten_summary}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if summ.key_facts_used:
                        st.markdown("##### Fact-Traceability Notes")
                        fact_chips = "".join([f'<span class="skill-tag">{f}</span>' for f in summ.key_facts_used])
                        st.markdown(f"<div>{fact_chips}</div>", unsafe_allow_html=True)
                else:
                    st.info("No summary rewrite generated.")

            # ---------------------------------------------------------------
            # SUB-TAB 4: Complete Tailored Resume & Download
            # ---------------------------------------------------------------
            with subtab_resume:
                tailored = cv_result.tailored_resume
                if tailored and tailored.tailored_resume_text:
                    st.markdown("#### Complete Tailored Resume")
                    st.markdown("""
                    <div class="grounded-banner">
                        🛡️ <strong>Strict Grounding Guarantee:</strong> This resume integrates your verified skills, improved bullet points,
                        relevant projects, and tailored executive summary. No unearned degrees, metrics, or technologies were fabricated.
                    </div>
                    """, unsafe_allow_html=True)

                    # Download button
                    cand_clean_name = (profile.get("name") or "tailored_resume").replace(" ", "_").lower()
                    dl_col1, dl_col2 = st.columns([1, 4])
                    with dl_col1:
                        st.download_button(
                            label="📥 Download Resume (.txt)",
                            data=tailored.tailored_resume_text,
                            file_name=f"{cand_clean_name}_tailored_cv.txt",
                            mime="text/plain",
                            type="primary",
                        )

                    with st.expander("📋 Strategic Changes & Adjustments Made", expanded=False):
                        if tailored.changes_made:
                            for chg in tailored.changes_made:
                                st.markdown(f"- {chg}")

                    # Formatted vs Raw view
                    view_mode = st.radio("Resume View", ["📄 Formatted Resume", "📝 Plaintext / Markdown Editor"], horizontal=True)
                    if view_mode == "📄 Formatted Resume":
                        # Render in a single st.markdown call with .resume-paper container to prevent auto-closing div bug and guarantee dark-mode visibility
                        st.markdown(
                            f'<div class="resume-paper">\n\n{tailored.tailored_resume_text}\n\n</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.text_area(
                            "Resume Plaintext",
                            value=tailored.tailored_resume_text,
                            height=480,
                        )
                else:
                    st.info("Complete tailored resume is not available.")

    st.write("")
    render_back_button("tab2_bottom")


# ===========================================================================
# TAB 3: AI CAREER MENTOR (GROUNDED RAG CHATBOT)
# ===========================================================================
with main_tab_mentor:
    render_back_button("tab3_top")

    # 1. State initialization
    if "mentor_memory" not in st.session_state or not hasattr(st.session_state.mentor_memory, "get_conversation_turns"):
        old_mem = st.session_state.get("mentor_memory")
        existing_msgs = getattr(old_mem, "messages", []) if old_mem else []
        st.session_state.mentor_memory = ConversationMemory(max_turns=10)
        st.session_state.mentor_memory.messages = list(existing_msgs)
    mentor_mem: ConversationMemory = st.session_state.mentor_memory

    # 2. Mentor Hero Card
    st.markdown("""
    <div class="model-hero-card model-hero-career-mentor">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <span class="model-category-badge">CORE AI MODEL 3 • MULTI-SOURCE KNOWLEDGE ASSISTANT</span>
                <h2 class="model-hero-title">🤖 AI Career Mentor</h2>
                <p class="model-hero-desc">
                    Intelligent career guidance grounded in verified career roadmaps and real-world job market data.
                    Ask about preparation roadmaps, role requirements, missing skills, or technical interview strategies.
                </p>
                <div style="margin-top: 10px;">
                    <span class="mentor-badge">🛡️ LangChain LCEL RAG Chain</span>
                    <span class="mentor-badge">💡 Grounded Career Roadmaps</span>
                    <span class="mentor-badge">⚡ Zero Meta-Talk Direct Answers</span>
                    <span class="mentor-badge">🔒 Strict Hallucination Refusal</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Top Action / Context Row
    col_ctx, col_clear = st.columns([4, 1], vertical_alignment="center")

    with col_clear:
        if st.button("🗑️ Clear Chat", help="Reset conversation history for this session", use_container_width=True):
            mentor_mem.clear()
            st.rerun()

    profile_for_mentor = st.session_state.get("parsed_profile")
    active_target_job = st.session_state.get("selected_target_job")
    tailor_to_profile = False
    with col_ctx:
        if profile_for_mentor and profile_for_mentor.get("name"):
            c_name = profile_for_mentor.get("name")
            c_role_raw = profile_for_mentor.get("target_role")
            c_role = (
                str(c_role_raw).strip()
                if c_role_raw and str(c_role_raw).strip().lower() not in ("none", "null", "not specified", "target role not specified")
                else "Target role not specified"
            )
            tailor_to_profile = st.checkbox(
                f"📌 Personalize guidance with your resume ({c_name} — {c_role})",
                value=True,
                key="mentor_personalize_guidance_chk",
                help="Provides your active profile details to the mentor for personalized advice.",
            )
        else:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; font-size: 0.88rem; color: #64748b; background: #f8fafc; border: 1px solid #e2e8f0; padding: 6px 14px; border-radius: 8px;">
                    <span>💡 <em>No resume uploaded. Answering general career, roadmap, and preparation questions.</em></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if active_target_job:
        t_comp = active_target_job.get("company", "Company")
        t_role = active_target_job.get("job_title") or active_target_job.get("title", "Role")
        t_loc = active_target_job.get("location", "")
        loc_str = f" • 📍 {t_loc}" if t_loc else ""
        st.markdown(
            f"""
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 8px 16px; margin: 10px 0 16px 0; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.1rem;">🎯</span>
                <span style="font-size: 0.9rem; color: #1e3a8a;">
                    <strong>Active Target Job Selected:</strong> <strong>{t_role}</strong> at <strong>{t_comp}</strong>{loc_str}
                    <span style="color: #3b82f6; font-size: 0.82rem; margin-left: 8px;">(Questions like <em>"how to get a job in this company"</em> will target {t_comp})</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Quick Starter Prompts
    st.markdown("##### 💡 Suggested Career Questions")
    quick_cols = st.columns(5)
    starter_queries = [
        ("🚀 Backend Roadmap", "What is the complete roadmap and required tech stack for a Backend Developer?"),
        ("📊 Data Scientist Skills", "What core skills, tools, and algorithms are essential for a Data Scientist?"),
        ("☁️ Cloud Engineering", "What certifications, tools, and skills are required to become a Cloud Engineer?"),
        ("🛡️ Cybersecurity Path", "What are the core domains, certifications, and skills needed for a Cybersecurity Analyst?"),
        ("💼 Fresher SDE Prep", "How should a fresher software engineer prepare for technical interviews and coding rounds?"),
    ]

    clicked_query = None
    for idx, (label, query_text) in enumerate(starter_queries):
        with quick_cols[idx]:
            if st.button(label, key=f"quick_btn_{idx}", use_container_width=True):
                clicked_query = query_text

    # 5. Handle Pending Quick Query or User Chat Input
    active_query = clicked_query

    chat_input_val = st.chat_input("Ask career guidance, roadmap, required skills, or interview tips...")
    if chat_input_val and chat_input_val.strip():
        active_query = chat_input_val.strip()

    # 6. Execute Query if Active (Runs at TOP as Current Conversation)
    if active_query:
        cand_prof = profile_for_mentor if tailor_to_profile else None
        curr_matched_jobs = (st.session_state.get("matched_jobs") or [])[:3] if cand_prof else None
        current_target_job = st.session_state.get("selected_target_job")

        st.markdown("##### 💬 Current Conversation")
        # Render immediate user bubble
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Consulting Career Knowledge Base & Synthesizing Guidance..."):
                prior_history = mentor_mem.get_history()
                result = ask_career_mentor(
                    query=active_query,
                    conversation_history=prior_history,
                    candidate_profile=cand_prof,
                    matched_jobs=curr_matched_jobs,
                    selected_target_job=current_target_job,
                )
                answer_text = strip_source_citations(result.get("answer", ""))
                sources = result.get("sources", [])
                status = result.get("status", "valid")

                # Store user message and assistant response in memory
                mentor_mem.add_user_message(active_query)
                mentor_mem.add_assistant_message(answer_text, sources=sources, status=status)

                # Render answer
                st.markdown(answer_text)
                if status == "unsupported":
                    st.caption("🛡️ *Strict grounding: Out-of-corpus query rejected without hallucinating.*")

        st.rerun()

    # 7. Render Conversation History: Current at TOP, Past at BOTTOM
    if hasattr(mentor_mem, "get_conversation_turns"):
        turns = mentor_mem.get_conversation_turns()
        current_turn = mentor_mem.get_current_turn()
        past_turns = mentor_mem.get_past_turns()
    else:
        raw_msgs = mentor_mem.get_messages() if hasattr(mentor_mem, "get_messages") else getattr(mentor_mem, "messages", [])
        turns = []
        c_turn = []
        for m in raw_msgs:
            if getattr(m, "role", "") == "user" and c_turn:
                turns.append(c_turn)
                c_turn = [m]
            else:
                c_turn.append(m)
        if c_turn:
            turns.append(c_turn)
        current_turn = turns[-1] if turns else []
        past_turns = list(reversed(turns[:-1])) if len(turns) > 1 else []

    if not turns:
        st.markdown("""
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 32px 24px; text-align: center; margin: 20px 0; color: #64748b;">
            <div style="font-size: 2rem; margin-bottom: 8px;">💬</div>
            <div style="font-weight: 700; color: #1e293b; font-size: 1.05rem;">No questions asked yet in this session.</div>
            <div style="font-size: 0.9rem; margin-top: 4px;">Click any suggested topic above or type your career question below to consult the AI Mentor.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        current_turn = mentor_mem.get_current_turn()
        past_turns = mentor_mem.get_past_turns()

        # (A) Current Conversation at TOP
        st.markdown("##### 💬 Current Conversation")
        for msg in current_turn:
            if msg.role == "user":
                with st.chat_message("user"):
                    st.markdown(msg.content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(strip_source_citations(msg.content))
                    if msg.status == "unsupported":
                        st.caption("🛡️ *Strict grounding: Out-of-corpus query rejected without hallucinating.*")

        # (B) Past Conversations at BOTTOM
        if past_turns:
            st.markdown("---")
            st.markdown("##### 📜 Past Conversations")
            for turn in past_turns:
                for msg in turn:
                    if msg.role == "user":
                        with st.chat_message("user"):
                            st.markdown(msg.content)
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(strip_source_citations(msg.content))
                            if msg.status == "unsupported":
                                st.caption("🛡️ *Strict grounding: Out-of-corpus query rejected without hallucinating.*")

