"""
Safety, guardrails, and grounding verification module for SmartHire GenAI.
"""

from src.safety.guardrails import (
    ALLOWED_TOPICS,
    MAX_INPUT_LENGTH,
    MAX_JOB_INPUT_LENGTH,
    MENTOR_MAX_INPUT_LENGTH,
    MENTOR_OFF_TOPIC_REJECTION_MESSAGE,
    MENTOR_UNSAFE_REJECTION_MESSAGE,
    MIN_JOB_INPUT_LENGTH,
    check_mentor_input_guardrail,
    detect_prompt_injection,
    is_career_related,
    is_unsafe_mentor_request,
    validate_cv_generator_inputs,
    validate_input,
    validate_mentor_input,
    verify_bullet_grounding,
    verify_summary_grounding,
    verify_tailored_resume,
)

__all__ = [
    "ALLOWED_TOPICS",
    "MAX_INPUT_LENGTH",
    "MAX_JOB_INPUT_LENGTH",
    "MENTOR_MAX_INPUT_LENGTH",
    "MENTOR_OFF_TOPIC_REJECTION_MESSAGE",
    "MENTOR_UNSAFE_REJECTION_MESSAGE",
    "MIN_JOB_INPUT_LENGTH",
    "check_mentor_input_guardrail",
    "detect_prompt_injection",
    "is_career_related",
    "is_unsafe_mentor_request",
    "validate_cv_generator_inputs",
    "validate_input",
    "validate_mentor_input",
    "verify_bullet_grounding",
    "verify_summary_grounding",
    "verify_tailored_resume",
]
