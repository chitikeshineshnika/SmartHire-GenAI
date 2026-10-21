"""
Unit tests for AI Career Mentor session memory (src.mentor.memory).
"""

from unittest.mock import MagicMock, patch
import pytest

from src.mentor.memory import ChatMessage, ConversationMemory
from src.mentor.rag_chain import ask_career_mentor


class TestConversationMemory:
    """Tests for in-memory session tracking, trimming, and serialization."""

    def test_add_messages_and_retrieval(self):
        memory = ConversationMemory(max_turns=3)
        assert len(memory.get_messages()) == 0

        memory.add_user_message("What is a DevOps Engineer?")
        memory.add_assistant_message("A DevOps engineer bridges software development and IT operations.")

        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What is a DevOps Engineer?"
        assert messages[1].role == "assistant"
        assert "DevOps engineer" in messages[1].content

        history = memory.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "What is a DevOps Engineer?"}

    def test_sliding_window_trimming(self):
        memory = ConversationMemory(max_turns=2)
        # Add 3 turns (6 messages)
        for i in range(1, 4):
            memory.add_user_message(f"Question {i}")
            memory.add_assistant_message(f"Answer {i}")

        messages = memory.get_messages()
        # Should be trimmed to 2 turns (4 messages): turns 2 and 3
        assert len(messages) == 4
        assert messages[0].content == "Question 2"
        assert messages[1].content == "Answer 2"
        assert messages[2].content == "Question 3"
        assert messages[3].content == "Answer 3"

    def test_clear_session(self):
        memory = ConversationMemory()
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there")
        assert len(memory.get_messages()) == 2

        memory.clear()
        assert len(memory.get_messages()) == 0
        assert memory.get_history() == []

    def test_serialization_and_deserialization(self):
        memory = ConversationMemory(max_turns=5)
        memory.add_user_message("How to prepare for Data Analyst interviews?")
        memory.add_assistant_message(
            "Focus on SQL and Excel.",
            sources=[{"source": "data_analyst.md", "source_type": "career_note"}],
        )

        dict_list = memory.to_dict_list()
        assert len(dict_list) == 2
        assert dict_list[0]["role"] == "user"
        assert dict_list[1]["role"] == "assistant"
        assert dict_list[1]["sources"][0]["source"] == "data_analyst.md"

        # Restore from dict list
        restored = ConversationMemory.from_dict_list(dict_list, max_turns=5)
        assert len(restored.get_messages()) == 2
        assert restored.get_messages()[1].sources[0]["source"] == "data_analyst.md"

    def test_current_and_past_turns_ordering(self):
        memory = ConversationMemory(max_turns=5)
        # Empty state
        assert memory.get_current_turn() == []
        assert memory.get_past_turns() == []

        # Turn 1
        memory.add_user_message("Query 1")
        memory.add_assistant_message("Answer 1")
        assert len(memory.get_current_turn()) == 2
        assert memory.get_current_turn()[0].content == "Query 1"
        assert memory.get_past_turns() == []

        # Turn 2
        memory.add_user_message("Query 2")
        memory.add_assistant_message("Answer 2")
        assert memory.get_current_turn()[0].content == "Query 2"
        assert len(memory.get_past_turns()) == 1
        assert memory.get_past_turns()[0][0].content == "Query 1"

        # Turn 3
        memory.add_user_message("Query 3")
        memory.add_assistant_message("Answer 3")
        # Current turn should be Turn 3
        assert memory.get_current_turn()[0].content == "Query 3"
        # Past turns should be in reverse-chronological order: Turn 2, then Turn 1
        past = memory.get_past_turns()
        assert len(past) == 2
        assert past[0][0].content == "Query 2"
        assert past[1][0].content == "Query 1"


class TestMultiTurnContextResolution:
    """Verifies that follow-up questions resolve referential context from memory."""

    @patch("src.mentor.rag_chain.retrieve_mentor_evidence")
    def test_follow_up_query_contextualization(self, mock_retrieve):
        mock_retrieve.return_value = [
            {
                "score": 0.82,
                "text": "Level 1: Language basics, SQL, REST APIs.",
                "source_type": "career_note",
                "source": "backend_developer.md",
                "title": "Backend Developer",
                "section": "Skill roadmap",
            }
        ]

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = (
            "**Answer**\nLevel 1 fundamentals focus on language basics, SQL, and REST APIs.\n\n"
            "**Key Points**\n- Learn HTTP and SQL\n\n"
            "**Sources**\n- backend_developer.md"
        )
        mock_client.models.generate_content.return_value = mock_resp

        memory = ConversationMemory()
        memory.add_user_message("I am interested in becoming a Backend Developer.")
        memory.add_assistant_message("A backend developer builds the server side of an application.")

        # Follow-up with relative pronoun 'this role'
        follow_up_query = "What are the Level 1 fundamentals for this role?"
        result = ask_career_mentor(
            query=follow_up_query,
            conversation_history=memory.get_history(),
            client_override=mock_client,
        )

        assert result["status"] == "valid"
        # Verify retrieve_mentor_evidence was called with enriched query containing context
        called_query = (
            mock_retrieve.call_args[0][0]
            if (mock_retrieve.call_args and mock_retrieve.call_args[0])
            else mock_retrieve.call_args.kwargs.get("query", "")
        )
        assert "Backend Developer" in called_query
        assert "Level 1 fundamentals" in called_query
