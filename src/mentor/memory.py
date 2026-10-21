"""
Session Conversation Memory module for SmartHire AI Career Mentor.

Provides:
1. In-session multi-turn dialogue management (turns, roles, timestamps).
2. Sliding context window to avoid token overflow.
3. Easy serialization/deserialization for Streamlit session_state (`st.session_state.mentor_memory`).
4. Structured query contextualization to resolve follow-up references ("this role", "it", etc.).
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Represents a single message in a conversation session."""
    role: str  # "user" | "assistant"
    content: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    status: Optional[str] = "valid"
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class ConversationMemory:
    """
    Manages multi-turn conversation memory for the current user's session.

    Guiding Principles:
    1. Session Isolation: Retains conversation only within the active session.
    2. Context vs. Truth: History is used to resolve pronouns and conversational context,
       never as a substitute for fresh document retrieval.
    3. Window Management: Automatically trims oldest turns beyond `max_turns` to prevent token bloat.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[ChatMessage] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        cleaned = content.strip()
        if not cleaned:
            return
        self.messages.append(ChatMessage(role="user", content=cleaned))
        self._trim_history()

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        status: str = "valid",
    ) -> None:
        """Add the assistant's response to the conversation history."""
        cleaned = content.strip()
        self.messages.append(
            ChatMessage(
                role="assistant",
                content=cleaned,
                sources=sources or [],
                status=status,
            )
        )
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep only the most recent `max_turns` message pairs."""
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        """Return a simplified list of dicts [{'role': 'user', 'content': '...'}] for RAG prompt ingestion."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def get_messages(self) -> List[ChatMessage]:
        """Return the full list of ChatMessage instances."""
        return list(self.messages)

    def get_conversation_turns(self) -> List[List[ChatMessage]]:
        """
        Group messages into conversation turns (each turn containing a user message
        and its corresponding assistant response).
        """
        turns: List[List[ChatMessage]] = []
        current_turn: List[ChatMessage] = []
        for msg in self.messages:
            if msg.role == "user" and current_turn:
                turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)
        return turns

    def get_current_turn(self) -> List[ChatMessage]:
        """Return the messages belonging to the most recent (current) conversation turn."""
        turns = self.get_conversation_turns()
        return turns[-1] if turns else []

    def get_past_turns(self) -> List[List[ChatMessage]]:
        """
        Return all conversation turns prior to the current one,
        in reverse-chronological order (most recent past turn first).
        """
        turns = self.get_conversation_turns()
        return list(reversed(turns[:-1])) if len(turns) > 1 else []

    def get_last_user_query(self) -> Optional[str]:
        """Get the text of the most recent user query."""
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return None

    def clear(self) -> None:
        """Clear all conversation history in the current session."""
        self.messages.clear()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serialize memory to a list of dictionaries (suitable for Streamlit session_state)."""
        return [m.model_dump() for m in self.messages]

    @classmethod
    def from_dict_list(
        cls,
        data: List[Dict[str, Any]],
        max_turns: int = 10,
    ) -> "ConversationMemory":
        """Deserialize memory from a list of dictionaries."""
        mem = cls(max_turns=max_turns)
        for item in data:
            try:
                mem.messages.append(ChatMessage(**item))
            except Exception as e:
                logger.warning(f"Error restoring chat message: {e}")
        return mem
