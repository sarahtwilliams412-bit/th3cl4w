"""
ASCII Analysis Session Manager — manages conversation sessions with the LLM.

Each session tracks chat history, associated camera, and ASCII frames sent.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    text: str
    timestamp: float = field(default_factory=time.time)
    cam_id: Optional[int] = None
    ascii_frame: Optional[str] = None  # stored only for user messages with captures

    def to_dict(self) -> dict:
        d = {"role": self.role, "text": self.text, "timestamp": round(self.timestamp, 3)}
        if self.cam_id is not None:
            d["cam_id"] = self.cam_id
        if self.ascii_frame:
            d["has_frame"] = True
        return d


@dataclass
class AnalysisSession:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    cam_id: Optional[int] = None
    messages: list[ChatMessage] = field(default_factory=list)

    def add_user_message(self, text: str, cam_id: Optional[int] = None, ascii_frame: Optional[str] = None):
        msg = ChatMessage(role="user", text=text, cam_id=cam_id, ascii_frame=ascii_frame)
        self.messages.append(msg)
        self.last_active = time.time()
        if cam_id is not None:
            self.cam_id = cam_id

    def add_assistant_message(self, text: str):
        msg = ChatMessage(role="assistant", text=text)
        self.messages.append(msg)
        self.last_active = time.time()

    def get_history(self) -> list[dict]:
        """Get history in format suitable for LLM analyst."""
        return [{"role": m.role, "text": m.text} for m in self.messages]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": round(self.created_at, 3),
            "last_active": round(self.last_active, 3),
            "cam_id": self.cam_id,
            "message_count": len(self.messages),
            "messages": [m.to_dict() for m in self.messages],
        }

    def to_summary(self) -> dict:
        return {
            "id": self.id,
            "created_at": round(self.created_at, 3),
            "last_active": round(self.last_active, 3),
            "cam_id": self.cam_id,
            "message_count": len(self.messages),
        }


class SessionManager:
    """Manages analysis sessions with auto-cleanup."""

    MAX_SESSIONS = 50
    SESSION_TTL = 3600 * 2  # 2 hours

    def __init__(self):
        self._sessions: dict[str, AnalysisSession] = {}

    def create(self, cam_id: Optional[int] = None) -> AnalysisSession:
        self._cleanup()
        session = AnalysisSession(cam_id=cam_id)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[AnalysisSession]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str] = None, cam_id: Optional[int] = None) -> AnalysisSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create(cam_id=cam_id)

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[dict]:
        self._cleanup()
        return [s.to_summary() for s in sorted(
            self._sessions.values(), key=lambda s: s.last_active, reverse=True
        )]

    def _cleanup(self):
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s.last_active > self.SESSION_TTL]
        for sid in expired:
            del self._sessions[sid]
        # Evict oldest if over limit
        while len(self._sessions) > self.MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda s: s.last_active)
            del self._sessions[oldest.id]
