from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


MAX_WINDOW_SIZE = 6


@dataclass
class SessionInfo:
    session_id: str
    create_time: int = field(default_factory=lambda: int(time.time() * 1000))
    _history: list[dict[str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_message(self, user_question: str, ai_answer: str) -> None:
        with self._lock:
            self._history.append({'role': 'user', 'content': user_question})
            self._history.append({'role': 'assistant', 'content': ai_answer})
            max_messages = MAX_WINDOW_SIZE * 2
            while len(self._history) > max_messages:
                self._history = self._history[2:]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    def history(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._history)

    def pair_count(self) -> int:
        with self._lock:
            return len(self._history) // 2


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionInfo] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: Optional[str]) -> SessionInfo:
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = SessionInfo(session_id=sid)
            return self._sessions[sid]

    def get(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            return self._sessions.get(session_id)
