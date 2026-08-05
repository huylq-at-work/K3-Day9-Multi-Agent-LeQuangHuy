"""Bộ điều tiết token dùng chung cho mọi thread.

Gói free của Groq giới hạn theo **tokens per minute**, không phải số request.
Chạy 4 case song song rất dễ đụng trần: mỗi case tiêu ~2.700 token, mà trần chỉ 6.000/phút.

Dùng token bucket kiểu sliding window: trước mỗi lần gọi, agent phải xin trước
một lượng token ước lượng; nếu cửa sổ 60 giây gần nhất đã đầy thì chờ tới khi
bản ghi cũ nhất hết hạn.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

# Trần của gói free. Chừa lại một phần vì token ước lượng không bao giờ khớp tuyệt đối.
DEFAULT_TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "6000"))
SAFETY_RATIO = 0.85
WINDOW_SECONDS = 60.0


class TokenRateLimiter:
    def __init__(self, tpm_limit: int = DEFAULT_TPM_LIMIT) -> None:
        self.budget = int(tpm_limit * SAFETY_RATIO)
        self._events: deque[tuple[float, int]] = deque()  # (thời điểm, số token)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def _used_locked(self, now: float) -> int:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()
        return sum(tokens for _, tokens in self._events)

    def acquire(self, estimated_tokens: int) -> None:
        """Chặn cho tới khi còn đủ hạn mức trong cửa sổ 60 giây."""
        estimated_tokens = min(estimated_tokens, self.budget)
        with self._condition:
            while True:
                now = time.monotonic()
                used = self._used_locked(now)
                if used + estimated_tokens <= self.budget:
                    self._events.append((now, estimated_tokens))
                    return
                # Chờ tới khi bản ghi cũ nhất rời khỏi cửa sổ.
                wait = WINDOW_SECONDS - (now - self._events[0][0]) + 0.05
                self._condition.wait(timeout=max(wait, 0.1))

    def record_actual(self, estimated: int, actual: int) -> None:
        """Bù chênh lệch sau khi biết số token thật, để cửa sổ bám sát thực tế."""
        delta = actual - estimated
        if delta == 0:
            return
        with self._condition:
            self._events.append((time.monotonic(), delta))
            self._condition.notify_all()

    def penalize(self, seconds: float) -> None:
        """Bị 429: coi như cửa sổ đã đầy trong `seconds` tới, chặn mọi thread khác."""
        with self._condition:
            now = time.monotonic()
            hold_until = now - WINDOW_SECONDS + seconds
            self._events.append((hold_until, self.budget))
            self._condition.notify_all()


_limiter: TokenRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> TokenRateLimiter:
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            _limiter = TokenRateLimiter()
    return _limiter


def estimate_tokens(system_prompt: str, user_prompt: str, max_completion: int) -> int:
    """Ước lượng thô: ~3.5 ký tự/token cho văn bản lẫn tiếng Việt, cộng phần sinh ra."""
    chars = len(system_prompt) + len(user_prompt)
    return int(chars / 3.5) + max_completion
