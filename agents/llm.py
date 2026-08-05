"""Client LLM dùng chung cho mọi agent.

Ràng buộc đề bài: mỗi agent chỉ được dùng model <= 10B tham số.
Model đang dùng khai báo ngay tại đây (không giấu trong .env) để người chấm đọc được.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .ratelimit import estimate_tokens, get_limiter

load_dotenv()

# === Khai báo model (yêu cầu mục 9.4 của đề) ===
MODEL_NAME = "llama-3.1-8b-instant"
MODEL_PARAM_SIZE_B = 8.0  # 8B <= 10B: thỏa ràng buộc
MODEL_PROVIDER = "groq"
BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_TEMPERATURE = 0.0  # cần tái lập được kết quả giữa các lần chạy
DEFAULT_MAX_TOKENS = 400  # các agent chỉ trả JSON ngắn; đặt rộng chỉ tốn hạn mức TPM
MAX_RETRIES = 6


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    content: dict[str, Any]
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMError(
                "Thiếu GROQ_API_KEY. Tạo file .env ở root repo với dòng: GROQ_API_KEY=gsk_..."
            )
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


def _extract_json(text: str) -> dict[str, Any]:
    """Model 8B đôi khi bọc JSON trong văn xuôi hoặc code fence; bóc lớp đó ra."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError(f"Không tìm thấy JSON trong phản hồi: {text[:200]!r}")
    return json.loads(text[start : end + 1])


_RETRY_HINT = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Bóc thời gian chờ khỏi lỗi 429 của Groq ('Please try again in 11.92s')."""
    if getattr(exc, "status_code", None) != 429 and "429" not in str(exc):
        return None
    header_value = None
    response = getattr(exc, "response", None)
    if response is not None:
        header_value = getattr(response, "headers", {}).get("retry-after")
    if header_value:
        try:
            return float(header_value)
        except ValueError:
            pass
    match = _RETRY_HINT.search(str(exc))
    if match:
        return float(match.group(1))
    return 15.0  # 429 nhưng không nói rõ: chờ trọn một cửa sổ cho chắc


def call_json(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> LLMResult:
    """Gọi model và bắt buộc trả về một JSON object.

    Retry có backoff cho cả lỗi mạng/rate-limit lẫn trường hợp model trả JSON hỏng.
    """
    client = get_client()
    limiter = get_limiter()
    estimated = estimate_tokens(system_prompt, user_prompt, max_tokens)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        limiter.acquire(estimated)
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            parsed = _extract_json(raw)
            usage = response.usage
            actual = (getattr(usage, "total_tokens", 0) or 0) if usage else 0
            if actual:
                limiter.record_actual(estimated, actual)
            return LLMResult(
                content=parsed,
                raw_text=raw,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - retry mọi lỗi tạm thời
            last_error = exc
            if attempt >= MAX_RETRIES - 1:
                break
            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                # Groq nói rõ phải chờ bao lâu: chặn luôn cả các thread khác,
                # nếu không chúng sẽ lao vào và cùng ăn 429.
                limiter.penalize(retry_after)
                time.sleep(retry_after + 0.5)
            else:
                time.sleep(2**attempt)

    raise LLMError(f"Gọi LLM thất bại sau {MAX_RETRIES} lần: {last_error}") from last_error


def model_metadata() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "parameter_size_b": MODEL_PARAM_SIZE_B,
        "provider": MODEL_PROVIDER,
        "base_url": BASE_URL,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
