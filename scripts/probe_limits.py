"""Đo hạn mức thật của provider đang cấu hình, thay vì tin vào tài liệu.

Provider tương thích OpenAI thường trả hạn mức trong header response. Script gọi
một request nhỏ nhất có thể rồi đọc các header đó, để biết trần token/phút thật
trước khi quyết định chạy đủ 50 case.

    uv run scripts/probe_limits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.llm import (  # noqa: E402
    BASE_URL,
    MODEL_NAME,
    MODEL_PROVIDER,
    PROVIDER_PROFILES,
    _PROFILE,
)

HEADERS_OF_INTEREST = [
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-tokens",
    "retry-after",
]

# Ước lượng của pipeline hiện tại, lấy từ trace thật.
TOKENS_PER_CASE = 2611
CASES = 50


def main() -> int:
    import os

    key_env = _PROFILE["api_key_env"]
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    if key_env and not api_key:
        raise SystemExit(f"Thiếu {key_env} trong .env cho provider {MODEL_PROVIDER!r}")

    print(f"Provider : {MODEL_PROVIDER}")
    print(f"Model    : {MODEL_NAME} ({_PROFILE['param_size_b']}B)")
    print(f"Base URL : {BASE_URL}\n")

    try:
        response = httpx.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1},
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Không gọi được: {exc}")

    print(f"HTTP {response.status_code}")
    if response.status_code >= 400:
        print(response.text[:400])
        return 1

    found = False
    for header in HEADERS_OF_INTEREST:
        value = response.headers.get(header)
        if value:
            found = True
            print(f"  {header:<34}{value}")
    if not found:
        print("  (provider này không trả header hạn mức — phải thử thực tế)")
        return 0

    tpm = response.headers.get("x-ratelimit-limit-tokens")
    if tpm and tpm.isdigit():
        total = TOKENS_PER_CASE * CASES
        minutes = total / int(tpm)
        print(f"\nPipeline cần ~{total:,} token cho {CASES} case.")
        print(f"Ở trần {int(tpm):,} token/phút -> tối thiểu {minutes:.1f} phút.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
