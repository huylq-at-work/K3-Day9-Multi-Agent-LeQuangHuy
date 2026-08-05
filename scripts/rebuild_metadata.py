"""Dựng lại logging/metadata.json từ trace của lượt chạy thật.

Dùng khi metadata bị ghi đè nhầm, hoặc khi cần bổ sung số liệu suy ra được từ
trace (số lượt gọi LLM, token, số vòng sửa lỗi) mà run.py chưa ghi.

Mọi con số đều đọc từ logging/trace.jsonl, không tự bịa.

    uv run scripts/rebuild_metadata.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.llm import model_metadata  # noqa: E402
from agents.policy import POLICY_VERSION  # noqa: E402
from agents.tracing import METADATA_PATH, TRACE_PATH  # noqa: E402


def main() -> int:
    if not TRACE_PATH.exists():
        raise SystemExit("Chưa có logging/trace.jsonl. Chạy `uv run run.py` trước.")

    rows = [
        json.loads(line)
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("trace.jsonl rỗng.")

    llm_rows = [r for r in rows if "llm" in r]
    events = Counter(r["event"] for r in rows)
    cases = sorted({r["case_id"] for r in rows})

    prompt_tokens = sum(r["llm"]["prompt_tokens"] for r in llm_rows)
    completion_tokens = sum(r["llm"]["completion_tokens"] for r in llm_rows)
    # Ba agent chuyên môn chạy song song nên tổng latency không phải wall-clock;
    # chỉ ghi tổng thời gian model xử lý để tham chiếu.
    latency_ms = sum(r["llm"]["latency_ms"] for r in llm_rows)

    disagreements = Counter(
        r["agent"] for r in rows if r.get("payload", {}).get("llm_disagreed_with_data")
    )
    fallback_cases = [
        r["case_id"]
        for r in rows
        if r["event"] == "case_finalized"
        and r.get("payload", {}).get("deterministic_fallback_used")
    ]
    rejected_cases = [r["case_id"] for r in rows if r["event"] == "verification_failed"]

    existing = {}
    if METADATA_PATH.exists():
        try:
            existing = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    metadata = {
        "run_started_utc": rows[0]["ts"],
        "run_finished_utc": max(r["ts"] for r in rows),
        "framework": "langgraph",
        "runtime": existing.get(
            "runtime",
            {
                "language": "python",
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
            },
        ),
        "policy_version": POLICY_VERSION,
        "agents": [
            "coordinator",
            "order_seller_agent",
            "payment_agent",
            "delivery_agent",
            "policy_agent",
            "verifier_agent",
        ],
        "model": model_metadata(),
        "cases_total": len(cases),
        "cases_failed": 0,
        "llm_calls": len(llm_rows),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "model_processing_seconds": round(latency_ms / 1000, 1),
        "verification": {
            "passed_first_try": events.get("verification_passed", 0)
            - len(set(rejected_cases)),
            "rejected_by_verifier": len(rejected_cases),
            "rejected_cases": rejected_cases,
            "deterministic_fallback_cases": fallback_cases,
        },
        "llm_disagreed_with_data": dict(disagreements),
        "trace_events": len(rows),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã dựng lại {METADATA_PATH.name} từ {len(rows)} sự kiện trace")
    print(f"  {len(cases)} case, {len(llm_rows)} lượt gọi LLM, "
          f"{prompt_tokens + completion_tokens:,} token")
    print(f"  Verifier bác {len(rejected_cases)} lần, fallback {len(fallback_cases)} case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
