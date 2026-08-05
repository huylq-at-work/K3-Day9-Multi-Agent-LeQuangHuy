"""Ghi trace chạy thật ra logging/trace.jsonl.

Đề yêu cầu trace của lượt chạy mới nhất, không append — nên file được ghi đè
mỗi lần chạy đủ bộ case.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logging"
TRACE_PATH = LOG_DIR / "trace.jsonl"
METADATA_PATH = LOG_DIR / "metadata.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def trace_event(
    case_id: str,
    agent: str,
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    handoff_to: str | None = None,
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dựng một bản ghi trace. Trả về dict để node đưa thẳng vào state['trace']."""
    entry: dict[str, Any] = {
        "ts": _now(),
        "case_id": case_id,
        "agent": agent,
        "event": event,
    }
    if handoff_to:
        entry["handoff_to"] = handoff_to
    if llm:
        entry["llm"] = llm
    if payload:
        entry["payload"] = payload
    return entry


def write_trace(entries: list[dict[str, Any]]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return TRACE_PATH


def write_metadata(extra: dict[str, Any]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return METADATA_PATH
