"""Đối chiếu kết quả multi-agent với baseline áp luật tất định.

Trả lời hai câu hỏi mà báo cáo cần:
  1. Model 8B tự chọn đúng primary_issue được bao nhiêu case?
  2. Bao nhiêu case phải nhờ Verifier chặn lại và rơi về fallback tất định?

    uv run scripts/compare_runs.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT / "output"
BASELINE_DIR = ROOT / "output_baseline"
TRACE_PATH = ROOT / "logging" / "trace.jsonl"


def load_dir(path: Path) -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(path.glob("EC_*.json"))
    }


def main() -> int:
    if not AGENT_DIR.exists() or not BASELINE_DIR.exists():
        raise SystemExit("Cần chạy cả `uv run run.py` và `uv run run.py --dry-run --output-dir output_baseline`")

    agent = load_dir(AGENT_DIR)
    baseline = load_dir(BASELINE_DIR)

    # --- Case nào phải dùng fallback? Đọc từ trace. ---
    fallback: set[str] = set()
    repair_rounds: Counter[str] = Counter()
    disagreements: Counter[str] = Counter()
    if TRACE_PATH.exists():
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            payload = e.get("payload", {})
            if e["event"] == "case_finalized" and payload.get("deterministic_fallback_used"):
                fallback.add(e["case_id"])
            if e["event"] == "verification_failed":
                repair_rounds[e["case_id"]] += 1
            if payload.get("llm_disagreed_with_data"):
                disagreements[e["agent"]] += 1

    missing = sorted(set(baseline) - set(agent))
    mismatched: list[tuple[str, str, str]] = []
    for case_id, ref in baseline.items():
        got = agent.get(case_id)
        if got is None:
            continue
        a, b = got["assessment"]["primary_issue"], ref["assessment"]["primary_issue"]
        if a != b:
            mismatched.append((case_id, b, a))

    total = len(baseline)
    print(f"Tổng case            : {total}")
    print(f"Có file output       : {len(agent)}")
    if missing:
        print(f"THIẾU FILE           : {', '.join(missing)}")

    print(f"\nKhớp baseline        : {total - len(mismatched)}/{total}")
    if mismatched:
        print("Lệch:")
        for case_id, expected, actual in mismatched:
            print(f"  {case_id}: luật={expected}  agent={actual}")

    clean = total - len(fallback)
    print(f"\nPolicy Agent tự đúng : {clean}/{total} case (không cần fallback)")
    if fallback:
        print(f"Phải fallback        : {', '.join(sorted(fallback))}")
    if repair_rounds:
        print(f"Case có vòng sửa     : {len(repair_rounds)} "
              f"(tổng {sum(repair_rounds.values())} lần Verifier bác)")

    print("\nSố lần agent chuyên môn lệch với dữ liệu:")
    for agent_name in ("order_seller_agent", "payment_agent", "delivery_agent"):
        print(f"  {agent_name:<20}: {disagreements.get(agent_name, 0)}")

    # --- Kiểm tra bộ nộp ---
    print("\n=== Kiểm tra điều kiện nộp ===")
    ok = True
    expected_names = {f"EC_{i:03d}" for i in range(1, 51)}
    actual_names = set(agent)
    if actual_names != expected_names:
        ok = False
        if expected_names - actual_names:
            print(f"  THIẾU: {sorted(expected_names - actual_names)}")
        if actual_names - expected_names:
            print(f"  THỪA : {sorted(actual_names - expected_names)}")
    else:
        print("  Đủ đúng 50 file EC_001..EC_050")

    stray = [p.name for p in AGENT_DIR.iterdir() if p.name not in {f"{n}.json" for n in expected_names}]
    if stray:
        ok = False
        print(f"  FILE LẠ trong output/: {stray}")
    else:
        print("  Không có file lạ trong output/")

    print("\nKẾT LUẬN:", "sẵn sàng nộp" if ok and not missing else "CẦN XỬ LÝ TRƯỚC KHI NỘP")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
