"""Dựng lại output từ quyết định đã có trong trace, không gọi lại LLM.

Dùng khi chỉ sửa tầng tất định (cách chọn evidence, entity...) mà không đụng tới
phần suy luận. Kết luận `primary_issue` và `confidence` được lấy nguyên từ
`logging/trace.jsonl` của lượt chạy thật, nên kết quả giống hệt như chạy lại
toàn bộ pipeline — chỉ khác là mất vài giây thay vì 25 phút và không tốn quota.

    uv run scripts/rebuild_outputs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.dataset import get_dataset  # noqa: E402
from agents.policy import build_reference_output, first_matching_issue  # noqa: E402

TRACE = ROOT / "logging" / "trace.jsonl"
INPUT_DIR = ROOT / "input"


def decisions_from_trace() -> dict[str, tuple[str, float]]:
    """Lấy (primary_issue, confidence) đã chốt của từng case từ sự kiện case_finalized."""
    out: dict[str, tuple[str, float]] = {}
    if not TRACE.exists():
        return out
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") != "case_finalized":
            continue
        out[e["case_id"]] = (e["payload"]["primary_issue"], 0.0)

    # confidence nằm ở sự kiện của policy_agent, lấy lần cuối cùng của mỗi case
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "decision_ready" and e["case_id"] in out:
            issue, _ = out[e["case_id"]]
            out[e["case_id"]] = (issue, float(e["payload"]["confidence"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    ds = get_dataset()
    decisions = decisions_from_trace()
    if not decisions:
        raise SystemExit(
            "Không đọc được quyết định nào từ trace. Cần chạy `uv run run.py` trước."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    changed = 0

    for path in sorted(INPUT_DIR.glob("EC_*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        case_id = case["case_id"]
        order_id = case["customer_request"]["claimed_order_id"]
        bundle = ds.get_bundle(order_id)
        if bundle is None:
            print(f"  {case_id}: order không tồn tại, bỏ qua")
            continue

        issue, confidence = decisions.get(case_id, (None, 0.0))
        if issue is None:
            issue = first_matching_issue(bundle) or "unsupported_late_claim"
            confidence = 0.55
            print(f"  {case_id}: thiếu trong trace, áp luật tất định")

        output = build_reference_output(case_id, bundle, issue)
        output["assessment"]["confidence"] = round(confidence, 2)

        target = args.output_dir / f"{case_id}.json"
        new_text = json.dumps(output, ensure_ascii=False, indent=2)
        old_text = target.read_text(encoding="utf-8") if target.exists() else ""
        if new_text != old_text:
            changed += 1
        target.write_text(new_text, encoding="utf-8")

    print(f"\nĐã dựng lại {len(list(INPUT_DIR.glob('EC_*.json')))} file, {changed} file thay đổi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
