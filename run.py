"""Chạy toàn bộ case trong input/ qua graph multi-agent và ghi kết quả ra output/.

Cách dùng:
    uv run run.py                      # chạy tất cả case
    uv run run.py --cases EC_001       # chạy một case để soi trace
    uv run run.py --workers 4          # số case chạy song song
    uv run run.py --dry-run            # bỏ qua LLM, chỉ áp luật tất định (dùng để đối chiếu)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.dataset import get_dataset
from agents.graph import build_graph
from agents.llm import model_metadata
from agents.nodes import coordinator_abort
from agents.policy import POLICY_VERSION, build_reference_output, first_matching_issue
from agents.tracing import write_metadata, write_trace

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def load_cases(only: list[str] | None, input_dir: Path = INPUT_DIR) -> list[dict[str, Any]]:
    if not input_dir.exists():
        raise SystemExit(f"Không tìm thấy thư mục input: {input_dir}")
    cases = []
    for path in sorted(input_dir.glob("EC_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if only and data.get("case_id") not in only:
            continue
        cases.append(data)
    if not cases:
        raise SystemExit(
            f"Không có case nào trong {input_dir}. "
            "Input đề bài được công bố ở Checkpoint 1 — hãy chép 50 file EC_*.json vào đây."
        )
    return cases


def initial_state(case: dict[str, Any]) -> dict[str, Any]:
    request = case.get("customer_request", {})
    return {
        "case_id": case["case_id"],
        "opened_at": case.get("opened_at", ""),
        "customer_message": request.get("message", ""),
        "claimed_order_id": request.get("claimed_order_id", ""),
        "policy_version": case.get("policy_version", POLICY_VERSION),
        "repair_count": 0,
        "trace": [],
    }


def run_deterministic(case: dict[str, Any]) -> dict[str, Any]:
    """Đường tắt không dùng LLM — dùng để đối chiếu và để chạy khi chưa có API key."""
    state = initial_state(case)
    bundle = get_dataset().get_bundle(state["claimed_order_id"])
    if bundle is None:
        return coordinator_abort(state)["final_output"]
    issue = first_matching_issue(bundle) or "unsupported_late_claim"
    output = build_reference_output(case["case_id"], bundle, issue)
    # Cùng quy ước với đường chạy đầy đủ: kết luận đã được luật tất định xác nhận.
    output["assessment"]["confidence"] = 1.0
    return output


def run_case(app, case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final = app.invoke(initial_state(case))
    return final["final_output"], final.get("trace", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", help="Chỉ chạy các case_id này")
    parser.add_argument("--workers", type=int, default=4, help="Số case chạy song song")
    parser.add_argument("--dry-run", action="store_true", help="Bỏ qua LLM, chỉ áp luật")
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT_DIR, help="Thư mục case (mặc định input/)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Thư mục ghi kết quả"
    )
    args = parser.parse_args()

    cases = load_cases(args.cases, args.input_dir)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    get_dataset()  # nạp CSV một lần trước khi fan-out

    started = time.perf_counter()
    all_traces: list[dict[str, Any]] = []
    failures: list[str] = []

    if args.dry_run:
        for case in cases:
            output = run_deterministic(case)
            (output_dir / f"{case['case_id']}.json").write_text(
                json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[dry-run] {case['case_id']} -> {output['assessment']['primary_issue']}")
    else:
        app = build_graph()

        def work(case: dict[str, Any]):
            try:
                return case["case_id"], *run_case(app, case), None
            except Exception as exc:  # noqa: BLE001
                return case["case_id"], None, [], exc

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for case_id, output, traces, error in pool.map(work, cases):
                all_traces.extend(traces)
                if error is not None or output is None:
                    failures.append(f"{case_id}: {error}")
                    print(f"[LỖI] {case_id}: {error}", file=sys.stderr)
                    continue
                (output_dir / f"{case_id}.json").write_text(
                    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[ok] {case_id} -> {output['assessment']['primary_issue']} "
                      f"(refund {output['financial_resolution']['recommended_refund_brl']} BRL)")

        all_traces.sort(key=lambda e: (e["case_id"], e["ts"]))
        write_trace(all_traces)

    elapsed = time.perf_counter() - started

    if args.dry_run:
        # Không ghi metadata: file này phải mô tả lượt chạy thật qua đủ 6 agent.
        # Trước đây dry-run ghi đè làm metadata báo elapsed 0.02s và dry_run=true.
        print(f"\n[dry-run] Xong {len(cases)} case trong {elapsed:.1f}s — giữ nguyên metadata.json")
        return 0

    write_metadata(
        {
            "run_started_utc": datetime.now(timezone.utc).isoformat(),
            "framework": "langgraph",
            "runtime": {
                "language": "python",
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
            },
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
            "cases_failed": len(failures),
            "elapsed_seconds": round(elapsed, 2),
            "dry_run": args.dry_run,
        }
    )

    print(f"\nXong {len(cases) - len(failures)}/{len(cases)} case trong {elapsed:.1f}s")
    if failures:
        print(f"Thất bại: {len(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
