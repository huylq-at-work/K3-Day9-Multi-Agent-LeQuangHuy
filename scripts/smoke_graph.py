"""Kiểm tra wiring của graph mà không tiêu tốn quota LLM.

Thay call_json bằng một stub trả lời đúng schema từng agent, rồi chạy bộ case mẫu
và đối chiếu primary_issue với nhãn kỳ vọng.

    uv run scripts/smoke_graph.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import llm, nodes  # noqa: E402


class StubResult:
    def __init__(self, content):
        self.content = content
        self.raw_text = json.dumps(content)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms = 0


def make_stub(policy_answer: str | None = None):
    """Stub đoán agent nào đang gọi dựa vào system prompt, rồi trả JSON hợp lệ."""

    def stub(system_prompt: str, user_prompt: str, **_kw):
        data = json.loads(user_prompt)
        if "Order & Seller Agent" in system_prompt:
            late = data.get("sellers_bàn_giao_sau_shipping_limit_do_hệ_thống_tính", [])
            return StubResult(
                {
                    "order_status_class": data.get("order_status", "other"),
                    "seller_handoff_late": bool(late),
                    "late_seller_ids": late,
                    "notes": "stub",
                }
            )
        if "Payment Agent" in system_prompt:
            return StubResult(
                {
                    "reconciled": data.get("khớp_trong_sai_số_0.10", False),
                    "is_split_payment": data.get("payment_row_count", 0) >= 2
                    and data.get("khớp_trong_sai_số_0.10", False),
                    "notes": "stub",
                }
            )
        if "Delivery Agent" in system_prompt:
            late = data.get("giao_trễ_do_hệ_thống_tính", False)
            sellers = data.get("sellers_bàn_giao_muộn_do_hệ_thống_tính", [])
            attribution = ("seller" if sellers else "logistics_provider") if late else "none"
            return StubResult(
                {"delivered_late": late, "attribution": attribution, "notes": "stub"}
            )
        if "Policy Agent" in system_prompt:
            if policy_answer is not None:
                return StubResult(
                    {"primary_issue": policy_answer, "confidence": 0.9, "rationale": "stub"}
                )
            matched = [
                e["issue"]
                for e in data["kết_quả_kiểm_tra_từng_luật"]
                if e["matched"]
            ]
            return StubResult(
                {
                    "primary_issue": matched[0] if matched else "unsupported_late_claim",
                    "confidence": 0.93,
                    "rationale": "stub",
                }
            )
        raise AssertionError(f"System prompt lạ: {system_prompt[:60]}")

    return stub


def run_suite(stub, cases) -> list[tuple[str, str, str, dict]]:
    llm.call_json = stub
    nodes.call_json = stub
    from agents.graph import build_graph
    from run import initial_state

    app = build_graph()
    rows = []
    for case in cases:
        final = app.invoke(initial_state(case))
        out = final["final_output"]
        rows.append(
            (
                case["case_id"],
                case["_expected_issue_chỉ_để_kiểm_thử"],
                out["assessment"]["primary_issue"],
                final,
            )
        )
    return rows


def main() -> int:
    sample_dir = ROOT / "input_sample"
    cases = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(sample_dir.glob("EC_*.json"))
    ]
    if not cases:
        raise SystemExit("Chưa có case mẫu. Chạy: uv run scripts/make_sample_inputs.py")

    print("=== A. Agent đồng thuận với dữ liệu ===")
    failures = 0
    agent_names = set()
    for case_id, expected, actual, final in run_suite(make_stub(), cases):
        ok = expected == actual
        failures += not ok
        agent_names.update(e["agent"] for e in final["trace"])
        print(f"{'PASS' if ok else 'FAIL'} {case_id}  kỳ vọng={expected:<24} thực tế={actual}")

    print("\n=== B. Policy Agent kết luận SAI -> Verifier phải chặn ===")
    rows = run_suite(make_stub(policy_answer="canceled_order_paid"), cases[-2:])
    for case_id, expected, actual, final in rows:
        report = final["verifier_report"]
        repaired = actual == expected
        events = [e["event"] for e in final["trace"] if e["agent"] == "verifier_agent"]
        print(
            f"{'PASS' if repaired else 'FAIL'} {case_id}  "
            f"LLM nói 'canceled_order_paid' -> output cuối={actual} "
            f"(verifier: {events}, vòng sửa={final.get('repair_count')})"
        )
        failures += not repaired
        if report["passed"]:
            print("   FAIL: verifier lẽ ra phải bác kết luận sai")
            failures += 1

    print(f"\nAgent đã tham gia trace: {sorted(agent_names)}")
    print("KẾT QUẢ:", "TẤT CẢ ĐỀU PASS" if failures == 0 else f"{failures} lỗi")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
