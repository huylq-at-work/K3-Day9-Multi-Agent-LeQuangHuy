"""Sinh case mẫu để kiểm thử pipeline khi chưa có input chính thức.

CẢNH BÁO: đây KHÔNG phải 50 case đề bài (được công bố ở Checkpoint 1).
Script chỉ quét CSV tìm order thật thuộc mỗi nhánh luật, để chạy thử end-to-end.
Ghi vào input_sample/ chứ không đụng vào input/.

    uv run scripts/make_sample_inputs.py --per-issue 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.dataset import get_dataset  # noqa: E402
from agents.policy import RULE_ORDER, first_matching_issue  # noqa: E402

SAMPLE_DIR = ROOT / "input_sample"

MESSAGE_BY_ISSUE = {
    "canceled_order_paid": "Đơn của tôi bị hủy nhưng tôi đã thanh toán. Tôi muốn được hoàn tiền.",
    "unavailable_order_paid": "Tôi đã trả tiền nhưng đơn báo không khả dụng. Xin xử lý giúp tôi.",
    "late_delivery_seller": "Đơn hàng của tôi có dấu hiệu giao trễ. Hãy kiểm tra nguyên nhân và quyền lợi phù hợp.",
    "late_delivery_logistics": "Hàng đến muộn hơn ngày dự kiến rất nhiều. Ai chịu trách nhiệm việc này?",
    "valid_split_payment": "Tôi thấy có nhiều giao dịch trừ tiền cho cùng một đơn. Tôi có bị thu thừa không?",
    "unsupported_late_claim": "Tôi nghĩ đơn của tôi bị giao trễ. Đề nghị kiểm tra và hoàn phí ship.",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-issue", type=int, default=2)
    parser.add_argument("--scan-limit", type=int, default=40000)
    args = parser.parse_args()

    dataset = get_dataset()
    order_ids = list(dataset._orders.keys())[: args.scan_limit]  # noqa: SLF001 - script nội bộ

    buckets: dict[str, list[str]] = {issue: [] for issue in RULE_ORDER}
    for order_id in order_ids:
        if all(len(v) >= args.per_issue for v in buckets.values()):
            break
        bundle = dataset.get_bundle(order_id)
        if bundle is None:
            continue
        issue = first_matching_issue(bundle)
        if issue and len(buckets[issue]) < args.per_issue:
            buckets[issue].append(order_id)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for old in SAMPLE_DIR.glob("EC_*.json"):
        old.unlink()

    index = 0
    for issue in RULE_ORDER:
        for order_id in buckets[issue]:
            index += 1
            case_id = f"EC_{index:03d}"
            bundle = dataset.get_bundle(order_id)
            assert bundle is not None
            opened = bundle.order.estimated_delivery_ts or bundle.order.purchase_ts
            (SAMPLE_DIR / f"{case_id}.json").write_text(
                json.dumps(
                    {
                        "case_id": case_id,
                        "opened_at": (opened.isoformat() if opened else "") + "-03:00",
                        "customer_request": {
                            "language": "vi",
                            "message": MESSAGE_BY_ISSUE[issue],
                            "claimed_order_id": order_id,
                        },
                        "policy_version": "EC_POLICY_V1",
                        "_expected_issue_chỉ_để_kiểm_thử": issue,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{case_id}  {issue:<26} {order_id}")

    missing = [i for i, v in buckets.items() if not v]
    if missing:
        print(f"\nKhông tìm thấy order mẫu cho: {', '.join(missing)}")
    print(f"\nĐã ghi {index} case vào {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
