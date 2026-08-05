"""EC_POLICY_V1 — bộ quy tắc nghiệp vụ dạng tất định.

Module này KHÔNG gọi LLM. Nó tồn tại để:
  1. Cung cấp cho Policy Agent một bảng luật đã được đánh giá sẵn từng điều kiện.
  2. Cho Verifier Agent một kết quả tham chiếu để đối chiếu với kết luận của LLM.

Tách bạch như vậy vì thang chấm so khớp chính xác từng con số và từng ID:
một con số do LLM tự bịa ra là mất trọn điểm hạng mục đó.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dataset import OrderBundle

POLICY_VERSION = "EC_POLICY_V1"

PLATFORM_PARTY_ID = "OLIST_PLATFORM"
LOGISTICS_PARTY_ID = "LOGISTICS_PROVIDER"

# Giới hạn kích thước output theo README mục 6.
MAX_ENTITY_IDS = 5
MAX_EVIDENCE_IDS = 10
MAX_ROOT_CAUSES = 3
MAX_RESPONSIBLE_PARTIES = 3
MAX_ACTIONS = 5

# primary_issue -> (root_cause_code, action, case_status)
ISSUE_SPEC: dict[str, tuple[str, str, str]] = {
    "canceled_order_paid": (
        "ORDER_CANCELED_AFTER_PAYMENT",
        "issue_full_refund",
        "action_required",
    ),
    "unavailable_order_paid": (
        "ORDER_UNAVAILABLE_AFTER_PAYMENT",
        "issue_full_refund",
        "action_required",
    ),
    "late_delivery_seller": (
        "SELLER_HANDOFF_AFTER_LIMIT",
        "refund_freight",
        "action_required",
    ),
    "late_delivery_logistics": (
        "CARRIER_DELIVERED_AFTER_ESTIMATE",
        "refund_freight",
        "action_required",
    ),
    "valid_split_payment": (
        "MULTIPLE_PAYMENTS_RECONCILED",
        "explain_valid_split_payment",
        "no_action",
    ),
    "unsupported_late_claim": (
        "DELIVERY_WITHIN_ESTIMATE",
        "reject_late_refund",
        "no_action",
    ),
}

# Thứ tự ưu tiên áp dụng luật (README mục 4).
RULE_ORDER: list[str] = [
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
    "unsupported_late_claim",
]


@dataclass
class RuleEvaluation:
    """Kết quả kiểm tra một luật, kèm lý do người đọc hiểu được."""

    issue: str
    matched: bool
    reason: str


def evaluate_rules(bundle: OrderBundle) -> list[RuleEvaluation]:
    """Đánh giá cả 6 luật theo đúng thứ tự ưu tiên, không dừng sớm.

    Trả về đầy đủ để Policy Agent thấy được luật nào trượt và vì sao —
    đó là thứ agent cần để giải thích, thay vì chỉ nhận một nhãn.
    """
    status = bundle.order.order_status
    paid = bundle.payment_total_brl > 0
    late = bundle.delivered_late
    late_sellers = bundle.late_seller_ids
    n_payments = len(bundle.payments)

    out: list[RuleEvaluation] = []

    out.append(
        RuleEvaluation(
            "canceled_order_paid",
            status == "canceled" and paid,
            f"order_status={status!r}, payment_total={bundle.payment_total_brl}",
        )
    )
    out.append(
        RuleEvaluation(
            "unavailable_order_paid",
            status == "unavailable" and paid,
            f"order_status={status!r}, payment_total={bundle.payment_total_brl}",
        )
    )
    out.append(
        RuleEvaluation(
            "late_delivery_seller",
            late and bool(late_sellers),
            f"delivered_late={late}, sellers bàn giao sau shipping_limit={late_sellers or 'không có'}",
        )
    )
    out.append(
        RuleEvaluation(
            "late_delivery_logistics",
            late and not late_sellers,
            f"delivered_late={late}, seller bàn giao đúng hạn={not late_sellers}",
        )
    )
    out.append(
        RuleEvaluation(
            "valid_split_payment",
            n_payments >= 2 and bundle.payment_reconciles,
            f"payment_rows={n_payments}, delta={bundle.payment_delta_brl} BRL",
        )
    )
    out.append(
        RuleEvaluation(
            "unsupported_late_claim",
            (not late) and bundle.payment_reconciles,
            f"delivered_late={late}, payment khớp={bundle.payment_reconciles}",
        )
    )
    return out


def first_matching_issue(bundle: OrderBundle) -> str | None:
    by_issue = {e.issue: e for e in evaluate_rules(bundle)}
    for issue in RULE_ORDER:
        if by_issue[issue].matched:
            return issue
    return None


def responsible_parties_for(bundle: OrderBundle, issue: str) -> list[dict[str, str]]:
    if issue in ("canceled_order_paid", "unavailable_order_paid"):
        return [{"party_type": "platform", "party_id": PLATFORM_PARTY_ID}]
    if issue == "late_delivery_seller":
        return [
            {"party_type": "seller", "party_id": sid}
            for sid in bundle.late_seller_ids[:MAX_RESPONSIBLE_PARTIES]
        ]
    if issue == "late_delivery_logistics":
        return [{"party_type": "logistics_provider", "party_id": LOGISTICS_PARTY_ID}]
    return []


def refund_for(bundle: OrderBundle, issue: str) -> float:
    if issue in ("canceled_order_paid", "unavailable_order_paid"):
        return bundle.payment_total_brl
    if issue in ("late_delivery_seller", "late_delivery_logistics"):
        return bundle.freight_total_brl
    return 0.0


def evidence_ids_for(bundle: OrderBundle, issue: str) -> list[str]:
    """Chỉ dựng evidence ID từ hàng dữ liệu có thật, theo đúng 5 định dạng cho phép."""
    order_id = bundle.order.order_id
    root_cause = ISSUE_SPEC[issue][0]

    ids: list[str] = [f"order:{order_id}"]
    for item in bundle.items[:3]:
        ids.append(f"item:{order_id}:{item.order_item_id}")
    for payment in bundle.payments[:3]:
        ids.append(f"payment:{order_id}:{payment.payment_sequential}")

    # Chỉ nộp seller nào thực sự liên quan tới kết luận.
    relevant_sellers = (
        bundle.late_seller_ids if issue == "late_delivery_seller" else bundle.seller_ids
    )
    for seller_id in relevant_sellers[:2]:
        ids.append(f"seller:{seller_id}")

    ids.append(f"policy:{root_cause}")

    # policy: là bằng chứng neo kết luận — phải giữ lại nếu phải cắt bớt.
    if len(ids) > MAX_EVIDENCE_IDS:
        ids = ids[: MAX_EVIDENCE_IDS - 1] + [f"policy:{root_cause}"]
    return ids


def build_reference_output(case_id: str, bundle: OrderBundle, issue: str) -> dict[str, Any]:
    """Output tham chiếu tất định cho một (case, issue) — Verifier so kết quả LLM với cái này."""
    root_cause, action, case_status = ISSUE_SPEC[issue]
    order_id = bundle.order.order_id

    return {
        "case_id": case_id,
        "assessment": {
            "primary_issue": issue,
            "case_status": case_status,
            "confidence": 0.0,  # do Policy Agent điền, Verifier chỉ kiểm khoảng [0,1]
        },
        "affected_entities": {
            "order_ids": [order_id],
            "item_ids": [
                f"{order_id}:{i.order_item_id}" for i in bundle.items[:MAX_ENTITY_IDS]
            ],
            "seller_ids": bundle.seller_ids[:MAX_ENTITY_IDS],
            "payment_ids": [
                f"{order_id}:{p.payment_sequential}"
                for p in bundle.payments[:MAX_ENTITY_IDS]
            ],
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": root_cause, "rank": 1}],
            "responsible_parties": responsible_parties_for(bundle, issue),
        },
        "evidence_ids": evidence_ids_for(bundle, issue),
        "financial_resolution": {
            "currency": "BRL",
            "item_total_brl": bundle.item_total_brl,
            "freight_total_brl": bundle.freight_total_brl,
            "payment_total_brl": bundle.payment_total_brl,
            "recommended_refund_brl": refund_for(bundle, issue),
        },
        "resolution_actions": [action],
    }
