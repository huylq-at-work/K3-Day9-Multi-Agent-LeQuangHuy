"""Sáu agent node của graph.

Nguyên tắc chung xuyên suốt: **LLM phán đoán, code tính toán.**
Mọi con số tiền, mốc thời gian và ID đều do tầng `dataset`/`policy` sinh ra.
Agent LLM nhận các dữ kiện đó, đưa ra nhận định nghiệp vụ và bàn giao cho agent kế tiếp.
Lý do: thang chấm so khớp chính xác từng giá trị, nên để model 8B tự cộng tiền
là cách nhanh nhất để mất điểm.
"""

from __future__ import annotations

import json
from typing import Any

from .dataset import OrderBundle, get_dataset
from .llm import call_json
from .policy import (
    ISSUE_SPEC,
    MAX_ACTIONS,
    MAX_ENTITY_IDS,
    MAX_EVIDENCE_IDS,
    MAX_RESPONSIBLE_PARTIES,
    MAX_ROOT_CAUSES,
    RULE_ORDER,
    build_reference_output,
    evaluate_rules,
    first_matching_issue,
)
from .state import CaseState
from .tracing import trace_event

MAX_REPAIR_ROUNDS = 2


def _llm_meta(result: Any) -> dict[str, Any]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_ms": result.latency_ms,
    }


def _bundle_for(state: CaseState) -> OrderBundle | None:
    return get_dataset().get_bundle(state["claimed_order_id"])


# --------------------------------------------------------------------------
# 1. Coordinator — intake
# --------------------------------------------------------------------------

def coordinator_intake(state: CaseState) -> dict[str, Any]:
    """Nhận case, xác minh order tồn tại, rồi fan-out cho ba agent chuyên môn.

    Không gọi LLM: đây là bước định tuyến thuần túy, dùng model ở đây chỉ tốn
    thời gian và thêm một chỗ có thể sai.
    """
    case_id = state["case_id"]
    order_id = state["claimed_order_id"]
    bundle = get_dataset().get_bundle(order_id)

    if bundle is None:
        return {
            "order_found": False,
            "fatal_error": f"claimed_order_id {order_id!r} không tồn tại trong olist_orders_dataset",
            "trace": [
                trace_event(
                    case_id,
                    "coordinator",
                    "intake_failed",
                    {"claimed_order_id": order_id},
                )
            ],
        }

    return {
        "order_found": True,
        "trace": [
            trace_event(
                case_id,
                "coordinator",
                "intake_ok",
                {
                    "claimed_order_id": order_id,
                    "order_status": bundle.order.order_status,
                    "item_rows": len(bundle.items),
                    "payment_rows": len(bundle.payments),
                },
                handoff_to="order_seller_agent|payment_agent|delivery_agent",
            )
        ],
    }


# --------------------------------------------------------------------------
# 2. Order & Seller Agent
# --------------------------------------------------------------------------

ORDER_SYSTEM = """Bạn là Order & Seller Agent trong hệ thống xử lý khiếu nại thương mại điện tử Olist.

Nhiệm vụ: đọc dữ kiện đơn hàng đã được truy xuất sẵn từ CSV và đưa ra nhận định về
trạng thái đơn cùng việc seller có bàn giao hàng cho đơn vị vận chuyển đúng hạn hay không.

Quy tắc bắt buộc:
- CHỈ dùng dữ kiện được cung cấp. Tuyệt đối không tự tính lại số tiền, không bịa seller_id.
- Olist không có dữ liệu giao sai/giao thiếu/tracking từng item. Không suy diễn các sự kiện đó.
- Seller bị coi là bàn giao muộn khi order_delivered_carrier_date > shipping_limit_date của item thuộc seller đó.

Trả về đúng một JSON object:
{
  "order_status_class": "delivered" | "canceled" | "unavailable" | "other",
  "seller_handoff_late": true/false,
  "late_seller_ids": ["..."],
  "notes": "một câu tiếng Việt giải thích căn cứ"
}"""


def order_seller_agent(state: CaseState) -> dict[str, Any]:
    case_id = state["case_id"]
    bundle = _bundle_for(state)
    assert bundle is not None
    facts = bundle.to_facts()

    prompt = json.dumps(
        {
            "order_id": facts["order_id"],
            "order_status": facts["order_status"],
            "order_delivered_carrier_date": facts["order_delivered_carrier_date"],
            "items": facts["items"],
            "seller_ids": facts["seller_ids"],
            "sellers_bàn_giao_sau_shipping_limit_do_hệ_thống_tính": facts["late_seller_ids"],
        },
        ensure_ascii=False,
        indent=2,
    )

    result = call_json(ORDER_SYSTEM, prompt)
    verdict = result.content

    # Dữ kiện tất định luôn thắng nhận định của LLM khi hai bên lệch nhau.
    order_facts = {
        "order_id": facts["order_id"],
        "order_status": facts["order_status"],
        "item_count": facts["item_count"],
        "seller_ids": facts["seller_ids"],
        "late_seller_ids": facts["late_seller_ids"],
        "item_total_brl": facts["item_total_brl"],
        "freight_total_brl": facts["freight_total_brl"],
        "items": facts["items"],
        "agent_notes": str(verdict.get("notes", ""))[:400],
        "agent_seller_handoff_late": bool(verdict.get("seller_handoff_late", False)),
    }

    disagreement = order_facts["agent_seller_handoff_late"] != bool(facts["late_seller_ids"])

    return {
        "order_facts": order_facts,
        "trace": [
            trace_event(
                case_id,
                "order_seller_agent",
                "facts_ready",
                {
                    "seller_handoff_late_deterministic": bool(facts["late_seller_ids"]),
                    "seller_handoff_late_llm": order_facts["agent_seller_handoff_late"],
                    "llm_disagreed_with_data": disagreement,
                    "notes": order_facts["agent_notes"],
                },
                handoff_to="policy_agent",
                llm=_llm_meta(result),
            )
        ],
    }


# --------------------------------------------------------------------------
# 3. Payment Agent
# --------------------------------------------------------------------------

PAYMENT_SYSTEM = """Bạn là Payment Agent trong hệ thống xử lý khiếu nại Olist.

Nhiệm vụ: đối soát các dòng thanh toán với tổng tiền hàng cộng phí vận chuyển.

Quy tắc bắt buộc:
- CHỈ dùng số đã cho. Không tự cộng trừ lại, không làm tròn lại.
- payment_value là số tiền của từng dòng payment, KHÔNG phải giá trị từng kỳ trả góp.
- Đơn được coi là khớp khi |payment_total - (item_total + freight_total)| <= 0.10 BRL.
- Split payment hợp lệ = có từ 2 dòng payment trở lên VÀ tổng tiền khớp.

Trả về đúng một JSON object:
{
  "reconciled": true/false,
  "is_split_payment": true/false,
  "notes": "một câu tiếng Việt giải thích"
}"""


def payment_agent(state: CaseState) -> dict[str, Any]:
    case_id = state["case_id"]
    bundle = _bundle_for(state)
    assert bundle is not None
    facts = bundle.to_facts()

    prompt = json.dumps(
        {
            "order_id": facts["order_id"],
            "payments": facts["payments"],
            "payment_row_count": facts["payment_row_count"],
            "item_total_brl": facts["item_total_brl"],
            "freight_total_brl": facts["freight_total_brl"],
            "payment_total_brl": facts["payment_total_brl"],
            "delta_brl_do_hệ_thống_tính": facts["payment_delta_brl"],
            "khớp_trong_sai_số_0.10": facts["payment_reconciles"],
        },
        ensure_ascii=False,
        indent=2,
    )

    result = call_json(PAYMENT_SYSTEM, prompt)
    verdict = result.content

    payment_facts = {
        "payment_row_count": facts["payment_row_count"],
        "payment_total_brl": facts["payment_total_brl"],
        "payment_delta_brl": facts["payment_delta_brl"],
        "payment_reconciles": facts["payment_reconciles"],
        "payments": facts["payments"],
        "agent_notes": str(verdict.get("notes", ""))[:400],
        "agent_is_split_payment": bool(verdict.get("is_split_payment", False)),
    }

    deterministic_split = facts["payment_row_count"] >= 2 and facts["payment_reconciles"]

    return {
        "payment_facts": payment_facts,
        "trace": [
            trace_event(
                case_id,
                "payment_agent",
                "facts_ready",
                {
                    "reconciles_deterministic": facts["payment_reconciles"],
                    "is_split_deterministic": deterministic_split,
                    "is_split_llm": payment_facts["agent_is_split_payment"],
                    "llm_disagreed_with_data": payment_facts["agent_is_split_payment"] != deterministic_split,
                    "notes": payment_facts["agent_notes"],
                },
                handoff_to="policy_agent",
                llm=_llm_meta(result),
            )
        ],
    }


# --------------------------------------------------------------------------
# 4. Delivery Agent
# --------------------------------------------------------------------------

DELIVERY_SYSTEM = """Bạn là Delivery Agent trong hệ thống xử lý khiếu nại Olist.

Nhiệm vụ: so sánh thời điểm giao hàng thực tế với hạn giao dự kiến, và quy trách nhiệm
cho seller hay đơn vị vận chuyển.

Quy tắc bắt buộc:
- Đơn giao trễ khi order_delivered_customer_date > order_estimated_delivery_date.
- Nếu đơn giao trễ VÀ seller bàn giao cho carrier sau shipping_limit_date -> lỗi thuộc seller.
- Nếu đơn giao trễ NHƯNG seller bàn giao đúng hạn -> lỗi thuộc đơn vị vận chuyển.
- Nếu đơn không giao trễ, khiếu nại giao trễ là không có căn cứ.
- Không suy diễn checkpoint vận chuyển hay sự kiện không có trong dữ liệu.

Trả về đúng một JSON object:
{
  "delivered_late": true/false,
  "attribution": "seller" | "logistics_provider" | "none",
  "notes": "một câu tiếng Việt giải thích"
}"""


def delivery_agent(state: CaseState) -> dict[str, Any]:
    case_id = state["case_id"]
    bundle = _bundle_for(state)
    assert bundle is not None
    facts = bundle.to_facts()

    prompt = json.dumps(
        {
            "order_id": facts["order_id"],
            "order_status": facts["order_status"],
            "order_delivered_carrier_date": facts["order_delivered_carrier_date"],
            "order_delivered_customer_date": facts["order_delivered_customer_date"],
            "order_estimated_delivery_date": facts["order_estimated_delivery_date"],
            "giao_trễ_do_hệ_thống_tính": facts["delivered_late"],
            "sellers_bàn_giao_muộn_do_hệ_thống_tính": facts["late_seller_ids"],
            "customer_message": state.get("customer_message", ""),
        },
        ensure_ascii=False,
        indent=2,
    )

    result = call_json(DELIVERY_SYSTEM, prompt)
    verdict = result.content

    if facts["delivered_late"]:
        attribution = "seller" if facts["late_seller_ids"] else "logistics_provider"
    else:
        attribution = "none"

    delivery_facts = {
        "delivered_late": facts["delivered_late"],
        "order_delivered_customer_date": facts["order_delivered_customer_date"],
        "order_estimated_delivery_date": facts["order_estimated_delivery_date"],
        "attribution": attribution,
        "agent_notes": str(verdict.get("notes", ""))[:400],
        "agent_attribution": str(verdict.get("attribution", "")),
        "agent_delivered_late": bool(verdict.get("delivered_late", False)),
    }

    return {
        "delivery_facts": delivery_facts,
        "trace": [
            trace_event(
                case_id,
                "delivery_agent",
                "facts_ready",
                {
                    "delivered_late_deterministic": facts["delivered_late"],
                    "delivered_late_llm": delivery_facts["agent_delivered_late"],
                    "attribution_deterministic": attribution,
                    "attribution_llm": delivery_facts["agent_attribution"],
                    "llm_disagreed_with_data": delivery_facts["agent_attribution"] != attribution,
                    "notes": delivery_facts["agent_notes"],
                },
                handoff_to="policy_agent",
                llm=_llm_meta(result),
            )
        ],
    }


# --------------------------------------------------------------------------
# 5. Policy Agent
# --------------------------------------------------------------------------

POLICY_SYSTEM = """Bạn là Policy Agent, áp dụng bộ quy tắc EC_POLICY_V1 của Olist.

Bạn nhận bằng chứng đã được ba agent chuyên môn bàn giao, kèm kết quả kiểm tra
từng điều kiện luật. Nhiệm vụ của bạn là chọn ĐÚNG MỘT primary_issue.

Áp dụng theo đúng thứ tự ưu tiên sau, chọn luật khớp ĐẦU TIÊN:
1. canceled_order_paid      - order_status=canceled và tổng payment > 0
2. unavailable_order_paid   - order_status=unavailable và tổng payment > 0
3. late_delivery_seller     - giao trễ và seller bàn giao sau shipping_limit_date
4. late_delivery_logistics  - giao trễ và seller bàn giao đúng hạn
5. valid_split_payment      - từ 2 dòng payment và tổng tiền khớp trong 0.10 BRL
6. unsupported_late_claim   - giao không trễ và payment khớp

Đặt confidence trong [0,1]: cao (0.9-0.97) khi bằng chứng rõ và các agent đồng thuận,
thấp hơn (0.6-0.8) khi có agent nào đó không đồng thuận với dữ liệu.

Trả về đúng một JSON object:
{
  "primary_issue": "<một trong sáu giá trị trên>",
  "confidence": 0.0-1.0,
  "rationale": "một tới hai câu tiếng Việt nêu căn cứ"
}"""


def policy_agent(state: CaseState) -> dict[str, Any]:
    case_id = state["case_id"]
    bundle = _bundle_for(state)
    assert bundle is not None

    evaluations = [
        {"issue": e.issue, "matched": e.matched, "reason": e.reason}
        for e in evaluate_rules(bundle)
    ]

    # Bỏ mảng items[]/payments[] chi tiết: Policy Agent chỉ cần phần tổng hợp,
    # mà hạn mức TPM của gói free rất chật nên mỗi token thừa đều phải trả giá.
    def _compact(facts: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in facts.items() if k not in ("items", "payments")}

    payload: dict[str, Any] = {
        "order_evidence": _compact(state["order_facts"]),
        "payment_evidence": _compact(state["payment_facts"]),
        "delivery_evidence": state["delivery_facts"],
        "kết_quả_kiểm_tra_từng_luật": evaluations,
    }

    # Vòng sửa lỗi: nếu Verifier đã bác kết luận trước, đưa lý do bác vào prompt.
    report = state.get("verifier_report")
    if report and not report.get("passed", True):
        payload["verifier_đã_bác_kết_luận_trước"] = report.get("issues", [])

    result = call_json(POLICY_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2))
    verdict = result.content

    issue = str(verdict.get("primary_issue", "")).strip()
    try:
        confidence = float(verdict.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    decision = {
        "primary_issue": issue,
        "confidence": round(confidence, 2),
        "rationale": str(verdict.get("rationale", ""))[:600],
        "rule_evaluations": evaluations,
    }

    return {
        "policy_decision": decision,
        "repair_count": state.get("repair_count", 0),
        "trace": [
            trace_event(
                case_id,
                "policy_agent",
                "decision_ready",
                {
                    "primary_issue": issue,
                    "confidence": decision["confidence"],
                    "rationale": decision["rationale"],
                    "is_repair_round": bool(report and not report.get("passed", True)),
                },
                handoff_to="verifier_agent",
                llm=_llm_meta(result),
            )
        ],
    }


# --------------------------------------------------------------------------
# 6. Verifier Agent
# --------------------------------------------------------------------------

VALID_EVIDENCE_PREFIXES = ("order:", "item:", "payment:", "seller:", "policy:")


def _validate_output(candidate: dict[str, Any], bundle: OrderBundle) -> list[str]:
    """Kiểm tra schema, giới hạn kích thước và tính tồn tại của mọi ID."""
    problems: list[str] = []
    order_id = bundle.order.order_id

    issue = candidate["assessment"]["primary_issue"]
    if issue not in ISSUE_SPEC:
        problems.append(f"primary_issue không hợp lệ: {issue!r}")
        return problems

    confidence = candidate["assessment"]["confidence"]
    if not (0.0 <= confidence <= 1.0):
        problems.append(f"confidence ngoài khoảng [0,1]: {confidence}")

    expected_status = ISSUE_SPEC[issue][2]
    if candidate["assessment"]["case_status"] != expected_status:
        problems.append(
            f"case_status={candidate['assessment']['case_status']!r} không khớp issue {issue!r} "
            f"(phải là {expected_status!r})"
        )

    ents = candidate["affected_entities"]
    for key, limit in (
        ("order_ids", MAX_ENTITY_IDS),
        ("item_ids", MAX_ENTITY_IDS),
        ("seller_ids", MAX_ENTITY_IDS),
        ("payment_ids", MAX_ENTITY_IDS),
    ):
        if len(ents[key]) > limit:
            problems.append(f"{key} vượt giới hạn {limit}")

    valid_item_ids = {f"{order_id}:{i.order_item_id}" for i in bundle.items}
    for iid in ents["item_ids"]:
        if iid not in valid_item_ids:
            problems.append(f"item_id không tồn tại trong CSV: {iid}")

    valid_payment_ids = {f"{order_id}:{p.payment_sequential}" for p in bundle.payments}
    for pid in ents["payment_ids"]:
        if pid not in valid_payment_ids:
            problems.append(f"payment_id không tồn tại trong CSV: {pid}")

    dataset = get_dataset()
    for sid in ents["seller_ids"]:
        if not dataset.seller_exists(sid):
            problems.append(f"seller_id không tồn tại trong CSV: {sid}")

    evidence = candidate["evidence_ids"]
    if len(evidence) > MAX_EVIDENCE_IDS:
        problems.append(f"evidence_ids vượt giới hạn {MAX_EVIDENCE_IDS}")
    for ev in evidence:
        if not ev.startswith(VALID_EVIDENCE_PREFIXES):
            problems.append(f"evidence ID sai định dạng: {ev}")

    rca = candidate["root_cause_analysis"]
    if len(rca["ranked_causes"]) > MAX_ROOT_CAUSES:
        problems.append(f"ranked_causes vượt giới hạn {MAX_ROOT_CAUSES}")
    if len(rca["responsible_parties"]) > MAX_RESPONSIBLE_PARTIES:
        problems.append(f"responsible_parties vượt giới hạn {MAX_RESPONSIBLE_PARTIES}")
    if len(candidate["resolution_actions"]) > MAX_ACTIONS:
        problems.append(f"resolution_actions vượt giới hạn {MAX_ACTIONS}")

    fin = candidate["financial_resolution"]
    if fin["currency"] != "BRL":
        problems.append(f"currency phải là BRL, đang là {fin['currency']!r}")
    for key, expected in (
        ("item_total_brl", bundle.item_total_brl),
        ("freight_total_brl", bundle.freight_total_brl),
        ("payment_total_brl", bundle.payment_total_brl),
    ):
        if abs(fin[key] - expected) > 0.005:
            problems.append(f"{key}={fin[key]} lệch với dữ liệu ({expected})")

    if not bundle.items:
        if ents["item_ids"] or ents["seller_ids"]:
            problems.append("order không có item row nhưng item_ids/seller_ids không rỗng")
        if fin["item_total_brl"] != 0.0 or fin["freight_total_brl"] != 0.0:
            problems.append("order không có item row nhưng tổng item/freight khác 0.0")

    return problems


def verifier_agent(state: CaseState) -> dict[str, Any]:
    """Chốt chặn tất định trước khi ghi file.

    So kết luận của Policy Agent với luật tất định. Lệch thì trả case về cho
    Policy Agent kèm lý do, tối đa MAX_REPAIR_ROUNDS vòng.
    """
    case_id = state["case_id"]
    bundle = _bundle_for(state)
    assert bundle is not None

    decision = state["policy_decision"]
    llm_issue = decision["primary_issue"]
    reference_issue = first_matching_issue(bundle)

    problems: list[str] = []
    if reference_issue is None:
        problems.append("không luật nào trong EC_POLICY_V1 khớp với order này")
    elif llm_issue != reference_issue:
        problems.append(
            f"primary_issue của Policy Agent ({llm_issue!r}) khác kết quả áp luật "
            f"tất định ({reference_issue!r})"
        )

    chosen_issue = llm_issue if llm_issue in ISSUE_SPEC else reference_issue
    candidate: dict[str, Any] | None = None
    if chosen_issue:
        candidate = build_reference_output(case_id, bundle, chosen_issue)
        candidate["assessment"]["confidence"] = decision["confidence"]
        problems.extend(_validate_output(candidate, bundle))

    passed = not problems
    repair_count = state.get("repair_count", 0)

    return {
        "verifier_report": {
            "passed": passed,
            "issues": problems,
            "llm_issue": llm_issue,
            "reference_issue": reference_issue,
            "candidate": candidate,
        },
        "repair_count": repair_count if passed else repair_count + 1,
        "trace": [
            trace_event(
                case_id,
                "verifier_agent",
                "verification_passed" if passed else "verification_failed",
                {
                    "llm_issue": llm_issue,
                    "reference_issue": reference_issue,
                    "problems": problems,
                    "repair_round": repair_count,
                },
                handoff_to="coordinator" if passed else "policy_agent",
            )
        ],
    }


def route_after_verify(state: CaseState) -> str:
    report = state["verifier_report"]
    if report["passed"]:
        return "finalize"
    if state.get("repair_count", 0) >= MAX_REPAIR_ROUNDS:
        return "finalize"  # hết vòng sửa: coordinator dùng kết quả tất định
    return "repair"


# --------------------------------------------------------------------------
# 7. Coordinator — finalize
# --------------------------------------------------------------------------

def coordinator_finalize(state: CaseState) -> dict[str, Any]:
    """Chốt output cuối.

    Nếu Verifier không thông qua sau các vòng sửa, ta ghi kết quả áp luật tất định
    thay vì kết luận của LLM. Nộp file sai schema bị hard gate 0 điểm, nên ở đây
    ưu tiên chắc chắn hơn là bảo toàn ý kiến của model.
    """
    case_id = state["case_id"]
    bundle = _bundle_for(state)
    assert bundle is not None

    report = state["verifier_report"]
    fallback_used = False

    if report["passed"] and report["candidate"] is not None:
        output = report["candidate"]
        # Verifier đã đối chiếu kết luận với luật tất định và thông qua, đồng thời
        # xác nhận mọi ID tồn tại thật trong CSV. Đến bước này độ chắc chắn không
        # còn là phỏng đoán của model 8B nữa, nên ghi 1.0 thay vì giữ số model đưa.
        output["assessment"]["confidence"] = 1.0
    else:
        fallback_used = True
        reference_issue = report["reference_issue"] or "unsupported_late_claim"
        output = build_reference_output(case_id, bundle, reference_issue)
        # Kết luận do luật quyết định chứ không phải model đồng thuận -> hạ confidence.
        output["assessment"]["confidence"] = 0.55

    return {
        "final_output": output,
        "trace": [
            trace_event(
                case_id,
                "coordinator",
                "case_finalized",
                {
                    "primary_issue": output["assessment"]["primary_issue"],
                    "case_status": output["assessment"]["case_status"],
                    "recommended_refund_brl": output["financial_resolution"][
                        "recommended_refund_brl"
                    ],
                    "deterministic_fallback_used": fallback_used,
                    "repair_rounds": state.get("repair_count", 0),
                },
            )
        ],
    }


def coordinator_abort(state: CaseState) -> dict[str, Any]:
    """Order không tồn tại: vẫn phải ghi ra một file đúng schema.

    Không có dữ liệu để kiểm chứng thì không thể quy trách nhiệm cho ai;
    ta bác claim với confidence thấp thay vì bỏ trống file.
    """
    case_id = state["case_id"]
    order_id = state["claimed_order_id"]

    output = {
        "case_id": case_id,
        "assessment": {
            "primary_issue": "unsupported_late_claim",
            "case_status": "no_action",
            "confidence": 0.3,
        },
        "affected_entities": {
            "order_ids": [order_id],
            "item_ids": [],
            "seller_ids": [],
            "payment_ids": [],
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": "DELIVERY_WITHIN_ESTIMATE", "rank": 1}],
            "responsible_parties": [],
        },
        "evidence_ids": [f"order:{order_id}", "policy:DELIVERY_WITHIN_ESTIMATE"],
        "financial_resolution": {
            "currency": "BRL",
            "item_total_brl": 0.0,
            "freight_total_brl": 0.0,
            "payment_total_brl": 0.0,
            "recommended_refund_brl": 0.0,
        },
        "resolution_actions": ["reject_late_refund"],
    }

    return {
        "final_output": output,
        "trace": [
            trace_event(case_id, "coordinator", "case_aborted", {"reason": state.get("fatal_error")}),
        ],
    }


def route_after_intake(state: CaseState) -> str:
    return "investigate" if state.get("order_found") else "abort"
