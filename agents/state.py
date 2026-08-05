"""State dùng chung cho graph — chính là "bảng tin" mà các agent handoff cho nhau.

Mỗi agent chỉ ghi vào khóa của riêng mình và chỉ đọc khóa của agent thượng nguồn.
Ranh giới này được ghi rõ trong architecture.md và là thứ khiến các agent
thực sự bàn giao công việc, thay vì cùng chia sẻ một prompt khổng lồ.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class CaseState(TypedDict, total=False):
    # --- Coordinator ghi ---
    case_id: str
    opened_at: str
    customer_message: str
    claimed_order_id: str
    policy_version: str
    order_found: bool
    fatal_error: str

    # --- Ba agent chuyên môn ghi song song ---
    order_facts: dict[str, Any]
    payment_facts: dict[str, Any]
    delivery_facts: dict[str, Any]

    # --- Policy Agent ghi ---
    policy_decision: dict[str, Any]

    # --- Verifier ghi ---
    verifier_report: dict[str, Any]
    repair_count: int

    # --- Coordinator finalize ---
    final_output: dict[str, Any]

    # Trace gộp từ mọi nhánh song song; operator.add để hai nhánh không ghi đè nhau.
    trace: Annotated[list[dict[str, Any]], operator.add]
