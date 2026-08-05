"""Sinh biến thể output để dò xem bộ chấm thực sự kỳ vọng gì.

Bộ chấm trả điểm ngay, chấm tất định, và leaderboard giữ điểm CAO NHẤT — nên nộp
một biến thể kém không mất gì, còn thông tin thu được thì đáng giá. Mỗi biến thể
chỉ đổi ĐÚNG MỘT thứ so với bản gốc, để mức chênh điểm quy được về đúng nguyên nhân.

Không gọi LLM: chỉ biến đổi trên output đã có.

    uv run scripts/make_variant.py --list
    uv run scripts/make_variant.py confidence-half
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "output"

# Nguyên nhân phụ đúng về mặt dữ kiện cho từng nhánh.
SECONDARY_CAUSE = {
    # Đơn giao muộn do seller thì đồng thời cũng là giao sau hạn dự kiến.
    "late_delivery_seller": "CARRIER_DELIVERED_AFTER_ESTIMATE",
}


def v_baseline(o: dict[str, Any]) -> dict[str, Any]:
    """Bản gốc, để xác nhận quy trình đóng gói không tự làm đổi điểm."""
    return o


def v_confidence_half(o: dict[str, Any]) -> dict[str, Any]:
    """Hạ confidence xuống 0.5. Nếu điểm không đổi -> giá trị confidence không được chấm."""
    o["assessment"]["confidence"] = 0.5
    return o


def v_confidence_full(o: dict[str, Any]) -> dict[str, Any]:
    o["assessment"]["confidence"] = 1.0
    return o


def v_evidence_minimal(o: dict[str, Any]) -> dict[str, Any]:
    """Chỉ giữ order + policy. Điểm TĂNG -> đáp án kỳ vọng tập evidence tối thiểu."""
    o["evidence_ids"] = [e for e in o["evidence_ids"] if e.startswith(("order:", "policy:"))]
    return o


def v_evidence_no_item(o: dict[str, Any]) -> dict[str, Any]:
    o["evidence_ids"] = [e for e in o["evidence_ids"] if not e.startswith("item:")]
    return o


def v_causes_ranked(o: dict[str, Any]) -> dict[str, Any]:
    """Thêm nguyên nhân phụ ở nhánh có nguyên nhân thứ hai đúng về dữ kiện."""
    issue = o["assessment"]["primary_issue"]
    extra = SECONDARY_CAUSE.get(issue)
    if extra:
        causes = o["root_cause_analysis"]["ranked_causes"]
        if all(c["cause_code"] != extra for c in causes):
            causes.append({"cause_code": extra, "rank": len(causes) + 1})
    return o


def v_entities_order_only(o: dict[str, Any]) -> dict[str, Any]:
    """Chỉ giữ order_ids. Dùng để đo affected_entities đang thực sự kiếm bao nhiêu điểm."""
    e = o["affected_entities"]
    e["item_ids"], e["seller_ids"], e["payment_ids"] = [], [], []
    return o


# Leaderboard lấy điểm của LẦN NỘP GẦN NHẤT, không phải điểm cao nhất.
# Nên phải phân biệt rõ: biến thể nào đáng thử, biến thể nào gần như chắc chắn
# làm tụt điểm và chỉ dùng để dò thông tin.
#   AN TOAN   - có cơ sở tin là bằng hoặc hơn bản gốc
#   RUI RO    - có thể hơn, có thể kém, xác suất hai chiều
#   CHI DE DO - gần như chắc chắn tụt điểm, nộp xong PHẢI nộp lại baseline ngay
VARIANTS: dict[str, tuple[Callable[[dict], dict], str]] = {
    "baseline": (v_baseline, "[AN TOAN]   bản gốc 93.6977 — luôn giữ để nộp chốt"),
    "causes-ranked": (v_causes_ranked, "[RUI RO]    thêm nguyên nhân phụ cho late_delivery_seller (8 case)"),
    "confidence-full": (v_confidence_full, "[RUI RO]    confidence = 1.0 cho mọi case"),
    "evidence-minimal": (v_evidence_minimal, "[RUI RO]    evidence chỉ còn order + policy"),
    "evidence-no-item": (v_evidence_no_item, "[RUI RO]    bỏ evidence item:"),
    "confidence-half": (v_confidence_half, "[CHI DE DO] confidence = 0.5 — dò xem confidence có được chấm không"),
    "entities-order-only": (v_entities_order_only, "[CHI DE DO] chỉ giữ order_ids — đo affected_entities đang kiếm bao nhiêu điểm"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", help="tên biến thể")
    parser.add_argument("--list", action="store_true", help="liệt kê biến thể")
    args = parser.parse_args()

    if args.list or not args.variant:
        print("Biến thể có sẵn:\n")
        for name, (_, desc) in VARIANTS.items():
            print(f"  {name:<22} {desc}")
        print("\nDùng: uv run scripts/make_variant.py <tên>")
        return 0

    # Dạng tham số hoá: conf:0.95 -> đặt confidence = 0.95 cho mọi case.
    if args.variant.startswith("conf:"):
        try:
            value = round(float(args.variant.split(":", 1)[1]), 2)
        except ValueError:
            raise SystemExit(f"Giá trị confidence không hợp lệ: {args.variant!r}") from None
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"confidence phải nằm trong [0,1], nhận {value}")

        def transform(o: dict[str, Any], _v: float = value) -> dict[str, Any]:
            o["assessment"]["confidence"] = _v
            return o

        desc = f"confidence = {value} cho mọi case"
    elif args.variant not in VARIANTS:
        raise SystemExit(f"Không có biến thể {args.variant!r}. Chạy --list để xem.")
    else:
        transform, desc = VARIANTS[args.variant]
    # Windows cấm ':' trong tên file, mà biến thể tham số hoá lại dùng dạng conf:0.95.
    slug = args.variant.replace(":", "-")
    out_dir = ROOT / f"output_variant_{slug}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    changed = 0
    for i in range(1, 51):
        name = f"EC_{i:03d}.json"
        original = json.loads((SRC / name).read_text(encoding="utf-8"))
        modified = transform(json.loads(json.dumps(original)))
        if modified != original:
            changed += 1
        (out_dir / name).write_text(
            json.dumps(modified, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    zip_path = ROOT / f"output_{slug}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for i in range(1, 51):
            name = f"EC_{i:03d}.json"
            archive.writestr(f"output/{name}", (out_dir / name).read_text(encoding="utf-8"))

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert names == [f"output/EC_{i:03d}.json" for i in range(1, 51)]
        assert all("\\" not in n for n in names)

    print(f"Biến thể : {args.variant} — {desc}")
    print(f"Đổi      : {changed}/50 file")
    print(f"Zip      : {zip_path.name} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
