"""Tự chấm output theo trọng số ở README mục 8.

LƯU Ý QUAN TRỌNG VỀ Ý NGHĨA CON SỐ:
Ta KHÔNG có đáp án chính thức. Script chấm theo hai lớp khác nhau:

  (A) So với engine luật tất định của chính ta -> đo mức ĐỒNG THUẬN, không phải
      độ chính xác thật. Nếu ta hiểu sai luật thì cả hai cùng sai và điểm vẫn 100%.
  (B) Kiểm chứng độc lập ngược về CSV -> cái này là SỰ THẬT KHÁCH QUAN: mọi ID có
      tồn tại không, mọi con số tiền có cộng đúng không, schema có hợp lệ không.

Lớp (B) mới là thứ chặn được hard gate. Lớp (A) chỉ nói model 8B có bám luật hay không.

    uv run scripts/score_outputs.py [--dir output]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.dataset import get_dataset  # noqa: E402
from agents.policy import (  # noqa: E402
    ISSUE_SPEC,
    build_reference_output,
    first_matching_issue,
)

WEIGHTS = {
    "primary_issue": 20.0,
    "affected_entities": 20.0,
    "root_cause": 15.0,
    "evidence": 15.0,
    "financial": 20.0,
    "actions": 10.0,
}

INPUT_DIR = ROOT / "input"


def f1(got: list[str], want: list[str]) -> float:
    g, w = set(got), set(want)
    if not g and not w:
        return 1.0
    if not g or not w:
        return 0.0
    inter = len(g & w)
    if inter == 0:
        return 0.0
    precision, recall = inter / len(g), inter / len(w)
    return 2 * precision * recall / (precision + recall)


def score_case(got: dict, ref: dict) -> tuple[dict[str, float], list[str]]:
    """Chấm từng hạng mục theo trọng số. Trả về (điểm, ghi chú lệch)."""
    parts: dict[str, float] = {}
    notes: list[str] = []

    # 1. Primary issue + confidence (20%)
    issue_ok = got["assessment"]["primary_issue"] == ref["assessment"]["primary_issue"]
    conf = got["assessment"]["confidence"]
    conf_ok = isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0
    status_ok = got["assessment"]["case_status"] == ref["assessment"]["case_status"]
    parts["primary_issue"] = WEIGHTS["primary_issue"] * (
        0.7 * issue_ok + 0.2 * status_ok + 0.1 * conf_ok
    )
    if not issue_ok:
        notes.append(
            f"issue: {got['assessment']['primary_issue']} != {ref['assessment']['primary_issue']}"
        )
    if not status_ok:
        notes.append("case_status lệch")

    # 2. Affected entities (20%) - trung bình F1 của 4 tập
    ge, re_ = got["affected_entities"], ref["affected_entities"]
    ent_scores = [f1(ge[k], re_[k]) for k in ("order_ids", "item_ids", "seller_ids", "payment_ids")]
    parts["affected_entities"] = WEIGHTS["affected_entities"] * (sum(ent_scores) / 4)
    for k, s in zip(("order_ids", "item_ids", "seller_ids", "payment_ids"), ent_scores):
        if s < 1.0:
            notes.append(f"{k} F1={s:.2f}")

    # 3. Root cause + responsible parties (15%)
    gc = [c["cause_code"] for c in got["root_cause_analysis"]["ranked_causes"]]
    rc = [c["cause_code"] for c in ref["root_cause_analysis"]["ranked_causes"]]
    gp = [f"{p['party_type']}:{p['party_id']}" for p in got["root_cause_analysis"]["responsible_parties"]]
    rp = [f"{p['party_type']}:{p['party_id']}" for p in ref["root_cause_analysis"]["responsible_parties"]]
    parts["root_cause"] = WEIGHTS["root_cause"] * (0.5 * f1(gc, rc) + 0.5 * f1(gp, rp))
    if f1(gc, rc) < 1.0:
        notes.append(f"cause {gc} != {rc}")
    if f1(gp, rp) < 1.0:
        notes.append(f"parties {gp} != {rp}")

    # 4. Evidence IDs (15%)
    ev = f1(got["evidence_ids"], ref["evidence_ids"])
    parts["evidence"] = WEIGHTS["evidence"] * ev
    if ev < 1.0:
        notes.append(f"evidence F1={ev:.2f}")

    # 5. Financial resolution (20%) - 4 trường, mỗi trường 1/4
    gf, rf = got["financial_resolution"], ref["financial_resolution"]
    money_keys = ("item_total_brl", "freight_total_brl", "payment_total_brl", "recommended_refund_brl")
    hits = sum(abs(gf[k] - rf[k]) <= 0.01 for k in money_keys)
    currency_ok = gf["currency"] == "BRL"
    parts["financial"] = WEIGHTS["financial"] * (hits / len(money_keys)) * (1.0 if currency_ok else 0.5)
    for k in money_keys:
        if abs(gf[k] - rf[k]) > 0.01:
            notes.append(f"{k}={gf[k]} != {rf[k]}")

    # 6. Resolution actions (10%)
    act = f1(got["resolution_actions"], ref["resolution_actions"])
    parts["actions"] = WEIGHTS["actions"] * act
    if act < 1.0:
        notes.append(f"actions {got['resolution_actions']} != {ref['resolution_actions']}")

    return parts, notes


def hard_gate_check(got: dict, order_id: str) -> list[str]:
    """Kiểm chứng khách quan ngược về CSV — đây mới là thứ chặn hard gate."""
    problems: list[str] = []
    ds = get_dataset()
    bundle = ds.get_bundle(order_id)

    required = {
        "case_id", "assessment", "affected_entities",
        "root_cause_analysis", "evidence_ids", "financial_resolution", "resolution_actions",
    }
    missing = required - set(got)
    if missing:
        problems.append(f"thiếu trường: {sorted(missing)}")
        return problems

    if got["assessment"]["primary_issue"] not in ISSUE_SPEC:
        problems.append("primary_issue không thuộc 6 giá trị hợp lệ")
    conf = got["assessment"]["confidence"]
    if not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
        problems.append(f"confidence ngoài [0,1]: {conf}")
    if got["assessment"]["case_status"] not in ("action_required", "no_action"):
        problems.append("case_status không hợp lệ")

    ents = got["affected_entities"]
    for key, limit in (("order_ids", 5), ("item_ids", 5), ("seller_ids", 5), ("payment_ids", 5)):
        if len(ents[key]) > limit:
            problems.append(f"{key} vượt giới hạn {limit}")
    if len(got["evidence_ids"]) > 10:
        problems.append("evidence_ids vượt 10")
    if len(got["root_cause_analysis"]["ranked_causes"]) > 3:
        problems.append("ranked_causes vượt 3")
    if len(got["root_cause_analysis"]["responsible_parties"]) > 3:
        problems.append("responsible_parties vượt 3")
    if len(got["resolution_actions"]) > 5:
        problems.append("resolution_actions vượt 5")

    if bundle is None:
        problems.append(f"order {order_id} không có trong CSV")
        return problems

    # ID phải tồn tại thật
    valid_items = {f"{order_id}:{i.order_item_id}" for i in bundle.items}
    valid_pays = {f"{order_id}:{p.payment_sequential}" for p in bundle.payments}
    for iid in ents["item_ids"]:
        if iid not in valid_items:
            problems.append(f"item_id ma: {iid}")
    for pid in ents["payment_ids"]:
        if pid not in valid_pays:
            problems.append(f"payment_id ma: {pid}")
    for sid in ents["seller_ids"]:
        if not ds.seller_exists(sid):
            problems.append(f"seller_id ma: {sid}")
    for ev in got["evidence_ids"]:
        if not ev.startswith(("order:", "item:", "payment:", "seller:", "policy:")):
            problems.append(f"evidence sai định dạng: {ev}")

    # Số tiền phải cộng đúng từ CSV
    fin = got["financial_resolution"]
    for key, truth in (
        ("item_total_brl", bundle.item_total_brl),
        ("freight_total_brl", bundle.freight_total_brl),
        ("payment_total_brl", bundle.payment_total_brl),
    ):
        if abs(fin[key] - truth) > 0.01:
            problems.append(f"{key}={fin[key]} nhưng CSV cho {truth}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    cases = {}
    for p in sorted(INPUT_DIR.glob("EC_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        cases[data["case_id"]] = data["customer_request"]["claimed_order_id"]

    files = sorted(args.dir.glob("EC_*.json"))
    if not files:
        raise SystemExit(f"Không có output nào trong {args.dir}")

    print(f"Chấm {len(files)} file trong {args.dir.name}/\n")
    header = f"{'case':<8} {'issue':<24} {'điểm':>6}  {'gate':<5} ghi chú"
    print(header)
    print("-" * len(header))

    totals: dict[str, float] = {k: 0.0 for k in WEIGHTS}
    grand = 0.0
    gate_failures = 0

    for path in files:
        case_id = path.stem
        got = json.loads(path.read_text(encoding="utf-8"))
        order_id = cases.get(case_id, "")
        bundle = get_dataset().get_bundle(order_id)

        if bundle is None:
            print(f"{case_id:<8} {'(order không tồn tại)':<24} {'--':>6}  --")
            continue

        ref_issue = first_matching_issue(bundle) or "unsupported_late_claim"
        ref = build_reference_output(case_id, bundle, ref_issue)

        parts, notes = score_case(got, ref)
        total = sum(parts.values())
        grand += total
        for k, v in parts.items():
            totals[k] += v

        gate = hard_gate_check(got, order_id)
        gate_failures += bool(gate)
        gate_mark = "FAIL" if gate else "ok"

        note_text = "; ".join(notes + [f"[GATE] {g}" for g in gate])
        print(f"{case_id:<8} {got['assessment']['primary_issue']:<24} {total:>6.1f}  {gate_mark:<5} {note_text}")

    n = len([p for p in files])
    print("\n" + "=" * 60)
    print(f"ĐIỂM TRUNG BÌNH: {grand / n:.2f} / 100   (trên {n} case đã chạy)")
    print("\nBóc theo hạng mục:")
    for k, w in WEIGHTS.items():
        print(f"  {k:<20} {totals[k] / n:>6.2f} / {w:>5.1f}")
    print(f"\nHard gate: {n - gate_failures}/{n} case đạt")
    print(
        "\nLƯU Ý: điểm trên đo mức ĐỒNG THUẬN với engine luật của chính ta, không phải\n"
        "đáp án chính thức. Cột 'gate' mới là kiểm chứng khách quan ngược về CSV."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
