"""Tầng truy xuất dữ liệu Olist.

Toàn bộ số liệu tiền tệ và mốc thời gian trong hệ thống đều sinh ra từ đây.
Agent LLM chỉ được đọc kết quả của module này, không bao giờ tự tính hay tự nhớ số.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Sai số cho phép khi đối soát payment với item + freight (theo EC_POLICY_V1).
PAYMENT_TOLERANCE_BRL = 0.10


def _parse_ts(value: Any) -> datetime | None:
    """CSV Olist dùng 'YYYY-MM-DD HH:MM:SS'; ô rỗng là NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat"}:
        return None
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat(sep=" ") if ts else None


def _money(value: float) -> float:
    """Mọi phép tính tiền làm tròn 2 chữ số thập phân."""
    return round(float(value) + 1e-9, 2)


@dataclass
class OrderRecord:
    order_id: str
    customer_id: str
    order_status: str
    purchase_ts: datetime | None
    approved_ts: datetime | None
    delivered_carrier_ts: datetime | None
    delivered_customer_ts: datetime | None
    estimated_delivery_ts: datetime | None


@dataclass
class ItemRecord:
    order_id: str
    order_item_id: int
    product_id: str
    seller_id: str
    shipping_limit_ts: datetime | None
    price: float
    freight_value: float


@dataclass
class PaymentRecord:
    order_id: str
    payment_sequential: int
    payment_type: str
    payment_installments: int
    payment_value: float


@dataclass
class OrderBundle:
    """Toàn bộ dữ liệu có thể kiểm chứng của một order, đã join sẵn."""

    order: OrderRecord
    items: list[ItemRecord] = field(default_factory=list)
    payments: list[PaymentRecord] = field(default_factory=list)

    # ---- Tổng hợp tài chính ----
    @property
    def item_total_brl(self) -> float:
        return _money(sum(i.price for i in self.items))

    @property
    def freight_total_brl(self) -> float:
        return _money(sum(i.freight_value for i in self.items))

    @property
    def payment_total_brl(self) -> float:
        return _money(sum(p.payment_value for p in self.payments))

    @property
    def payment_delta_brl(self) -> float:
        """payment_total - (item_total + freight_total). Dương = khách trả dư."""
        expected = _money(self.item_total_brl + self.freight_total_brl)
        return _money(self.payment_total_brl - expected)

    @property
    def payment_reconciles(self) -> bool:
        return abs(self.payment_delta_brl) <= PAYMENT_TOLERANCE_BRL

    # ---- Tổng hợp giao hàng ----
    @property
    def delivered_late(self) -> bool:
        o = self.order
        if o.delivered_customer_ts is None or o.estimated_delivery_ts is None:
            return False
        return o.delivered_customer_ts > o.estimated_delivery_ts

    @property
    def late_seller_ids(self) -> list[str]:
        """Seller bàn giao muộn: order_delivered_carrier_date > shipping_limit_date của item họ."""
        carrier = self.order.delivered_carrier_ts
        if carrier is None:
            return []
        late: list[str] = []
        for item in self.items:
            if item.shipping_limit_ts is not None and carrier > item.shipping_limit_ts:
                if item.seller_id not in late:
                    late.append(item.seller_id)
        return late

    @property
    def seller_ids(self) -> list[str]:
        out: list[str] = []
        for item in self.items:
            if item.seller_id not in out:
                out.append(item.seller_id)
        return out

    def to_facts(self) -> dict[str, Any]:
        """Bản tóm tắt phẳng, an toàn để đưa vào prompt LLM."""
        o = self.order
        return {
            "order_id": o.order_id,
            "order_status": o.order_status,
            "order_purchase_timestamp": _iso(o.purchase_ts),
            "order_delivered_carrier_date": _iso(o.delivered_carrier_ts),
            "order_delivered_customer_date": _iso(o.delivered_customer_ts),
            "order_estimated_delivery_date": _iso(o.estimated_delivery_ts),
            "item_count": len(self.items),
            "payment_row_count": len(self.payments),
            "item_total_brl": self.item_total_brl,
            "freight_total_brl": self.freight_total_brl,
            "payment_total_brl": self.payment_total_brl,
            "payment_delta_brl": self.payment_delta_brl,
            "payment_reconciles": self.payment_reconciles,
            "delivered_late": self.delivered_late,
            "seller_ids": self.seller_ids,
            "late_seller_ids": self.late_seller_ids,
            "items": [
                {
                    "order_item_id": i.order_item_id,
                    "seller_id": i.seller_id,
                    "shipping_limit_date": _iso(i.shipping_limit_ts),
                    "price": _money(i.price),
                    "freight_value": _money(i.freight_value),
                }
                for i in self.items
            ],
            "payments": [
                {
                    "payment_sequential": p.payment_sequential,
                    "payment_type": p.payment_type,
                    "payment_installments": p.payment_installments,
                    "payment_value": _money(p.payment_value),
                }
                for p in self.payments
            ],
        }


class OlistDataset:
    """Index in-memory của 4 bảng cần dùng, join theo order_id."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self._orders: dict[str, OrderRecord] = {}
        self._items: dict[str, list[ItemRecord]] = {}
        self._payments: dict[str, list[PaymentRecord]] = {}
        self._seller_ids: set[str] = set()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        orders = pd.read_csv(self.data_dir / "olist_orders_dataset.csv")
        for row in orders.itertuples(index=False):
            self._orders[row.order_id] = OrderRecord(
                order_id=row.order_id,
                customer_id=row.customer_id,
                order_status=str(row.order_status).strip().lower(),
                purchase_ts=_parse_ts(row.order_purchase_timestamp),
                approved_ts=_parse_ts(row.order_approved_at),
                delivered_carrier_ts=_parse_ts(row.order_delivered_carrier_date),
                delivered_customer_ts=_parse_ts(row.order_delivered_customer_date),
                estimated_delivery_ts=_parse_ts(row.order_estimated_delivery_date),
            )

        items = pd.read_csv(self.data_dir / "olist_order_items_dataset.csv")
        for row in items.itertuples(index=False):
            self._items.setdefault(row.order_id, []).append(
                ItemRecord(
                    order_id=row.order_id,
                    order_item_id=int(row.order_item_id),
                    product_id=row.product_id,
                    seller_id=row.seller_id,
                    shipping_limit_ts=_parse_ts(row.shipping_limit_date),
                    price=float(row.price),
                    freight_value=float(row.freight_value),
                )
            )
        for bucket in self._items.values():
            bucket.sort(key=lambda i: i.order_item_id)

        payments = pd.read_csv(self.data_dir / "olist_order_payments_dataset.csv")
        for row in payments.itertuples(index=False):
            self._payments.setdefault(row.order_id, []).append(
                PaymentRecord(
                    order_id=row.order_id,
                    payment_sequential=int(row.payment_sequential),
                    payment_type=str(row.payment_type),
                    payment_installments=int(row.payment_installments),
                    payment_value=float(row.payment_value),
                )
            )
        for bucket in self._payments.values():
            bucket.sort(key=lambda p: p.payment_sequential)

        sellers = pd.read_csv(self.data_dir / "olist_sellers_dataset.csv")
        self._seller_ids = set(sellers["seller_id"].astype(str))

        self._loaded = True

    def get_bundle(self, order_id: str) -> OrderBundle | None:
        self.load()
        order = self._orders.get(order_id)
        if order is None:
            return None
        return OrderBundle(
            order=order,
            items=list(self._items.get(order_id, [])),
            payments=list(self._payments.get(order_id, [])),
        )

    def seller_exists(self, seller_id: str) -> bool:
        self.load()
        return seller_id in self._seller_ids


@functools.lru_cache(maxsize=1)
def get_dataset() -> OlistDataset:
    """Singleton — 100k+ dòng CSV chỉ nạp một lần cho cả 50 case."""
    ds = OlistDataset()
    ds.load()
    return ds
