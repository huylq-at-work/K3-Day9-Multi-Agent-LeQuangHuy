"""Nối các agent thành graph LangGraph.

Cấu trúc handoff:

                      coordinator_intake
                       |              |
              (order tồn tại)     (không tồn tại)
                       |              |
        +--------------+--------+     coordinator_abort
        |              |        |            |
  order_seller     payment   delivery       END
        |              |        |
        +--------------+--------+
                       |            <- barrier: gộp bằng chứng ba nhánh
                  policy_agent  <----------+
                       |                   |
                  verifier_agent           | (lệch luật -> gửi trả kèm lý do,
                       |                   |  tối đa 2 vòng)
              +--------+--------+----------+
              |
      coordinator_finalize
              |
             END

Ba agent chuyên môn chạy song song vì chúng đọc ba domain dữ liệu độc lập;
policy_agent là điểm gộp, nơi bằng chứng của cả ba được đối chiếu với nhau.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    coordinator_abort,
    coordinator_finalize,
    coordinator_intake,
    delivery_agent,
    order_seller_agent,
    payment_agent,
    policy_agent,
    route_after_intake,
    route_after_verify,
    verifier_agent,
)
from .state import CaseState


def build_graph():
    graph = StateGraph(CaseState)

    graph.add_node("coordinator_intake", coordinator_intake)
    # Node rỗng: LangGraph không cho một nhánh điều kiện trỏ thẳng tới nhiều node,
    # nên dùng điểm phân nhánh trung gian để fan-out ra ba agent chuyên môn.
    graph.add_node("dispatch", lambda _state: {})
    graph.add_node("order_seller_agent", order_seller_agent)
    graph.add_node("payment_agent", payment_agent)
    graph.add_node("delivery_agent", delivery_agent)
    graph.add_node("policy_agent", policy_agent)
    graph.add_node("verifier_agent", verifier_agent)
    graph.add_node("coordinator_finalize", coordinator_finalize)
    graph.add_node("coordinator_abort", coordinator_abort)

    graph.add_edge(START, "coordinator_intake")

    graph.add_conditional_edges(
        "coordinator_intake",
        route_after_intake,
        {
            "investigate": "dispatch",
            "abort": "coordinator_abort",
        },
    )

    for specialist in ("order_seller_agent", "payment_agent", "delivery_agent"):
        graph.add_edge("dispatch", specialist)

    # Ba nhánh cùng trỏ vào policy_agent: LangGraph chờ đủ cả ba mới chạy tiếp.
    graph.add_edge("order_seller_agent", "policy_agent")
    graph.add_edge("payment_agent", "policy_agent")
    graph.add_edge("delivery_agent", "policy_agent")

    graph.add_edge("policy_agent", "verifier_agent")

    graph.add_conditional_edges(
        "verifier_agent",
        route_after_verify,
        {
            "repair": "policy_agent",
            "finalize": "coordinator_finalize",
        },
    )

    graph.add_edge("coordinator_finalize", END)
    graph.add_edge("coordinator_abort", END)

    return graph.compile()
