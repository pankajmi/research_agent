"""
agent/graph.py
Assembles the LangGraph state machine for the research agent.

Flow:
  parse_query
      ↓
  web_search
      ↓
  gather_data  ←──────────────────────────────────┐
      ↓                                            │
  extract_insights                                 │  (retry loop)
      ↓                                            │
  check_sufficiency ── insufficient + retries left ┤
      │                                            │
      │  insufficient + max retries hit            │
      │  ↓                                         │
      │  followup_question ─────────────────────── ┘
      │
      │  sufficient
      ↓
  store_findings
      ↓
  generate_report
      ↓
  END
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    parse_query_node,
    web_search_node,
    gather_data_node,
    extract_insights_node,
    check_sufficiency_node,
    followup_question_node,
    store_findings_node,
    generate_report_node,
)
from config.settings import MAX_RETRY_LOOPS


def _route_after_sufficiency_check(state: AgentState) -> str:
    """
    Conditional edge after check_sufficiency_node.

    Returns the name of the next node to execute:
      - "store_findings"      if data is good enough
      - "followup_question"   if we still have retries left
      - "store_findings"      if we've exhausted retries (generate best-effort report)
    """
    if state["data_sufficient"]:
        return "store_findings"
    if state["retry_count"] < MAX_RETRY_LOOPS:
        return "followup_question"
    # Give up retrying — generate a best-effort report with whatever we have
    return "store_findings"


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph research agent.

    Every node is a pure function: AgentState → dict[str, Any].
    LangGraph merges returned dicts back into the shared state automatically.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────────
    graph.add_node("parse_query",       parse_query_node)
    graph.add_node("web_search",        web_search_node)
    graph.add_node("gather_data",       gather_data_node)
    graph.add_node("extract_insights",  extract_insights_node)
    graph.add_node("check_sufficiency", check_sufficiency_node)
    graph.add_node("followup_question", followup_question_node)
    graph.add_node("store_findings",    store_findings_node)
    graph.add_node("generate_report",   generate_report_node)

    # ── Entry point ─────────────────────────────────────────────────────────
    graph.set_entry_point("parse_query")

    # ── Linear edges ────────────────────────────────────────────────────────
    graph.add_edge("parse_query",       "web_search")
    graph.add_edge("web_search",        "gather_data")
    graph.add_edge("gather_data",       "extract_insights")
    graph.add_edge("extract_insights",  "check_sufficiency")

    # ── Conditional edge — retry loop or proceed ─────────────────────────────
    graph.add_conditional_edges(
        "check_sufficiency",
        _route_after_sufficiency_check,
        {
            "store_findings":    "store_findings",
            "followup_question": "followup_question",
        },
    )

    # ── Retry loop: followup → gather → extract → check again ───────────────
    graph.add_edge("followup_question", "gather_data")

    # ── Final linear edges ───────────────────────────────────────────────────
    graph.add_edge("store_findings",  "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
