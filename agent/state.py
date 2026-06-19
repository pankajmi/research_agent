"""
agent/state.py
Typed state passed between every LangGraph node.
"""

from __future__ import annotations
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    query: str                          # original user query
    company_name: str                   # extracted target company

    # ── Intermediate data ──────────────────────────────────────────────────────
    search_results: list[dict]          # raw results from web search
    scraped_pages: list[dict]           # {url, title, content} per page
    competitors: list[dict]             # structured competitor profiles
    raw_text: Annotated[str, operator.add]  # accumulated raw text (append-only)

    # ── Control flow ───────────────────────────────────────────────────────────
    retry_count: int                    # how many search loops so far
    data_sufficient: bool               # gate before report generation
    missing_fields: list[str]           # what the LLM says is still missing

    # ── Output ─────────────────────────────────────────────────────────────────
    insights: dict[str, Any]            # extracted key insights
    report_path: str                    # path to generated HTML report
    error: Optional[str]                # last error message if any
