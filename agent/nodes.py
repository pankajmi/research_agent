"""
agent/nodes.py
Every node function in the LangGraph research agent.

Each function receives the current AgentState and returns a dict of
state updates (LangGraph merges them automatically).
"""

from __future__ import annotations
import json
import re
import logging
from datetime import datetime
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    MAX_SEARCH_RESULTS,
    MAX_SCRAPE_CHARS,
    SEARCH_QUERIES_COUNT,
    MIN_COMPETITORS,
    SUFFICIENT_DATA_KEYS,
)
from tools.search_tool import web_search
from tools.scraper_tool import scrape_page
from storage.cache import save_findings, load_findings
from report.generator import build_report
from agent.state import AgentState

log = logging.getLogger(__name__)

# ── Shared LLM instance (LangChain → Ollama abstraction) ─────────────────────


def _get_llm() -> OllamaLLM:
    """
    LangChain abstracts the LLM provider.  Swap OLLAMA_MODEL in settings.py
    or replace OllamaLLM with ChatOpenAI / ChatAnthropic without touching
    any node logic.
    """
    return OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_MAX_TOKENS,
    )


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Parse user query
# ══════════════════════════════════════════════════════════════════════════════


def parse_query_node(state: AgentState) -> dict:
    """
    Extract the target company name from the free-text user query using the LLM.
    """
    log.info("▶ Node: parse_query")
    llm = _get_llm()

    prompt = PromptTemplate.from_template(
        "Extract only the company name from this query. "
        "Reply with just the company name, nothing else.\n\nQuery: {query}"
    )
    chain = prompt | llm
    company_name = chain.invoke({"query": state["query"]}).strip().strip('"')

    log.info("  Company identified: %s", company_name)
    return {
        "company_name": company_name,
        "retry_count": 0,
        "data_sufficient": False,
        "competitors": [],
        "search_results": [],
        "scraped_pages": [],
        "raw_text": "",
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Web search
# ══════════════════════════════════════════════════════════════════════════════


def web_search_node(state: AgentState) -> dict:
    """
    Run multiple search queries for competitors.
    Uses the OpenAPI-spec DuckDuckGo tool from tools/search_tool.py.
    """
    log.info("▶ Node: web_search (retry=%d)", state["retry_count"])
    company = state["company_name"]

    # Generate diverse search queries via LLM
    llm = _get_llm()
    prompt = PromptTemplate.from_template(
        "Generate {n} distinct Google search queries to find competitors of '{company}'. "
        "Return one query per line, no numbering and no other lines at the start or end."
    )
    chain = prompt | llm
    raw_queries = chain.invoke({"n": SEARCH_QUERIES_COUNT, "company": company})
    queries = [q.strip() for q in raw_queries.strip().splitlines() if q.strip()][
        :SEARCH_QUERIES_COUNT
    ]

    log.info("  Search queries: %s", queries)

    all_results: list[dict] = []
    for q in queries:
        results = web_search(q, max_results=MAX_SEARCH_RESULTS)
        all_results.extend(results)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    log.info("  Found %d unique results", len(unique))
    return {"search_results": unique}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Data gathering (scrape pages)
# ══════════════════════════════════════════════════════════════════════════════


def gather_data_node(state: AgentState) -> dict:
    """
    Scrape the top search results and accumulate raw text for the LLM.
    """
    log.info("▶ Node: gather_data (%d results to scrape)", len(state["search_results"]))
    scraped: list[dict] = []
    combined_text_parts: list[str] = []

    for result in state["search_results"][:6]:  # limit to top 6 URLs
        page = scrape_page(result["url"], max_chars=MAX_SCRAPE_CHARS)
        if page["content"]:
            scraped.append(page)
            combined_text_parts.append(
                f"### Source: {page['title']} ({page['url']})\n{page['content']}"
            )

    raw_text = "\n\n".join(combined_text_parts)
    log.info("  Scraped %d pages, %d chars total", len(scraped), len(raw_text))
    return {
        "scraped_pages": scraped,
        "raw_text": raw_text,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Extract insights
# ══════════════════════════════════════════════════════════════════════════════

_EXTRACT_PROMPT = PromptTemplate.from_template(
    """You are a market research analyst. Analyze the text below about competitors of {company}.

Extract a JSON object with this exact structure:
{{
  "competitors": [
    {{
      "name": "...",
      "website": "...",
      "pricing": "...",
      "key_features": ["...", "..."],
      "target_market": "...",
      "strengths": ["...", "..."],
      "weaknesses": ["...", "..."],
      "positioning": "..."
    }}
  ],
  "market_trends": ["...", "..."],
  "strategic_gaps": ["...", "..."],
  "executive_summary": "..."
}}

Return ONLY valid JSON. No markdown fences. No explanation.

TEXT:
{text}
"""
)


def extract_insights_node(state: AgentState) -> dict:
    """
    Ask the local LLM to extract structured competitor data from raw scraped text.
    """
    log.info("▶ Node: extract_insights")
    llm = _get_llm()
    chain = _EXTRACT_PROMPT | llm

    raw_response = chain.invoke(
        {
            "company": state["company_name"],
            "text": state["raw_text"][:12000],  # stay within context window
        }
    )

    # Robust JSON extraction
    insights = _parse_json_safely(raw_response)
    competitors = insights.get("competitors", [])
    log.info("  Extracted %d competitors", len(competitors))

    return {
        "insights": insights,
        "competitors": competitors,
    }


def _parse_json_safely(text: str) -> dict:
    """Strip markdown fences and parse JSON; return empty dict on failure."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    # Find the first '{' to '}' block
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        log.warning("  No JSON block found in LLM response")
        return {}
    try:
        return json.loads(cleaned[start:end])
    except json.JSONDecodeError as exc:
        log.warning("  JSON parse error: %s", exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Check data sufficiency (conditional edge logic)
# ══════════════════════════════════════════════════════════════════════════════


def check_sufficiency_node(state: AgentState) -> dict:
    """
    Decide whether we have enough data to generate the report.
    If not, identify what's missing so the retry loop can search more specifically.
    """
    log.info("▶ Node: check_sufficiency")
    competitors = state.get("competitors", [])

    # Rule 1: need minimum number of competitors
    if len(competitors) < MIN_COMPETITORS:
        log.info(
            "  Insufficient competitors (%d < %d)", len(competitors), MIN_COMPETITORS
        )
        return {
            "data_sufficient": False,
            "missing_fields": [
                f"Need at least {MIN_COMPETITORS} competitors, found {len(competitors)}"
            ],
        }

    # Rule 2: each competitor must have required fields populated
    missing: list[str] = []
    for comp in competitors:
        for key in SUFFICIENT_DATA_KEYS:
            val = comp.get(key, "")
            if not val or val in ("unknown", "N/A", "..."):
                missing.append(f"{comp.get('name','?')} → missing '{key}'")

    if missing:
        log.info("  Missing fields: %s", missing[:3])
        return {"data_sufficient": False, "missing_fields": missing}

    log.info("  Data is sufficient — proceeding to report")
    return {"data_sufficient": True, "missing_fields": []}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Ask follow-up questions (retry branch)
# ══════════════════════════════════════════════════════════════════════════════


def followup_question_node(state: AgentState) -> dict:
    """
    When data is insufficient, use the LLM to generate better search queries
    targeting exactly what's missing.  Then bump the retry counter.
    """
    log.info("▶ Node: followup_question (retry %d)", state["retry_count"])
    llm = _get_llm()
    missing_summary = "; ".join(state.get("missing_fields", [])[:5])

    prompt = PromptTemplate.from_template(
        "We are researching competitors of '{company}'. "
        "The following information is still missing: {missing}. "
        "Generate 2 targeted web search queries to find this information. "
        "Return just one query per line, nothing else."
    )
    chain = prompt | llm
    raw = chain.invoke(
        {
            "company": state["company_name"],
            "missing": missing_summary,
        }
    )

    followup_queries = [q.strip() for q in raw.strip().splitlines() if q.strip()][:2]
    log.info("  Follow-up queries: %s", followup_queries)

    # Merge new search results into existing state
    extra_results: list[dict] = []
    for q in followup_queries:
        extra_results.extend(web_search(q, max_results=4))

    existing = state.get("search_results", [])
    seen = {r["url"] for r in existing}
    new_unique = [r for r in extra_results if r["url"] not in seen]

    return {
        "search_results": existing + new_unique,
        "retry_count": state["retry_count"] + 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 7 — Store findings
# ══════════════════════════════════════════════════════════════════════════════


def store_findings_node(state: AgentState) -> dict:
    """
    Persist structured findings to disk as JSON for caching and auditability.
    """
    log.info("▶ Node: store_findings")
    payload = {
        "query": state["query"],
        "company": state["company_name"],
        "competitors": state["competitors"],
        "insights": state["insights"],
        "scraped_urls": [p["url"] for p in state.get("scraped_pages", [])],
        "timestamp": datetime.utcnow().isoformat(),
    }
    cache_path = save_findings(state["company_name"], payload)
    log.info("  Saved to %s", cache_path)
    return {}  # no state changes needed


# ══════════════════════════════════════════════════════════════════════════════
# NODE 8 — Generate report
# ══════════════════════════════════════════════════════════════════════════════


def generate_report_node(state: AgentState) -> dict:
    """
    Build a PDF-ready HTML market report from the extracted insights.
    """
    log.info("▶ Node: generate_report")
    report_path = build_report(
        company_name=state["company_name"],
        insights=state["insights"],
        query=state["query"],
    )
    log.info("  Report written to %s", report_path)
    print(f"\n✅  Report ready: {report_path}\n")
    return {"report_path": str(report_path)}
