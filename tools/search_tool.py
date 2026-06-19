"""
tools/search_tool.py
Web search via DuckDuckGo.

OpenAPI Spec (tools/openapi_search.yaml) documents this as an external tool
so any OpenAPI-aware client can call it without knowing the implementation.

The function signature mirrors what you'd define in an OpenAPI spec:
  POST /search
  body: { query: string, max_results: integer }
  response: [{ title, url, snippet }]
"""

import logging
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

log = logging.getLogger(__name__)

# ── OpenAPI tool schema ────────────────────────────────────────────────────────
# This dict describes the tool in the format LangChain / LangGraph expects
# when you register a custom tool.  It matches OpenAPI 3.0 structure so you
# can also expose this as a REST endpoint and generate a spec from it.
SEARCH_TOOL_SPEC = {
    "name": "web_search",
    "description": (
        "Search the web for information about a company's competitors, "
        "market positioning, pricing, and product features."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query":       {"type": "string",  "description": "Search query string"},
            "max_results": {"type": "integer", "description": "Max results to return", "default": 8},
        },
        "required": ["query"],
    },
}


def web_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Execute a web search and return structured results.

    Returns:
        List of dicts with keys: title, url, snippet
    """
    log.debug("  web_search: %r (max=%d)", query, max_results)
    try:
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results)
            return [
                {
                    "title":   r.get("title", ""),
                    "url":     r.get("href",  ""),
                    "snippet": r.get("body",  ""),
                }
                for r in raw
                if r.get("href")
            ]
    except Exception as exc:
        log.warning("  web_search failed for %r: %s", query, exc)
        return []
