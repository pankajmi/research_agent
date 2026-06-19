"""
main.py
Entry point for the AI Product Research Agent.

Usage:
    python main.py
    python main.py --query "Analyze Salesforce competitors and create a market report"
    python main.py --query "Who competes with HubSpot?" --verbose
"""

# Suppress cosmetic warnings before any third-party imports
import warnings

warnings.filterwarnings("ignore", message=".*NotOpenSSL.*")
warnings.filterwarnings("ignore", message=".*allowed_objects.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")

import argparse
import logging
import sys

from agent.graph import build_graph
from agent.state import AgentState

DEFAULT_QUERY = "Analyze Impact Analytics competitors and create a market report."


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress noisy third-party loggers
    for lib in (
        "httpx",
        "urllib3",
        "requests",
        "duckduckgo_search",
        "ddgs",
        "langchain",
        "langgraph",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)

# def simulateRuntimeError():
#     raise RuntimeError("Simulated runtime error for testing purposes.")

def main() -> None:
    # simulateRuntimeError()
    parser = argparse.ArgumentParser(description="AI Product Research Agent")
    parser.add_argument(
        "--query",
        "-q",
        default=DEFAULT_QUERY,
        help="Research query (default: Impact Analytics competitor analysis)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show debug-level logs",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("main")

    print("\n" + "═" * 60)
    print("  AI Product Research Agent")
    print("  LangGraph + LangChain + Ollama (llama3)")
    print("═" * 60)
    print(f"\n  Query: {args.query}\n")

    # Build the LangGraph state machine
    graph = build_graph()

    # Initial state — only 'query' is required; all other fields set by nodes
    initial_state: AgentState = {
        "query": args.query,
        "company_name": "",
        "search_results": [],
        "scraped_pages": [],
        "competitors": [],
        "raw_text": "",
        "retry_count": 0,
        "data_sufficient": False,
        "missing_fields": [],
        "insights": {},
        "report_path": "",
        "error": None,
    }

    # Run the graph — LangGraph executes nodes sequentially / conditionally
    try:
        final_state = graph.invoke(initial_state)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as exc:
        log.error("Agent failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Summary
    report_path = final_state.get("report_path", "")
    comp_count = len(final_state.get("competitors", []))
    retries = final_state.get("retry_count", 0)

    print("\n" + "═" * 60)
    print(f"  ✅  Done!")
    print(f"  Competitors found : {comp_count}")
    print(f"  Search retries    : {retries}")
    if report_path:
        print(f"  Report            : {report_path}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
