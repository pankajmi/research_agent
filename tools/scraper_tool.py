"""
tools/scraper_tool.py
Lightweight HTML → text extractor used by gather_data_node.
"""

import logging
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchAgent/1.0; "
        "+https://github.com/your-org/research-agent)"
    )
}
TIMEOUT = 10


def scrape_page(url: str, max_chars: int = 3000) -> dict:
    """
    Fetch a URL and extract clean paragraph text.

    Returns:
        {url, title, content}  — content is truncated to max_chars
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url

        # Collect paragraph text
        paragraphs = [
            p.get_text(separator=" ", strip=True)
            for p in soup.find_all(["p", "li", "h1", "h2", "h3", "td"])
            if len(p.get_text(strip=True)) > 40
        ]
        content = " ".join(paragraphs)[:max_chars]

        log.debug("  scraped %s → %d chars", url, len(content))
        return {"url": url, "title": title, "content": content}

    except Exception as exc:
        log.warning("  scrape failed for %s: %s", url, exc)
        return {"url": url, "title": url, "content": ""}
