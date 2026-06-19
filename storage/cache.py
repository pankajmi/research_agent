"""
storage/cache.py
Simple JSON-based findings cache.

Saves structured competitor data to disk so:
  1. Re-runs skip expensive LLM + scraping for the same company
  2. Results are auditable and inspectable
  3. The report generator can read cached data without re-running the agent
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import CACHE_DIR

log = logging.getLogger(__name__)


def _cache_path(company_name: str) -> Path:
    safe = company_name.lower().replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.json"


def save_findings(company_name: str, payload: dict) -> Path:
    """
    Persist findings dict to a per-company JSON file.
    Overwrites previous run for the same company.
    """
    path = _cache_path(company_name)
    payload["_saved_at"] = datetime.utcnow().isoformat()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.debug("  Cache saved: %s", path)
    return path


def load_findings(company_name: str) -> Optional[dict]:
    """
    Load cached findings for a company. Returns None if not found.
    """
    path = _cache_path(company_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        log.info("  Cache hit for %s (saved at %s)", company_name, data.get("_saved_at"))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("  Cache read error for %s: %s", company_name, exc)
        return None


def list_cached_companies() -> list[str]:
    """Return list of company names that have cached findings."""
    return [p.stem.replace("_", " ").title() for p in CACHE_DIR.glob("*.json")]
