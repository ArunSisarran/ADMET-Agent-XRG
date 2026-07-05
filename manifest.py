"""Persistent JSON manifest of every paper the pipeline has looked at.

Prevents re-downloading/re-extracting the same paper across runs and gives a
visible audit trail. Both discover.py and knowledge_extractor.py's manual
single-paper path write to this manifest, keyed by candidates.dedupe_key().

Writes are saved immediately after every record() call rather than batched at
the end of a run, so a crash mid-run doesn't lose already-decided papers or
force re-scoring abstracts that were already filtered.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent / "processed_papers.json"

# Bump this when the relevance-filter prompt changes meaningfully, so stale
# rejections made under an older prompt can be identified and re-checked
# instead of being buried permanently.
FILTER_VERSION = "v1"


def load() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)


def record(
    manifest: dict,
    key: str,
    status: str,
    *,
    title: str | None = None,
    doi: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    reason: str | None = None,
    extracted_file: str | None = None,
) -> None:
    """status: extracted | rejected | skipped_no_oa_pdf | skipped_text_extraction_failed | error"""
    manifest[key] = {
        "title": title,
        "doi": doi,
        "source": source,
        "source_type": source_type,
        "status": status,
        "reason": reason,
        "filter_version": FILTER_VERSION,
        "extracted_file": extracted_file,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save(manifest)


def is_rejected_stale(entry: dict) -> bool:
    """A rejection recorded under an older filter prompt version is worth re-checking."""
    return entry.get("status") == "rejected" and entry.get("filter_version") != FILTER_VERSION
