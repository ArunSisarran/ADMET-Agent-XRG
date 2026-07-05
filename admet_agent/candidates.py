"""Candidate paper representation shared by every discovery source connector."""

import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class Candidate:
    title: str
    source: str
    source_type: str  # "preprint" | "published" | "unknown"
    authors: list[str] = field(default_factory=list)
    year: str | None = None
    doi: str | None = None
    abstract: str | None = None
    pdf_url: str | None = None
    landing_url: str | None = None


def normalize_title(title: str) -> str:
    stripped = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return re.sub(r"\s+", " ", stripped)


def dedupe_key(candidate: Candidate) -> str:
    """DOI when available, else a hash of the normalized title.

    Note: a preprint and its later published version will usually have
    different DOIs and can slip past this as two separate entries — see
    PAPER_DISCOVERY_PLAN.md for why that's an accepted limitation for now.
    """
    if candidate.doi:
        return f"doi:{candidate.doi.strip().lower()}"
    digest = hashlib.sha1(normalize_title(candidate.title).encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"
