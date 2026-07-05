"""Resolves a legal open-access PDF URL for a candidate paper.

Tries, in order: the direct PDF link the source connector already surfaced
(Europe PMC / arXiv / chemRxiv return these directly when OA), then falls
back to Unpaywall by DOI. Never falls back to scraping a publisher landing
page — if no OA copy can be resolved this way, the paper is skipped.
"""

from candidates import Candidate
from sources import CONTACT_EMAIL, get_with_backoff


def resolve_pdf_url(candidate: Candidate) -> str | None:
    if candidate.pdf_url:
        return candidate.pdf_url

    if not candidate.doi:
        return None

    if not CONTACT_EMAIL:
        print("NOTE: set CONTACT_EMAIL in .env to enable Unpaywall lookups (required by their API terms).")
        return None

    resp = get_with_backoff(
        f"https://api.unpaywall.org/v2/{candidate.doi}",
        params={"email": CONTACT_EMAIL},
    )
    if resp is None or resp.status_code != 200:
        return None

    best = resp.json().get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")
