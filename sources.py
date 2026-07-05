"""Keyless connectors for free/open-access literature search APIs.

Each search_* function takes a topic string and a max result count and
returns a list[Candidate]. All of these are free and require no paid access;
see PAPER_DISCOVERY_PLAN.md for the rationale behind each source.
"""

import os
import time
import xml.etree.ElementTree as ET

import requests

from candidates import Candidate

CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
DEFAULT_HEADERS = {
    "User-Agent": f"ADMET-Agent-XRG-discovery/0.1 (contact: {CONTACT_EMAIL or 'not set - see README'})"
}

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def get_with_backoff(url: str, params: dict | None = None, timeout: int = 20, max_retries: int = 3):
    """GET with exponential backoff on request errors and HTTP 429s. Returns None on final failure."""
    delay = 1.0
    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=DEFAULT_HEADERS)
        except requests.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 429 and attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue

        return resp
    return resp


def search_europepmc(topic: str, max_results: int = 5) -> list[Candidate]:
    """Europe PMC's unified index covers PubMed/MEDLINE, PMC, and preprints
    (including bioRxiv/medRxiv), and surfaces direct OA PDF links when available."""
    resp = get_with_backoff(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": topic, "format": "json", "pageSize": max_results, "resultType": "core"},
    )
    if resp is None or resp.status_code != 200:
        return []

    candidates = []
    for item in resp.json().get("resultList", {}).get("result", []):
        title = (item.get("title") or "").strip()
        if not title:
            continue

        pdf_url = None
        for ft in (item.get("fullTextUrlList") or {}).get("fullTextUrl", []):
            if ft.get("documentStyle") == "pdf" and str(ft.get("availability", "")).lower().startswith("open"):
                pdf_url = ft.get("url")
                break

        source_type = "preprint" if item.get("source") == "PPR" else "published"
        landing_url = (
            f"https://europepmc.org/article/{item['source']}/{item['id']}"
            if item.get("source") and item.get("id")
            else None
        )

        candidates.append(Candidate(
            title=title,
            authors=[a.strip() for a in (item.get("authorString") or "").split(",") if a.strip()],
            year=item.get("pubYear"),
            doi=item.get("doi"),
            abstract=item.get("abstractText"),
            pdf_url=pdf_url,
            landing_url=landing_url,
            source="europepmc",
            source_type=source_type,
        ))
    return candidates


def search_semantic_scholar(topic: str, max_results: int = 5) -> list[Candidate]:
    """Anonymous/keyless tier — shares a low rate-limit pool, so keep max_results small."""
    resp = get_with_backoff(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": topic,
            "limit": max_results,
            "fields": "title,abstract,year,authors,externalIds,openAccessPdf,publicationTypes",
        },
    )
    if resp is None or resp.status_code != 200:
        return []

    candidates = []
    for item in resp.json().get("data", []):
        title = (item.get("title") or "").strip()
        if not title:
            continue

        oa_pdf = item.get("openAccessPdf") or {}
        pub_types = item.get("publicationTypes") or []
        # Heuristic, not authoritative: Semantic Scholar doesn't cleanly flag preprints.
        source_type = "preprint" if "Preprint" in pub_types else "published"

        candidates.append(Candidate(
            title=title,
            authors=[a.get("name") for a in item.get("authors", []) if a.get("name")],
            year=str(item["year"]) if item.get("year") else None,
            doi=(item.get("externalIds") or {}).get("DOI"),
            abstract=item.get("abstract"),
            pdf_url=oa_pdf.get("url"),
            landing_url=None,
            source="semantic_scholar",
            source_type=source_type,
        ))
    return candidates


def search_arxiv(topic: str, max_results: int = 5) -> list[Candidate]:
    resp = get_with_backoff(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{topic}", "start": 0, "max_results": max_results},
    )
    if resp is None or resp.status_code != 200:
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    candidates = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        title = " ".join(title.split())
        if not title:
            continue

        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        authors = [
            (a.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
            for a in entry.findall("atom:author", ATOM_NS)
        ]
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS) or ""

        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")

        candidates.append(Candidate(
            title=title,
            authors=[a for a in authors if a],
            year=published[:4] if published else None,
            doi=None,
            abstract=summary,
            pdf_url=pdf_url,
            landing_url=entry.findtext("atom:id", default=None, namespaces=ATOM_NS),
            source="arxiv",
            source_type="preprint",
        ))
    return candidates


def search_chemrxiv(topic: str, max_results: int = 5) -> list[Candidate]:
    """Best-effort connector for chemRxiv's public API (Cambridge Open Engage).

    This API's schema has shifted before without notice. If this silently
    returns nothing, check
    https://chemrxiv.org/engage/chemrxiv/public-api/v1/docs before assuming
    the topic just has no matches.
    """
    resp = get_with_backoff(
        "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items",
        params={"term": topic, "limit": max_results},
    )
    if resp is None or resp.status_code != 200:
        return []

    try:
        hits = resp.json().get("itemHits", [])
    except ValueError:
        return []

    candidates = []
    for hit in hits:
        item = hit.get("item", {})
        title = (item.get("title") or "").strip()
        if not title:
            continue

        authors = [
            " ".join(filter(None, [a.get("firstName"), a.get("lastName")])).strip()
            for a in item.get("authors", [])
        ]
        published = item.get("publishedDate") or ""
        asset = (item.get("asset") or {}).get("original") or {}

        candidates.append(Candidate(
            title=title,
            authors=[a for a in authors if a],
            year=published[:4] if published else None,
            doi=item.get("doi"),
            abstract=item.get("abstract"),
            pdf_url=asset.get("url"),
            landing_url=(
                f"https://chemrxiv.org/engage/chemrxiv/article-details/{item['id']}"
                if item.get("id") else None
            ),
            source="chemrxiv",
            source_type="preprint",
        ))
    return candidates


ALL_SOURCES = {
    "europepmc": search_europepmc,
    "semantic_scholar": search_semantic_scholar,
    "arxiv": search_arxiv,
    "chemrxiv": search_chemrxiv,
}
