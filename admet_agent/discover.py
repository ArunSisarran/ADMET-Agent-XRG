"""
Autonomous paper discovery for the ADMET knowledge base.

Searches free/open-access sources for papers matching a seed list of ADMET
topics, filters out irrelevant results cheaply (title/abstract only), resolves
a legal open-access PDF for anything that passes, and runs it through the
existing knowledge_extractor.extract_knowledge() pipeline. See
docs/PAPER_DISCOVERY_PLAN.md for the full design rationale.

Usage (run from the repo root):
    python admet_agent/discover.py
    python admet_agent/discover.py --topics "hERG inhibition,CYP450 inhibition"
    python admet_agent/discover.py --sources europepmc,arxiv
    python admet_agent/discover.py --max-papers 10
    python admet_agent/discover.py --dry-run
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

import manifest as manifest_store
from candidates import Candidate, dedupe_key
from knowledge_extractor import extract_knowledge
from pdf_resolver import resolve_pdf_url
from relevance_filter import is_relevant
from sources import ALL_SOURCES, DEFAULT_HEADERS

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_TOPICS_PATH = REPO_ROOT / "config" / "seed_topics.json"
DOWNLOAD_DIR = REPO_ROOT / "data" / "downloaded_papers"
EXTRACTED_DIR = REPO_ROOT / "data" / "discovered_extractions"
INDEX_PATH = REPO_ROOT / "data" / "knowledge_base_index.json"

SOURCE_DELAY_SECONDS = 0.5  # politeness delay between source-API calls
GEMINI_PACE_SECONDS = 13  # free tier is 5 req/min for gemini-2.5-flash (~12s/req); pace proactively rather than relying only on 429 backoff


def safe_filename(key: str) -> str:
    """DOIs routinely contain '/', which would otherwise be read as a path separator."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def load_seed_topics() -> list[str]:
    with open(SEED_TOPICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def download_pdf(url: str, dest_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30, headers=DEFAULT_HEADERS)
        resp.raise_for_status()
    except requests.RequestException:
        return False
    dest_path.write_bytes(resp.content)
    return True


def update_index(entry: dict) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index = []
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            index = json.load(f)
    index.append(entry)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def collect_candidates(topics: list[str], source_names: list[str], max_results_per_source: int, manifest: dict) -> dict:
    candidates_by_key: dict[str, Candidate] = {}

    for topic in topics:
        for source_name in source_names:
            search_fn = ALL_SOURCES[source_name]
            try:
                found = search_fn(topic, max_results_per_source)
            except Exception as e:
                print(f"WARNING: source '{source_name}' failed for topic '{topic}': {e}")
                continue

            for candidate in found:
                key = dedupe_key(candidate)
                if key in candidates_by_key:
                    continue
                existing = manifest.get(key)
                if existing and not manifest_store.is_rejected_stale(existing):
                    continue
                candidates_by_key[key] = candidate

            time.sleep(SOURCE_DELAY_SECONDS)

    return candidates_by_key


async def run_discovery(topics: list[str], source_names: list[str], max_results_per_source: int, max_papers: int, dry_run: bool) -> None:
    manifest = manifest_store.load()
    candidates_by_key = collect_candidates(topics, source_names, max_results_per_source, manifest)
    print(f"Found {len(candidates_by_key)} new candidate(s) after dedup against the manifest.\n")

    processed_count = 0
    for key, candidate in candidates_by_key.items():
        if processed_count >= max_papers:
            print(f"Reached --max-papers limit ({max_papers}); stopping.")
            break

        verdict = await is_relevant(candidate.title, candidate.abstract)
        await asyncio.sleep(GEMINI_PACE_SECONDS)
        if not verdict.get("relevant", True):
            print(f"REJECTED: {candidate.title!r} -- {verdict.get('reason')}")
            if not dry_run:
                manifest_store.record(
                    manifest, key, "rejected",
                    title=candidate.title, doi=candidate.doi,
                    source=candidate.source, source_type=candidate.source_type,
                    reason=verdict.get("reason"),
                )
            continue

        print(f"RELEVANT: {candidate.title!r} -- {verdict.get('reason')}")
        if dry_run:
            continue

        pdf_url = resolve_pdf_url(candidate)
        if not pdf_url:
            print(f"  SKIPPED (no open-access PDF found)")
            manifest_store.record(
                manifest, key, "skipped_no_oa_pdf",
                title=candidate.title, doi=candidate.doi,
                source=candidate.source, source_type=candidate.source_type,
            )
            continue

        DOWNLOAD_DIR.mkdir(exist_ok=True)
        dest = DOWNLOAD_DIR / f"{safe_filename(key)}.pdf"
        if not download_pdf(pdf_url, dest):
            print(f"  SKIPPED (PDF download failed)")
            manifest_store.record(
                manifest, key, "skipped_no_oa_pdf",
                title=candidate.title, doi=candidate.doi,
                source=candidate.source, source_type=candidate.source_type,
                reason="download_failed",
            )
            continue

        result = await extract_knowledge(str(dest))
        processed_count += 1

        if result.get("skipped") == "text_extraction_failed":
            print(f"  SKIPPED (text extraction failed, likely scanned PDF)")
            manifest_store.record(
                manifest, key, "skipped_text_extraction_failed",
                title=candidate.title, doi=candidate.doi,
                source=candidate.source, source_type=candidate.source_type,
            )
            continue

        result["_meta"]["source"] = candidate.source
        result["_meta"]["source_type"] = candidate.source_type
        result["_meta"]["landing_url"] = candidate.landing_url

        EXTRACTED_DIR.mkdir(exist_ok=True)
        extracted_path = EXTRACTED_DIR / f"{safe_filename(key)}_extracted.json"
        with open(extracted_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        status = "error" if "parse_error" in result else "extracted"
        manifest_store.record(
            manifest, key, status,
            title=candidate.title, doi=candidate.doi,
            source=candidate.source, source_type=candidate.source_type,
            extracted_file=str(extracted_path),
        )

        if status == "extracted":
            update_index({
                "key": key,
                "title": candidate.title,
                "doi": candidate.doi,
                "source": candidate.source,
                "source_type": candidate.source_type,
                "extracted_file": str(extracted_path),
            })
            print(f"  EXTRACTED -> {extracted_path}")
        else:
            print(f"  ERROR during extraction -> {extracted_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and extract ADMET papers from free/open-access sources."
    )
    parser.add_argument(
        "--topics",
        help="Comma-separated search topics (default: seed_topics.json)",
        default=None,
    )
    parser.add_argument(
        "--sources",
        help=f"Comma-separated sources to use: {','.join(ALL_SOURCES.keys())} (default: all)",
        default=None,
    )
    parser.add_argument(
        "--max-results-per-source", type=int, default=5,
        help="Max search results to pull per topic per source (default: 5)",
    )
    parser.add_argument(
        "--max-papers", type=int, default=10,
        help="Max papers to fully download+extract in this run (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search and relevance-filter only; do not download, extract, or update the manifest for accepted papers",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    topics = [t.strip() for t in args.topics.split(",")] if args.topics else load_seed_topics()
    source_names = [s.strip() for s in args.sources.split(",")] if args.sources else list(ALL_SOURCES.keys())

    unknown = set(source_names) - set(ALL_SOURCES.keys())
    if unknown:
        raise SystemExit(f"Unknown source(s): {', '.join(unknown)}. Valid: {', '.join(ALL_SOURCES.keys())}")

    await run_discovery(
        topics=topics,
        source_names=source_names,
        max_results_per_source=args.max_results_per_source,
        max_papers=args.max_papers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    asyncio.run(main())
