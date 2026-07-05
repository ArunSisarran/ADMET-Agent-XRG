# Autonomous Paper Discovery — Design Plan

## Goal

`knowledge_extractor.py` currently requires a human to hand it a PDF. The next
step is to let the agent find and pull in ADMET-relevant papers **on its own**,
using only free APIs (no paid search/scraping services), and feed them through
the existing extraction pipeline (`extract_knowledge`) with no changes to that
pipeline's schema or logic.

Decisions locked in with the user:
- **Sources**: PubMed / Europe PMC, Semantic Scholar API, arXiv / chemRxiv /
  bioRxiv, CrossRef + Unpaywall (as an OA-link resolver/fallback).
- **Query strategy**: fixed seed topic list to start (no LLM-generated query
  expansion yet — keep it debuggable).
- **Relevance filtering**: cheap title/abstract pre-filter via Gemini before
  spending a full PDF download + extraction pass on a paper.
- **Trigger model**: on-demand CLI command (e.g. `--discover`), not a
  scheduled/cron job, for this phase.

Everything below fills in the details needed to hand this to an implementer.

---

## 1. Pipeline overview

```
[seed topics] -> [search each source API] -> [merge + dedupe candidates]
     -> [title/abstract relevance filter (Gemini, cheap)]
     -> [resolve open-access PDF URL]
     -> [download PDF]
     -> [existing extract_knowledge() pipeline]
     -> [save JSON result + update seen-papers manifest]
```

Only the first four stages are new. Once a PDF is on disk, it hands off to the
current `extract_knowledge(pdf_path, page_range)` unchanged.

## 2. Seed topics

A small, hand-maintained list of ADMET-ML search terms, e.g.:

```
hERG inhibition prediction
Caco-2 permeability machine learning
CYP450 inhibition deep learning
blood-brain barrier permeability prediction
AMES mutagenicity prediction
DILI (drug-induced liver injury) prediction
oral bioavailability QSAR
plasma protein binding prediction
clearance prediction machine learning ADMET
solubility prediction graph neural network
```

Stored as a plain list (Python list or a small `seed_topics.json`) so it's easy
to edit without touching code. Each discovery run iterates over every topic
against every enabled source.

**Open question for you**: do you want this list versioned in the repo (so
changes show up in git history / PRs), or in a gitignored local config so
different environments can run different topic sets? Recommend versioned in
the repo since it's core behavior, not secrets/environment config.

## 3. Source APIs

All of these are free and either keyless or free-tier-with-signup. None
require a credit card.

| Source | Auth | What it's good for | Notes |
|---|---|---|---|
| **Europe PMC REST API** | none | Search + full-text OA papers, great pharma/biomed coverage | `https://www.ebi.ac.uk/europepmc/webservices/rest/search` — response includes `isOpenAccess` flag and, for OA hits, a direct full-text/PDF link |
| **PubMed E-utilities (NCBI)** | none (API key optional, raises rate limit) | Broadest biomed metadata search | `esearch` + `esummary`/`efetch`; abstracts only — still need Europe PMC/Unpaywall to get the PDF |
| **Semantic Scholar API** | none for low volume; free API key recommended for higher rate limit | Search + abstracts + citation graph (find papers citing/cited-by a known good paper) | `https://api.semanticscholar.org/graph/v1/paper/search` |
| **arXiv API** | none | ML methodology papers (q-bio.BM, cs.LG) that use ADMET datasets | Simple XML/Atom API |
| **chemRxiv API** | none | Chemistry preprints, sometimes ADMET-adjacent | REST API via Figshare-based platform |
| **bioRxiv API** | none | Biology preprints | REST API, keyless |
| **CrossRef API** | none | DOI metadata resolution, backfilling missing metadata | `https://api.crossref.org/works` |
| **Unpaywall API** | none (requires an email param, not a key) | Given a DOI, returns a legal OA PDF link if one exists | `https://api.unpaywall.org/v2/{doi}?email=you@example.com` |

**Design point**: not every paper found via search has a legal free PDF. The
pipeline must gracefully **skip** (and log) any paper where no OA full text
can be resolved via Europe PMC, arXiv/bioRxiv/chemRxiv direct PDF, or
Unpaywall — never attempt to scrape a paywalled publisher page.

**Coverage bias — accept this going in**: restricting to free/OA sources is
not just "fewer papers," it's a non-random sample. A lot of strong ADMET-ML
work sits behind paywalls (J. Chem. Inf. Model., J. Med. Chem., Drug Metab.
Dispos.), so this pipeline will systematically favor arXiv/bioRxiv/chemRxiv
preprints and OA-friendly journals (PLOS, Scientific Reports, MDPI). That
skew will carry into anything trained on the resulting knowledge base — worth
remembering when interpreting downstream ML results, not something to try to
engineer away in this phase.

**Preprint vs. published version**: the same paper can appear as a preprint
and later as a peer-reviewed publication with different DOIs and slightly
different numbers. Title/DOI-based dedupe (see below) will not reliably catch
this, so the same underlying study could get double-counted with two
different benchmark values in the knowledge base. Tag every candidate with a
`source_type` field (`preprint` | `published`) at discovery time so this is
at least visible/attributable later, even though a full fuzzy-dedupe solution
is out of scope for this phase.

## 4. Candidate merging & deduplication

Each source returns candidates with (at minimum): title, authors, year, DOI
(if available), abstract, and a link. Because the same paper can surface from
multiple sources/topics:

- Normalize on **DOI** when present; fall back to a normalized-title match
  (lowercase, strip punctuation/whitespace) when DOI is missing (common for
  preprints).
- Maintain a persistent **manifest** (`processed_papers.json` or a small
  SQLite file) recording every DOI/title-hash already processed — either
  successfully extracted, skipped (no OA PDF), skipped (text extraction
  failed), or rejected (irrelevant). This is what prevents
  re-downloading/re-extracting the same paper on every run and gives you a
  visible audit trail of what the agent has looked at.
- **Manual single-PDF runs should register too.** Today's single-paper CLI
  path doesn't touch any manifest. Once discovery exists, have the
  single-paper path compute the same DOI/title key and write an entry, so
  discovery doesn't re-find and re-process a paper you already ran by hand.
- **Write manifest entries incrementally**, one per paper as it's decided —
  not only in a single batch write at the end of a run. If a discovery run
  crashes partway through (network error, rate limit, etc.), you don't want
  to lose the filtering/extraction work already done and re-spend API calls
  re-scoring the same abstracts on the next run.

**Open question for you**: JSON manifest file vs. SQLite? JSON is simpler and
human-readable (fits a small-scale personal project); SQLite scales better if
this grows to thousands of papers and you want to query it. Recommend
starting with JSON given current scale, migrate later if needed.

## 5. Relevance pre-filter

Before downloading any PDF:

1. Take title + abstract from the search result.
2. Send a short Gemini prompt: "Is this paper relevant to ADMET property
   prediction via machine learning? Answer with a JSON `{"relevant": bool,
   "reason": "..."}`" — deliberately not the full extraction schema, just a
   cheap yes/no gate.
3. Only papers marked relevant proceed to PDF resolution + download +ull
   `extract_knowledge` pipeline.

This keeps Gemini free-tier usage focused on papers worth the larger
extraction call, and keeps junk out of the knowledge base.

**Bias the filter toward inclusion.** A false positive just costs one wasted
extraction call. A false negative is silent and permanent — a borderline but
genuinely relevant paper (e.g. a general molecular-property model that
happens to include one ADMET endpoint among many tasks) gets marked
`rejected` in the manifest and is never looked at again. When in doubt, the
prompt should lean toward "relevant." Also store a `filter_version` alongside
each rejection in the manifest, so if the filter prompt improves later you
can deliberately re-run just the old rejections instead of re-scoring
everything or leaving stale false negatives buried forever.

## 6. PDF resolution & download order

For a candidate that passes the relevance filter, try in order until one
succeeds:
1. Europe PMC full-text/PDF link (if `isOpenAccess`).
2. Direct PDF link from arXiv/bioRxiv/chemRxiv (these are always OA).
3. Unpaywall lookup by DOI for a `best_oa_location.url_for_pdf`.
4. If none succeed: log as `skipped_no_oa_pdf` in the manifest and move on.

Downloaded PDFs can go to a working directory (e.g. `downloaded_papers/`,
gitignored) before being handed to `extract_knowledge`.

## 7. CLI design

Extend `knowledge_extractor.py`'s argparse (or add a new `discover.py` module
that imports `extract_knowledge`) with a discovery mode:

```
python knowledge_extractor.py --discover
python knowledge_extractor.py --discover --topics "hERG inhibition,CYP450 inhibition"
python knowledge_extractor.py --discover --max-papers 10
python knowledge_extractor.py --discover --sources europepmc,arxiv
python knowledge_extractor.py --discover --dry-run
```

`--dry-run` should run search + dedupe + relevance filtering and print which
candidates passed/failed and why, **without** downloading any PDF or calling
the full extraction pipeline. This makes it possible to sanity-check the
filter and source connectors before trusting the pipeline to run unattended
and spend real API quota.

Each run: pull candidates for all (or specified) seed topics from all (or
specified) sources, dedupe against the manifest, relevance-filter, resolve +
download PDFs, run extraction, save results the same way single-paper mode
does today (`<name>_extracted.json`), and update the manifest.

**Open question for you**: should discovery live inside `knowledge_extractor.py`
as a new mode, or as a separate `discover.py` script that calls into
`extract_knowledge()`? Recommend a separate module — keeps the single-paper
extractor simple and makes the discovery pipeline easier to test in
isolation, while reusing the extraction logic via import.

## 8. Rate limiting & politeness

- Respect each API's documented rate limits (Semantic Scholar and NCBI both
  publish these; add small delays / exponential backoff on 429s).
- Set a descriptive `User-Agent` / `email` param where APIs request one
  (Unpaywall requires this; NCBI recommends it).
- Cap papers processed per discovery run (`--max-papers`, default e.g. 20) so
  a single run can't blow through free-tier Gemini quota or take hours.
- **Gemini quota needs its own guardrail, separate from the source-API
  limits.** The relevance filter fires one Gemini call per *candidate*
  (before you even know if it's worth a full extraction), and the existing
  extraction pipeline already fires two Gemini calls per accepted paper (text
  + figures). Across several seed topics and several sources, the filtering
  stage alone can burn through free-tier quota before extraction even starts.
  Cap candidates fetched per topic per source (e.g. top 5 results) rather
  than paging exhaustively, and add explicit backoff/retry on Gemini 429s, not
  just on the source-API calls.
- **Don't run the expensive pipeline on unreadable PDFs.** The current
  `extract_knowledge` prints a warning when very little text is extracted
  (likely a scanned/image-only PDF) but still proceeds to spend both the text
  and figure Gemini calls on it. That's fine when a human triggered it
  deliberately on one known file; in an unattended discovery loop it will
  quietly waste calls on garbage. Short-circuit instead: if extracted text
  falls below the threshold, log `text_extraction_failed` in the manifest and
  skip the paper rather than running the full pipeline on it.

## 9. Output & storage

- Per-paper extracted JSON continues to land as it does today.
- Consider a lightweight aggregate index (e.g. `knowledge_base_index.json`)
  that lists every processed paper with its extracted-file path, so
  downstream ML tooling can enumerate the full knowledge base without
  re-parsing every JSON file. This is a small addition on top of the existing
  per-paper output, not a schema change.
- Tag each processed paper's `_meta` with `source_type` (`preprint` |
  `published`) and `source` (which API surfaced it), so downstream consumers
  can weight or filter by provenance (e.g. discount preprint-only benchmark
  numbers) — see the preprint/coverage-bias notes above.

## 10. Phased implementation

1. **Phase 1** — Source connectors: thin wrapper functions per API
   (`search_europepmc(topic)`, `search_semantic_scholar(topic)`, etc.) each
   returning a common candidate shape `{title, authors, year, doi, abstract,
   pdf_url_hint, source}`.
2. **Phase 2** — Manifest/dedupe layer + relevance filter.
3. **Phase 3** — OA PDF resolution + download.
4. **Phase 4** — Wire into `extract_knowledge`, add `--discover` CLI mode,
   add aggregate index output.
5. **Phase 5** (later, out of scope for now) — adaptive/LLM-generated query
   expansion, scheduled runs.

## 11. Open questions to resolve before/while building

- Seed topic list: versioned in repo vs. local config? (recommend: repo)
- Manifest format: JSON vs SQLite? (recommend: JSON for now)
- Module layout: extend `knowledge_extractor.py` vs new `discover.py`?
  (recommend: new module, import extraction logic)
- Do you want a Semantic Scholar API key (free signup, higher rate limit) or
  stay keyless and accept the lower anonymous rate limit to start?
- Default `--max-papers` per run — how many papers per invocation feels right
  given your Gemini free-tier quota?
