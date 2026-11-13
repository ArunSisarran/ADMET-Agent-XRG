# ADMET-Agent-XRG

An agent that reads ADMET research papers and turns them into a structured,
machine-readable knowledge base.

ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction is
a crowded field: every paper uses its own datasets, splits, representations,
models and metrics, and the useful details are scattered across body text,
tables and figures. This project finds relevant open-access papers on its own,
downloads them legally, and uses Gemini to extract the same fixed set of fields
from each one, so papers can be compared, filtered and fed to downstream
planning agents.

## What it extracts

Every paper becomes one JSON file with the same schema:

| Field | What it holds |
|---|---|
| `title`, `authors`, `year` | Bibliographic info |
| `admet_endpoints_covered` | Endpoints the paper actually modeled or measured (canonical TDC names, e.g. `caco2_wang`, `herg`), not ones it only mentions in passing |
| `datasets_used` | Name, source, split strategy, molecule count |
| `external_data_sources` | Databases used in the study (ChEMBL, PubChem, ...) |
| `molecular_representations` | ECFP4, MACCS, graphs, SMILES embeddings, ... |
| `model_architectures`, `multitask_combinations`, `hyperparameters` | How the models were built and trained |
| `benchmark_results` | `{endpoint, metric, value, dataset, baseline_comparison}` per reported result |
| `key_findings`, `negative_findings` | Main results, plus limitations, leakage risks and overfitting warnings |
| `competition_context` | e.g. "TDC ADMET Benchmark" |
| `figure_notes` | Figure captions carrying methodological detail |
| `recommended_for` | Whether the paper helps a data planner, a model planner, or both |

The extractor is told to record only what the paper states. Values it cannot
find stay `null` rather than being filled in from general ADMET/TDC knowledge.

## How it works

```
config/seed_topics.json        search topics (hERG, Caco-2, CYP450, BBB, ...)
        │
        ▼
sources.py                     search Europe PMC, Semantic Scholar, arXiv, chemRxiv (no API keys)
        │
        ▼
candidates.py + manifest.py    dedupe by DOI / title hash, skip papers already seen
        │
        ▼
relevance_filter.py            cheap Gemini check on title + abstract
        │
        ▼
pdf_resolver.py                open-access PDF from the source, else Unpaywall by DOI
        │
        ▼
knowledge_extractor.py         text + tables (pdfplumber) and figure pages (PyMuPDF)
                               → two Gemini calls in parallel → merge → normalize endpoint names
        │
        ▼
data/discovered_extractions/   one <key>_extracted.json per paper
data/knowledge_base_index.json index of everything extracted
data/processed_papers.json     manifest of every paper looked at, with status
```

Things worth knowing:

- **Open access only.** PDFs come from the source's own link or Unpaywall.
  Publisher pages are never scraped; papers without a free PDF are skipped and
  logged as `skipped_no_oa_pdf`.
- **Text and figures are read separately.** One Gemini call reads text and
  tables. A second, vision-based call reads only the figure pages and records
  only labeled values, never numbers estimated from plot positions.
- **Normalization.** Endpoint names are mapped to canonical TDC names and
  deduped across spelling variants (`Half-Life` / `half_life` / `Half Life`).
  A missing metric is backfilled from TDC convention, but a metric the paper
  reported is never overwritten.
- **Resumable.** The manifest is saved after every paper, so a crashed run
  picks up where it stopped and never re-downloads or re-extracts a paper.
- **Free-tier friendly.** Gemini calls are paced and retried on HTTP 429.
- **Known bias.** Using only free sources favors preprints and open-access
  journals over paywalled ones. Each record is tagged `preprint` or
  `published` so you can filter by it later.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
GOOGLE_API_KEY=...              # Gemini API key
CONTACT_EMAIL=you@example.com   # required by Unpaywall's terms for PDF lookups
```

LangSmith tracing environment variables are optional. If you set them, extraction runs are traced.

## Usage

Run every command from the repo root (the modules use bare imports).

**Find and extract papers automatically:**

```bash
python admet_agent/discover.py --dry-run          # search + relevance filter only, nothing downloaded
python admet_agent/discover.py --max-papers 5     # download and extract up to 5 new papers
python admet_agent/discover.py --topics "hERG inhibition,CYP450 inhibition"
python admet_agent/discover.py --sources europepmc,arxiv --max-results-per-source 5
```

To change what it searches for, edit `config/seed_topics.json`.

**Extract a paper you already have:**

```bash
python admet_agent/knowledge_extractor.py path/to/paper.pdf
python admet_agent/knowledge_extractor.py paper.pdf --output out.json --pages 1-8
```

## Project structure

```
admet_agent/
  discover.py             discovery CLI: search → filter → download → extract
  knowledge_extractor.py  core PDF → structured JSON extraction (also a standalone CLI)
  sources.py              Europe PMC, Semantic Scholar, arXiv, chemRxiv connectors
  candidates.py           Candidate dataclass and dedupe keys
  relevance_filter.py     title/abstract relevance check
  pdf_resolver.py         finds a legal open-access PDF
  manifest.py             record of every paper processed
  gemini_backoff.py       retry/backoff for Gemini rate limits
config/seed_topics.json   search topics
docs/                     design notes (see PAPER_DISCOVERY_PLAN.md for the rationale)
data/                     output: manifest, index, extractions, downloaded PDFs (gitignored)
```

## License

Apache 2.0. See [LICENSE](LICENSE).
