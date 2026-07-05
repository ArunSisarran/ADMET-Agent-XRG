# ADMET-Agent-XRG

Builds a machine-learning knowledge base about ADMET (Absorption, Distribution,
Metabolism, Excretion, Toxicity) property prediction by extracting structured
data from research papers with Gemini, either supplied manually or found
automatically from free/open-access literature sources.

## Setup

```
pip install -r requirements.txt
```

Create a `.env` in the repo root with:

```
GOOGLE_API_KEY=...
CONTACT_EMAIL=you@example.com   # required for the Unpaywall PDF fallback
```

## Usage

Extract a single paper you already have:

```
python admet_agent/knowledge_extractor.py path/to/paper.pdf
```

Search free sources (Europe PMC, Semantic Scholar, arXiv, chemRxiv) for new
ADMET papers, filter for relevance, and extract them automatically:

```
python admet_agent/discover.py --dry-run   # preview what would be found/filtered
python admet_agent/discover.py --max-papers 5
```

See `docs/PAPER_DISCOVERY_PLAN.md` for the full design rationale.

## Project structure

```
admet_agent/    pipeline code (extraction, discovery, sources, manifest)
config/         seed_topics.json — editable list of search topics
docs/           design docs
examples/       standalone demo scripts, not part of the pipeline
data/           runtime output — manifest, extracted JSON, downloaded PDFs
```
