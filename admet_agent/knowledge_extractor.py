"""
Usage (run from the repo root):
    python admet_agent/knowledge_extractor.py paper.pdf
    python admet_agent/knowledge_extractor.py paper.pdf --output results.json
    python admet_agent/knowledge_extractor.py paper.pdf --pages 1-5
"""

import asyncio
import json
import argparse
import sys
import base64
from pathlib import Path

import fitz
import pdfplumber
from dotenv import load_dotenv
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from candidates import Candidate, dedupe_key
from gemini_backoff import ainvoke_with_backoff
import manifest as manifest_store

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

MIN_TEXT_CHARS = 100

SYSTEM_PROMPT = """
You are a scientific information extraction assistant specialised in
computational chemistry and ADMET (Absorption, Distribution, Metabolism,
Excretion, Toxicity) prediction research.

Your job is to read text and figures extracted from a research paper and return a
single JSON object — no markdown fences, no preamble, just raw JSON.

Return EXACTLY this schema (use null for missing fields, [] for empty lists):

{
  "title": "string",
  "authors": ["string"],
  "year": "string or null",
  "admet_endpoints_covered": ["list EVERY endpoint mentioned anywhere in the paper, e.g. Caco-2, hERG, BBB, logD, CYP3A4, AMES, DILI, VDss, ..."],
  "datasets_used": [
    {
      "name": "string",
      "source": "string or null  (e.g. TDC, ChEMBL, PubChem)",
      "split_strategy": "string or null  (e.g. scaffold, random)",
      "size": "integer or null  (number of molecules/compounds — NOT the number of datasets; use null if molecule count is not stated)"
    }
  ],
  "external_data_sources": ["string — databases or datasets directly used in THIS study for training, validation, evaluation, OR as a reference/inference set. Do NOT include sources merely cited in background, introduction, or related work."],
  "molecular_representations": [
    {
      "type": "string  (e.g. ECFP4, MACCS, graph, 3D, Uni-Mol)",
      "details": "string or null  (e.g. radius=2, bits=1024)"
    }
  ],
  "model_architectures": ["string  (e.g. GNN+FP fusion, Transformer, RF)"],
  "multitask_combinations": ["string  (e.g. hERG + Nav1.5 co-trained)"],
  "hyperparameters": {
    "learning_rate": "string or null",
    "batch_size": "string or null",
    "epochs": "string or null",
    "dropout": "string or null",
    "other": {}
  },
  "benchmark_results": [
    {
      "endpoint": "string",
      "metric": "string  (e.g. AUC-ROC, RMSE, MAE, R2)",
      "value": "number or null — use the PRIMARY reported value (e.g. TDC-reported/submitted score). If the paper also gives a reproduced value, put it in baseline_comparison, not here.",
      "dataset": "string or null",
      "baseline_comparison": "string or null — include model name, rank, reproduced value, or any comparison context. If the value is null because results are in a supplementary table or external source, write e.g. 'ADMET-AI TDC Leaderboard result; see Supplementary Table 1' rather than leaving this null."
    }
  ],
  "key_findings": ["string  (1-2 sentence bullet points)"],
  "negative_findings": ["string — include ALL of: limitations, failure cases, things that did NOT work, caveats about methodology, data leakage risks, overfitting warnings, split strategy concerns, or any result the authors flag as problematic"],
  "competition_context": "string or null  (e.g. TDC ADMET Benchmark 2023)",
  "recommended_for": ["data_planner", "model_planner"],
  "external_links_detected": ["list any URLs or DOIs to supplementary tables or external data that could not be extracted from the PDF"]
}

IMPORTANT:
- Extract only what is explicitly stated in the text or figures. Do not hallucinate.
- For admet_endpoints_covered, only include names that appear VERBATIM in the paper text or figure labels. Prefer the canonical TDC dataset name (e.g. "caco2_wang", "herg", "half_life_obach") over abbreviations. Do NOT include general biological concepts (e.g. "toxicity", "oral bioavailability", "logD"), do NOT paraphrase endpoint names, and do NOT invent or conflate names that do not appear exactly in the source. Do NOT include endpoints the paper explicitly states were excluded, not used, or treated as redundant.
- For benchmark_results, capture every reported number you can find including from figures and charts.
- For negative_findings, actively look for: limitations sections, caveats, scaffold leakage warnings, overfitting risks, dataset bias mentions, failure modes, and any result described as worse or unexpected.
- For hyperparameters.other, include training configuration details like number of splits, ensemble size, or random seeds even if LR/batch/dropout are not reported.
- recommended_for: include "data_planner" if the paper has dataset/external-data insights; include "model_planner" if it has architecture/representation insights.
"""

FIGURE_PROMPT = """
The following images are pages from a research paper containing figures, charts, and tables.
Extract all numerical results, endpoint names, and model performance data visible in these figures.
Return a JSON object with this schema — no markdown fences, just raw JSON:

{
  "endpoints_from_figures": ["every ADMET prediction endpoint name visible in any figure or table"],
  "benchmark_results_from_figures": [
    {
      "endpoint": "string",
      "metric": "string",
      "value": "number or null",
      "dataset": "string or null",
      "notes": "string or null"
    }
  ],
  "figure_notes": ["verbatim or close paraphrase of figure captions that contain important methodological details. Do NOT rephrase or reinterpret — quote the caption as written."]
}

TDC STANDARD METRICS — use these unless the figure caption explicitly states otherwise:
- MAE: Caco2, Lipophilicity, Solubility, PPBR, LD50
- Spearman: VDss, Half_Life, Clearance_Hepatocyte, Clearance_Microsome
- AUROC: hERG, AMES, DILI, BBB, HIA, Bioavailability, Pgp, CYP3A4_Substrate
- AUPRC: CYP2C9_Veith, CYP2D6_Veith, CYP3A4_Veith, CYP1A2_Veith, CYP2C19_Veith, CYP2C9_Substrate, CYP2D6_Substrate, CYP2C19_Substrate, CYP1A2_Substrate

STRICT RULES:
- endpoints_from_figures: only include molecular property or toxicity endpoint names such as
  hERG, Caco-2, BBB, AMES, DILI, CYP3A4, VDss, logD, solubility, clearance, half-life, etc.
  Use the exact name as it appears in the figure (e.g. "Bioavailability_Ma", "HIA_Hou").
- Do NOT include ATC drug category names such as "CARDIOVASCULAR SYSTEM", "NERVOUS SYSTEM",
  "ANTIBACTERIALS FOR SYSTEMIC USE", or any other all-caps therapeutic category labels.
- benchmark_results_from_figures: only include model performance metrics (AUROC, MAE, R2,
  AUPRC, Spearman) for ADMET prediction tasks. Do NOT include frequency counts, drug counts,
  or bar chart values from reference set distribution figures (e.g. DrugBank ATC code charts).
- If a figure shows only reference set demographics or drug category distributions, skip it entirely.
- For value, use the TDC-reported/submitted score. If a figure shows both a reported and a reproduced
  value, put the reported value in value and describe the reproduced value in notes.
- Set dataset to "TDC" for any result from TDC leaderboard figures.
- CRITICAL: Only extract values that are explicitly labeled with a precise number in the figure (e.g. in a table cell, axis tick, or data label). Do NOT estimate or read approximate positions off scatter plots or bar charts — if a value is not precisely labeled, set value to null rather than guessing. Round numbers like 0.5, 0.7, 0.9 are a red flag that you are guessing.
"""


def extract_text_from_pdf(pdf_path: str, page_range: tuple[int, int] | None = None) -> tuple[str, int]:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        start = (page_range[0] - 1) if page_range else 0
        end = page_range[1] if page_range else total_pages

        for i, page in enumerate(pdf.pages[start:end], start=start + 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"\n--- PAGE {i} ---\n{page_text}")

            tables = page.extract_tables()
            for table in tables:
                rows = ["\t".join(str(cell) for cell in row if cell) for row in table if row]
                if rows:
                    text_parts.append("\n[TABLE]\n" + "\n".join(rows))

    return "\n".join(text_parts), total_pages


def rasterize_figure_pages(pdf_path: str, page_range: tuple[int, int] | None = None, dpi: int = 150) -> list[str]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start = (page_range[0] - 1) if page_range else 0
    end = page_range[1] if page_range else total_pages

    figure_pages = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for i in range(start, min(end, total_pages)):
        page = doc[i]
        if not page.get_images(full=False):
            continue
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg")
        figure_pages.append(base64.b64encode(img_bytes).decode("utf-8"))

    doc.close()
    return figure_pages


async def extract_figures(figure_images: list[str]) -> dict:
    if not figure_images:
        return {}

    content = [{"type": "text", "text": FIGURE_PROMPT}]
    for img_b64 in figure_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    response = await ainvoke_with_backoff(model, [HumanMessage(content=content)])
    raw = response.content.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"figure_parse_error": raw[:500]}


KNOWN_CORRECTIONS = {
    "Bioavailability_Ha": "Bioavailability_Ma",
    "Pgp_Brocateelli": "Pgp_Broccatelli",
    "Pgp_Brocatelli": "Pgp_Broccatelli",
    "Pgp_Boccatelli": "Pgp_Broccatelli",
    "Skin_Reaction_Ma": "Skin_Reaction",
    "Skin_Sensitization": "Skin_Reaction",
}

# Maps paper abbreviations and known pdfplumber garbles → canonical TDC endpoint names (lowercase keys).
ENDPOINT_ALIASES = {
    "caco2": "caco2_wang",
    "caco-2": "caco2_wang",
    "lipo": "lipophilicity_astrazeneca",
    "lipophilicity": "lipophilicity_astrazeneca",
    "solu": "solubility_aqsoldb",
    "solubility": "solubility_aqsoldb",
    "hia": "hia_hou",
    "pgp": "pgp_broccatelli",
    "bbb": "bbb_martins",
    "ppbr": "ppbr_az",
    "vdss": "vdss_lombardo",
    "half": "half_life_obach",
    "half_life": "half_life_obach",
    "bio": "bioavailability_ma",
    "bioavailability": "bioavailability_ma",
    "ld50": "ld50_zhu",
    "ld50_zh": "ld50_zhu",
    "clearance_hepa": "clearance_hepatocyte_az",
    "clearence_hepa": "clearance_hepatocyte_az",
    "clearenc_e_hepa": "clearance_hepatocyte_az",
    "clearance_micro": "clearance_microsome_az",
    "clearence_micro": "clearance_microsome_az",
    "clearenc_e_micro": "clearance_microsome_az",
    "cyp2d6_sub": "cyp2d6_substrate_carbonmangels",
    "cyp3a4_sub": "cyp3a4_substrate_carbonmangels",
    "cyp2c9_sub": "cyp2c9_substrate_carbonmangels",
    "cyp2c19_sub": "cyp2c19_substrate_carbonmangels",
    "cyp1a2_sub": "cyp1a2_substrate_carbonmangels",
}

# TDC standard metrics per endpoint (lowercase keys). Used to correct figure extraction errors.
KNOWN_ENDPOINT_METRICS = {
    "caco2_wang": "MAE",
    "lipophilicity_astrazeneca": "MAE",
    "solubility_aqsoldb": "MAE",
    "ppbr_az": "MAE",
    "ld50_zhu": "MAE",
    "vdss_lombardo": "Spearman",
    "half_life_obach": "Spearman",
    "clearance_hepatocyte_az": "Spearman",
    "clearance_microsome_az": "Spearman",
    "bioavailability_ma": "AUROC",
    "hia_hou": "AUROC",
    "pgp_broccatelli": "AUROC",
    "pgp_brocattelli": "AUROC",
    "bbb_martins": "AUROC",
    "herg": "AUROC",
    "ames": "AUROC",
    "dili": "AUROC",
    "cyp3a4_substrate_carbonmangels": "AUROC",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_veith": "AUPRC",
    "cyp1a2_veith": "AUPRC",
    "cyp2c19_veith": "AUPRC",
    "cyp2c9_substrate_carbonmangels": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUPRC",
    "cyp2c19_substrate_carbonmangels": "AUPRC",
    "cyp1a2_substrate_carbonmangels": "AUPRC",
}


def _is_atc_noise(name: str) -> bool:
    if name is None:
        return False
    stripped = name.strip()
    return stripped == stripped.upper() and len(stripped.split()) > 1


def merge_figure_data(result: dict, figure_data: dict) -> dict:
    if not figure_data or "figure_parse_error" in figure_data:
        return result
    if "parse_error" in result:
        return result

    existing_endpoints_lower = {e.lower() for e in result.get("admet_endpoints_covered", [])}
    for ep in figure_data.get("endpoints_from_figures", []):
        if _is_atc_noise(ep):
            continue
        ep = KNOWN_CORRECTIONS.get(ep, ep)
        ep = ENDPOINT_ALIASES.get(ep.lower(), ep)
        if ep.lower() not in existing_endpoints_lower:
            result["admet_endpoints_covered"].append(ep)
            existing_endpoints_lower.add(ep.lower())

    existing_benchmarks = {
        (b["endpoint"].lower(), b["metric"]) for b in result.get("benchmark_results", [])
    }
    for b in figure_data.get("benchmark_results_from_figures", []):
        endpoint = b.get("endpoint")
        metric = b.get("metric")
        if _is_atc_noise(endpoint) or metric == "Frequency":
            continue
        endpoint = KNOWN_CORRECTIONS.get(endpoint, endpoint)
        endpoint = ENDPOINT_ALIASES.get(endpoint.lower(), endpoint)
        key = (endpoint.lower(), metric)
        if key not in existing_benchmarks:
            result["benchmark_results"].append({
                "endpoint": endpoint,
                "metric": metric,
                "value": b.get("value"),
                "dataset": b.get("dataset"),
                "baseline_comparison": b.get("notes"),
            })
            existing_benchmarks.add(key)

    if figure_data.get("figure_notes"):
        result.setdefault("figure_notes", []).extend(figure_data["figure_notes"])

    return result


def normalize_result(result: dict) -> dict:
    # Resolve corrections and aliases, then deduplicate admet_endpoints_covered case-insensitively.
    seen_lower = set()
    deduped = []
    for ep in result.get("admet_endpoints_covered", []):
        ep = ep.strip()
        ep = KNOWN_CORRECTIONS.get(ep, ep)
        canonical = ENDPOINT_ALIASES.get(ep.lower(), ep)
        canonical_key = canonical.lower()
        if canonical_key not in seen_lower:
            seen_lower.add(canonical_key)
            deduped.append(canonical)
    result["admet_endpoints_covered"] = deduped

    # Resolve corrections and aliases, correct metrics, then deduplicate benchmark_results.
    seen_benchmarks = set()
    deduped_benchmarks = []
    for b in result.get("benchmark_results", []):
        ep_raw = (b.get("endpoint") or "").strip()
        ep_raw = KNOWN_CORRECTIONS.get(ep_raw, ep_raw)
        canonical = ENDPOINT_ALIASES.get(ep_raw.lower(), ep_raw)
        b["endpoint"] = canonical
        if not b.get("metric") and canonical.lower() in KNOWN_ENDPOINT_METRICS:
            b["metric"] = KNOWN_ENDPOINT_METRICS[canonical.lower()]
        key = (canonical.lower(), b.get("metric"), str(b.get("value")), str(b.get("baseline_comparison")))
        if key not in seen_benchmarks:
            seen_benchmarks.add(key)
            deduped_benchmarks.append(b)
    result["benchmark_results"] = deduped_benchmarks

    return result


@traceable(
    name="knowledge_extractor_run",
    tags=["gemini-2.5-flash", "knowledge-layer", "admet", "pdf-parsing"],
)
async def extract_knowledge(pdf_path: str, page_range: tuple[int, int] | None = None) -> dict:
    path = Path(pdf_path)

    raw_text, total_pages = extract_text_from_pdf(pdf_path, page_range)
    page_info = f"pages {page_range[0]}-{page_range[1]}" if page_range else f"all {total_pages} pages"
    char_count = len(raw_text)

    if char_count < MIN_TEXT_CHARS:
        print("WARNING: Very little text extracted — PDF is likely scanned/image-based. Skipping Gemini extraction.")
        return {
            "skipped": "text_extraction_failed",
            "_meta": {
                "source_file": path.name,
                "pages_processed": page_info,
                "total_pages": total_pages,
                "chars_extracted": char_count,
                "figure_pages_rasterized": 0,
                "model": "gemini-2.5-flash",
            },
        }

    MAX_CHARS = 80_000
    if char_count > MAX_CHARS:
        print(f"NOTE: Text truncated to {MAX_CHARS:,} chars to stay within token limits.")
        raw_text = raw_text[:MAX_CHARS] + "\n\n[... text truncated ...]"

    figure_images = rasterize_figure_pages(pdf_path, page_range)

    user_message = f"""
Below is the full text extracted from a research paper. Extract all relevant
ADMET information according to the schema and return only the JSON object.

PAPER TEXT:
{raw_text}
"""

    text_task = ainvoke_with_backoff(model, [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    figure_task = extract_figures(figure_images)

    text_response, figure_data = await asyncio.gather(text_task, figure_task)

    raw_output = text_response.content.strip()
    if raw_output.startswith("```"):
        lines = raw_output.split("\n")
        raw_output = "\n".join(lines[1:-1])

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError as e:
        result = {"parse_error": str(e), "raw_output": raw_output}

    result = merge_figure_data(result, figure_data)
    result = normalize_result(result)

    result["_meta"] = {
        "source_file": path.name,
        "pages_processed": page_info,
        "total_pages": total_pages,
        "chars_extracted": char_count,
        "figure_pages_rasterized": len(figure_images),
        "model": "gemini-2.5-flash",
    }

    return result


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADMET Knowledge Extractor — prototype for the Knowledge Layer"
    )
    parser.add_argument("pdf", help="Path to the research paper PDF")
    parser.add_argument("--output", "-o", help="Save full JSON to this file", default=None)
    parser.add_argument(
        "--pages", "-p",
        help="Page range to process, e.g. --pages 1-8  (default: all)",
        default=None,
    )
    args = parser.parse_args()

    page_range = None
    if args.pages:
        try:
            start, end = args.pages.split("-")
            page_range = (int(start), int(end))
        except ValueError:
            print("ERROR: --pages must be in format START-END, e.g. --pages 1-10")
            sys.exit(1)

    result = await extract_knowledge(args.pdf, page_range)

    output_path = args.output or Path(args.pdf).stem + "_extracted.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}\n")

    title = result.get("title") or Path(args.pdf).stem
    if result.get("skipped"):
        status = result["skipped"]
    elif "parse_error" in result:
        status = "error"
    else:
        status = "extracted"

    key = dedupe_key(Candidate(title=title, source="manual", source_type="unknown"))
    manifest_store.record(
        manifest_store.load(),
        key,
        status,
        title=title,
        source="manual",
        source_type="unknown",
        extracted_file=output_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
