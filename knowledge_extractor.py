"""
Usage:
    python knowledge_extractor.py paper.pdf
    python knowledge_extractor.py paper.pdf --output results.json
    python knowledge_extractor.py paper.pdf --pages 1-5
"""

import asyncio
import json
import argparse
import sys
import base64
import math
from pathlib import Path

import fitz
import pdfplumber
from dotenv import load_dotenv
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

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
      "size": "integer or null"
    }
  ],
  "external_data_sources": ["string"],
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
      "value": "number or null",
      "dataset": "string or null",
      "baseline_comparison": "string or null"
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
- For admet_endpoints_covered, list every single endpoint name you encounter anywhere — in tables, figures, supplementary sections, or body text.
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
  "figure_notes": ["any important methodological details visible only in figures/captions"]
}

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
    scoped_total = end - start

    cutoff = start + math.floor(scoped_total * 0.7)

    figure_pages = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for i in range(cutoff, min(end, total_pages)):
        page = doc[i]
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

    response = await model.ainvoke([HumanMessage(content=content)])
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
}

def _is_atc_noise(name: str) -> bool:
    if name is None:
        return False
    stripped = name.strip()
    return stripped == stripped.upper() and len(stripped.split()) > 1


def merge_figure_data(result: dict, figure_data: dict) -> dict:
    if not figure_data or "figure_parse_error" in figure_data:
        return result

    existing_endpoints = set(result.get("admet_endpoints_covered", []))
    for ep in figure_data.get("endpoints_from_figures", []):
        if _is_atc_noise(ep):
            continue
        ep = KNOWN_CORRECTIONS.get(ep, ep)
        if ep not in existing_endpoints:
            result["admet_endpoints_covered"].append(ep)
            existing_endpoints.add(ep)

    existing_benchmarks = {
        (b["endpoint"], b["metric"]) for b in result.get("benchmark_results", [])
    }
    for b in figure_data.get("benchmark_results_from_figures", []):
        endpoint = b.get("endpoint")
        metric = b.get("metric")
        if _is_atc_noise(endpoint) or metric == "Frequency":
            continue
        endpoint = KNOWN_CORRECTIONS.get(endpoint, endpoint)
        key = (endpoint, metric)
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


@traceable(
    name="knowledge_extractor_run",
    tags=["gemini-2.5-flash", "knowledge-layer", "admet", "pdf-parsing"],
)
async def extract_knowledge(pdf_path: str, page_range: tuple[int, int] | None = None) -> dict:
    path = Path(pdf_path)

    raw_text, total_pages = extract_text_from_pdf(pdf_path, page_range)
    page_info = f"pages {page_range[0]}-{page_range[1]}" if page_range else f"all {total_pages} pages"
    char_count = len(raw_text)

    if char_count < 100:
        print("WARNING: Very little text extracted — PDF may be scanned/image-based.")

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

    text_task = model.ainvoke([
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


if __name__ == "__main__":
    asyncio.run(main())
