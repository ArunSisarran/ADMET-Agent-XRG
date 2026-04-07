"""
Usage:
    python knowledge_extractor.py paper.pdf
    python knowledge_extractor.py paper.pdf --output results.json
    python knowledge_extractor.py paper.pdf --pages 1-5   # limit pages (cheaper)
"""

import asyncio
import json
import argparse
import sys
from pathlib import Path

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

Your job is to read text extracted from a research paper and return a
single JSON object — no markdown fences, no preamble, just raw JSON.

Return EXACTLY this schema (use null for missing fields, [] for empty lists):

{
  "title": "string",
  "authors": ["string"],
  "year": "string or null",
  "admet_endpoints_covered": ["e.g. Caco-2, hERG, BBB, logD, ..."],
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
  "key_findings": ["string  (1–2 sentence bullet points)"],
  "negative_findings": ["string  (what did NOT work)"],
  "competition_context": "string or null  (e.g. TDC ADMET Benchmark 2023)",
  "recommended_for": ["data_planner", "model_planner"]
}

IMPORTANT:
- Extract only what is explicitly stated in the text. Do not hallucinate.
- For benchmark_results, capture every reported number you can find.
- For key_findings and negative_findings, be concise but specific.
- recommended_for: include "data_planner" if the paper has dataset/external-data
  insights; include "model_planner" if it has architecture/representation insights.
"""

def extract_text_from_pdf(pdf_path: str, page_range: tuple[int, int] | None = None) -> tuple[str, int]:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        start = (page_range[0] - 1) if page_range else 0
        end   = page_range[1] if page_range else total_pages

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
        print("Very little text recognized")

    MAX_CHARS = 80_000
    if char_count > MAX_CHARS:
        print(f"Text truncated to {MAX_CHARS:,} chars to stay within token limits.")
        raw_text = raw_text[:MAX_CHARS] + "\n\n[... text truncated ...]"


    user_message = f"""
Below is the full text extracted from a research paper. Extract all relevant
ADMET information according to the schema and return only the JSON object.

PAPER TEXT:
{raw_text}
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = await model.ainvoke(messages)
    raw_output = response.content.strip()


    if raw_output.startswith("```"):
        lines = raw_output.split("\n")
        raw_output = "\n".join(lines[1:-1])

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError as e:
        result = {"parse_error": str(e), "raw_output": raw_output}

    result["_meta"] = {
        "source_file": path.name,
        "pages_processed": page_info,
        "total_pages": total_pages,
        "chars_extracted": char_count,
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
            print(f"ERROR: --pages must be in format START-END, e.g. --pages 1-10")
            sys.exit(1)

    result = await extract_knowledge(args.pdf, page_range)

    output_path = args.output or Path(args.pdf).stem + "_extracted.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full JSON saved to: {output_path}\n")

if __name__ == "__main__":
    asyncio.run(main())
