"""Cheap title/abstract relevance gate, run before the expensive PDF download
and full extraction pipeline.

Biased toward inclusion: a false positive here only costs one wasted
extraction call later; a false negative silently and permanently buries a
paper in the manifest as "rejected".
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from gemini_backoff import ainvoke_with_backoff
from knowledge_extractor import model
from manifest import FILTER_VERSION

FILTER_SYSTEM_PROMPT = """
You are a fast relevance gate for a pipeline that builds a machine-learning
knowledge base about ADMET (Absorption, Distribution, Metabolism, Excretion,
Toxicity) property prediction.

Given a paper's title and abstract, decide if it is worth a full extraction
pass. Answer "relevant" if the paper is about ANY of: predicting or modeling
an ADMET-related property (e.g. permeability, solubility, clearance, protein
binding, CYP inhibition, hERG, mutagenicity, bioavailability, toxicity),
molecular representation learning applied to such properties, or benchmark
datasets/models for pharmacokinetic or toxicity endpoints — even if it is
only one part of a broader paper.

When uncertain, prefer "relevant" — a missed relevant paper is a worse
outcome than one wasted extraction pass on an irrelevant paper.

Return ONLY raw JSON, no markdown fences:
{"relevant": true or false, "reason": "one short sentence"}
"""


async def is_relevant(title: str, abstract: str | None) -> dict:
    abstract_text = abstract or "(no abstract available)"
    user_content = f"TITLE: {title}\n\nABSTRACT: {abstract_text}"

    response = await ainvoke_with_backoff(model, [
        SystemMessage(content=FILTER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Bias toward inclusion on parse failure too.
        result = {"relevant": True, "reason": f"filter_parse_error: {raw[:200]}"}

    result["filter_version"] = FILTER_VERSION
    return result
