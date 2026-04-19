import os

from pydantic import BaseModel, Field

from utils.llm_client import llm
from utils.model_capabilities import (
    MODEL_CAPABILITIES,
    capability_catalog_markdown,
)


class QueryValidationResult(BaseModel):
    """Verdict on whether the user's selected ensemble can answer the query."""

    is_valid: bool = Field(
        description=(
            "True if at least one of the selected models is reasonably capable "
            "of answering the query given the imaging modality; false only when "
            "every selected model is a poor fit."
        )
    )
    reasoning: str = Field(
        description=(
            "One or two sentences explaining the verdict in clinical terms, "
            "referring to specific models by name."
        )
    )
    incompatible_models: list[str] = Field(
        default_factory=list,
        description=(
            "Selected models that are unsuitable for this query/modality. "
            "Can be non-empty even when is_valid is true (as warnings)."
        ),
    )
    recommended_models: list[str] = Field(
        default_factory=list,
        description=(
            "Models from the full available pool that are best suited for this "
            "query. Provide 1-3 names when is_valid is false; can also provide "
            "better alternatives when is_valid is true."
        ),
    )


class _ClinicalQuestions(BaseModel):
    """A list of clinically focused analysis questions about a medical image."""

    questions: list[str] = Field(
        description="Distinct, clinically specific questions. One question per string."
    )


class _ImprovedPrompt(BaseModel):
    """A single rewritten clinical analysis question."""

    improved_question: str = Field(
        description="The rewritten question, on a single line, no quotes or markers."
    )


def _file_hint(file_path: str) -> str:
    base = os.path.basename(file_path)
    stem, _ = os.path.splitext(base)
    stem, _ = os.path.splitext(stem)
    return stem


def _modality_context(modality: str) -> str:
    if modality and modality.strip().lower() != "unknown":
        return f"Imaging modality: {modality}."
    return "Imaging modality is not known with confidence; rely on the file name hints."


def generate_clinical_questions(file_path: str, modality: str, n: int = 10) -> list[str]:
    """
    Ask the LLM to propose `n` clinically focused analysis questions grounded
    in the file name and modality. Uses Gemini structured output so the
    response arrives as a parsed Pydantic object — no text parsing.
    """
    hint = _file_hint(file_path)
    mod_ctx = _modality_context(modality)

    prompt = f"""You are a clinical AI assistant helping a radiologist choose a focused analysis question for a medical image.

Propose exactly {n} distinct clinical questions that could reasonably be asked about this image. Requirements:
- Each question is self-contained and clinically specific (mention anatomy, pathology, or finding type where possible).
- Prefer directive phrasing: "Analyze...", "Identify...", "Describe...", "Assess...", "Segment...", "Report...".
- Avoid yes/no phrasing and avoid duplicating questions.
- Ground the questions in the file-name hints and modality when relevant.

File name: {hint}
{mod_ctx}"""

    structured_llm = llm.with_structured_output(_ClinicalQuestions, method="json_schema")
    result: _ClinicalQuestions = structured_llm.invoke(prompt)
    return result.questions[:n]


def improve_clinical_prompt(draft: str, file_path: str, modality: str) -> str:
    """
    Rewrite the user's draft question to be clearer, more clinically specific,
    and better scoped for an AI medical imaging ensemble.
    """
    hint = _file_hint(file_path)
    mod_ctx = _modality_context(modality)

    prompt = f"""You are a clinical AI assistant. Rewrite the user's analysis question to be clearer, more clinically specific, and well scoped for an AI medical imaging ensemble. Preserve the user's intent.

File name: {hint}
{mod_ctx}

User's draft question:
{draft}"""

    structured_llm = llm.with_structured_output(_ImprovedPrompt, method="json_schema")
    result: _ImprovedPrompt = structured_llm.invoke(prompt)
    improved = result.improved_question.strip()
    return improved if improved else draft


def validate_query_compatibility(
    query: str,
    selected_models: list[str],
    modality: str,
    file_path: str,
) -> QueryValidationResult:
    """
    Judge whether the user's selected ensemble is capable of answering the
    query for the detected modality. Uses the model capability catalog as
    ground truth. Returns a structured verdict with reasoning and
    recommendations.
    """
    hint = _file_hint(file_path)
    mod_ctx = _modality_context(modality)
    all_models = list(MODEL_CAPABILITIES.keys())
    catalog_md = capability_catalog_markdown(all_models)
    selected_md = capability_catalog_markdown(selected_models)

    prompt = f"""You are a senior clinical AI selector. Your job is to decide whether the user's currently selected ensemble of models is capable of answering their query for the detected imaging modality.

DECISION RULE:
- is_valid = true when AT LEAST ONE selected model is reasonably capable of answering the query given the modality.
- is_valid = false ONLY when EVERY selected model is a poor fit (e.g. a brain-MRI query sent only to chest-X-ray and pathology models).
- When in doubt (query is generic), lean towards is_valid = true.

ALL AVAILABLE MODELS AND CAPABILITIES:
{catalog_md}

USER'S CURRENT SELECTION:
{selected_md}

FILE NAME HINT: {hint}
{mod_ctx}
USER'S QUERY: {query}

Produce a verdict. When is_valid=false, recommended_models must be drawn from the AVAILABLE pool above and must be reasonable for the query/modality. When is_valid=true, incompatible_models may still list any selected model that is a poor fit as a soft warning."""

    structured_llm = llm.with_structured_output(QueryValidationResult, method="json_schema")
    result: QueryValidationResult = structured_llm.invoke(prompt)

    # Defensive filtering: drop any hallucinated names the LLM might invent
    available = set(all_models)
    result.incompatible_models = [m for m in result.incompatible_models if m in available]
    result.recommended_models = [m for m in result.recommended_models if m in available]
    return result
