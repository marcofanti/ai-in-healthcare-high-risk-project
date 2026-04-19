import os

from pydantic import BaseModel, Field

from utils.llm_client import llm


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
