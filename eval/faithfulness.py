from typing import Dict, List


def faithfulness_score(claim: str, excerpts: List[str]) -> float:
    """Fraction of claim tokens that appear in retrieved context (Ragas-style)."""
    claim_tokens = set(_tok(claim))
    if not claim_tokens:
        return 0.0
    context = set(_tok(" ".join(excerpts)))
    if not context:
        return 0.0
    return len(claim_tokens & context) / len(claim_tokens)


def answer_relevance(rationale: str, mandate: str) -> float:
    """Does the proposal language align with the risk mandate?"""
    r = set(_tok(rationale))
    m = set(_tok(mandate))
    if not r or not m:
        return 0.0
    return len(r & m) / len(m)


def evaluate_research(brief_summary: str, excerpts: List[str], mandate: str) -> Dict[str, float]:
    return {
        "faithfulness": faithfulness_score(brief_summary, excerpts),
        "answer_relevance": answer_relevance(brief_summary, mandate),
    }


def _tok(text: str) -> List[str]:
    return [p for p in (text or "").lower().replace(",", " ").split() if len(p) > 2]
