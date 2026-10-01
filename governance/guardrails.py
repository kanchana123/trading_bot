import re
from typing import Iterable, List, Tuple

INJECTION_PATTERNS = (
    r"ignore (all|any|previous|prior) instructions",
    r"you are now",
    r"system prompt",
    r"disregard (the )?(risk|policy|compliance)",
    r"execute (the )?trade now",
    r"override (the )?guardrail",
    r"<\|.*?\|>",
)


def screen_text(text: str) -> Tuple[bool, List[str]]:
    """Heuristic semantic screen (Llama Guard / NeMo stand-in without a GPU)."""
    flags: List[str] = []
    blob = (text or "").lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, blob, re.IGNORECASE):
            flags.append(f"injection:{pattern}")
    if "placeorder" in blob.replace(" ", "") or "place_order" in blob:
        flags.append("execution_exfiltration")
    return (len(flags) == 0, flags)


def require_grounding(claim: str, excerpts: Iterable[str], min_overlap: int = 3) -> bool:
    """Reject rationale that cannot be found in retrieved filings/news."""
    tokens = _tokens(claim)
    if len(tokens) < min_overlap:
        return False
    corpus = " ".join(_tokens(" ".join(excerpts)))
    hits = sum(1 for tok in set(tokens) if tok in corpus)
    return hits >= min_overlap


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())
