"""Automated evaluation suite (faithfulness / mandate relevance)."""

from eval.faithfulness import evaluate_research
from rag.corpus import SEED_FILINGS


MANDATE = "portfolio concentration risk capital liquidity credit stop-loss policy"


def test_research_summary_is_faithful_to_filings():
    excerpts = [row["text"] for row in SEED_FILINGS if "ICICI" in row["symbol"]]
    summary = excerpts[0]
    scores = evaluate_research(summary, excerpts, MANDATE)
    assert scores["faithfulness"] >= 0.5


def test_fabricated_claim_fails_faithfulness():
    excerpts = [row["text"] for row in SEED_FILINGS]
    scores = evaluate_research(
        "Xylophone quantum banana teleportation yields 400x weekly returns.",
        excerpts,
        MANDATE,
    )
    assert scores["faithfulness"] < 0.2


def test_mandate_relevance_detects_risk_language():
    excerpts = [row["text"] for row in SEED_FILINGS if "ICICI" in row["symbol"]]
    scores = evaluate_research(excerpts[0], excerpts, MANDATE)
    assert scores["answer_relevance"] > 0
