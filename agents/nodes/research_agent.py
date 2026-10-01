from typing import List

from agents.state import AgentState, Citation, ResearchBrief
from governance.guardrails import require_grounding, screen_text
from observability.tracing import TRACER
from rag.reranker import CrossEncoderReranker
from rag.vector_store import HybridRetriever

_retriever = HybridRetriever()
_reranker = CrossEncoderReranker()


def research_node(state: AgentState) -> AgentState:
    with TRACER.start_as_current_span("research_node", symbol=state.symbol):
        query = (
            f"{state.symbol} 10-K 10-Q risk factors liquidity capital credit "
            "earnings disclosure"
        )
        hits = _retriever.search(query, symbol=state.symbol, top_k=8)
        ranked = _reranker.rerank(query, hits, top_k=3)
        citations: List[Citation] = [
            Citation(
                chunk_id=doc["chunk_id"],
                source_uri=doc["source_uri"],
                form_type=doc.get("form_type", "10-K"),
                excerpt=(doc.get("text") or "")[:400],
                score=float(doc.get("rerank_score") or doc.get("score") or 0),
            )
            for doc in ranked
        ]
        excerpts = [c.excerpt for c in citations]
        if not citations:
            summary = "No grounded filings retrieved; research node refuses to speculate."
            grounded = False
            sentiment = 0.0
        else:
            summary = " ".join(excerpts)[:800]
            grounded = require_grounding(summary, excerpts, min_overlap=3)
            lowered = summary.lower()
            sentiment = 0.0
            if any(w in lowered for w in ("growth", "adequate", "stable")):
                sentiment += 0.15
            if any(w in lowered for w in ("risk", "volatility", "npa", "shock")):
                sentiment -= 0.1
        ok, flags = screen_text(summary)
        state.research = ResearchBrief(
            summary=summary,
            sentiment=max(-1.0, min(1.0, sentiment)),
            citations=citations,
            grounded=grounded and ok,
            injection_flagged=not ok,
        )
        state.append_trace(
            "research",
            "Retrieved filings and news; extractive summary only.",
            grounded=state.research.grounded,
            citations=[c.chunk_id for c in citations],
            guardrail_flags=flags,
        )
        if not ok:
            state.reject_reason = "research_guardrail:" + ",".join(flags)
        return state
