from agents.state import AgentState, CandidateSignal, OrderProposal, RouteDecision, Side
from governance.guardrails import require_grounding, screen_text
from observability.tracing import TRACER


def adjudicator_node(state: AgentState) -> AgentState:
    with TRACER.start_as_current_span("adjudicator_node", symbol=state.symbol):
        tradable = [
            s
            for s in state.signals
            if s.source != "QuantHold" and s.strength >= 0.45
        ]
        if not tradable and state.research.grounded and state.risk.passed and state.research.sentiment >= -0.15:
            qty = max(int(state.nav * state.max_name_weight / max(state.last_price, 1e-9)), 1)
            tradable = [
                CandidateSignal(
                    source="ResearchWatch",
                    side=Side.BUY,
                    symbol=state.symbol,
                    strength=0.35,
                    price=state.last_price,
                    quantity=qty,
                    reason="Grounded filings retrieved; no technical trigger. HITL-only exploratory size.",
                    indicators={"close": state.last_price},
                )
            ]
        if not tradable or not state.risk.passed or state.research.injection_flagged:
            state.proposal = None
            state.route = RouteDecision.REJECT
            state.reject_reason = state.reject_reason or "adjudicator_no_consensus"
            state.append_trace(
                "adjudicator",
                "No consensus order. Graph will reject.",
                tradable=len(tradable),
                risk_passed=state.risk.passed,
            )
            return state

        best = max(tradable, key=lambda s: s.strength)
        if best.side == Side.BUY and state.research.sentiment < -0.4:
            state.proposal = None
            state.route = RouteDecision.REJECT
            state.reject_reason = "research_conflict"
            state.append_trace("adjudicator", "Research sentiment conflicts with buy.")
            return state

        citations = list(state.research.citations)
        rationale = (
            f"{best.reason} Research (grounded={state.research.grounded}): "
            f"{state.research.summary[:280]}"
        )
        ok, flags = screen_text(rationale)
        excerpts = [c.excerpt for c in citations]
        grounded = require_grounding(rationale, excerpts or [state.research.summary], min_overlap=2)
        if not ok or (citations and not grounded):
            state.proposal = None
            state.route = RouteDecision.REJECT
            state.reject_reason = "ungrounded_or_unsafe_rationale"
            state.append_trace("adjudicator", "Rationale failed guardrails.", flags=flags)
            return state

        qty = max(int(best.quantity), 1)
        proposal = OrderProposal(
            symbol=state.symbol,
            instrument_token=state.instrument_token,
            exchange=state.exchange,
            side=best.side,
            quantity=qty,
            limit_price=state.last_price,
            stop_loss_pct=1.0,
            confidence=min(0.99, 0.4 + best.strength * 0.5 + max(state.research.sentiment, 0) * 0.1),
            rationale=rationale,
            citations=citations,
            notional=qty * state.last_price,
        )
        state.proposal = proposal
        state.append_trace(
            "adjudicator",
            "Synthesized a single order proposal.",
            side=proposal.side.value,
            quantity=proposal.quantity,
            confidence=proposal.confidence,
        )
        return state
