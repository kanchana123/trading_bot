from agents.state import AgentState, RouteDecision
from governance.policy_engine import PolicyEngine
from observability.tracing import TRACER

_engine = PolicyEngine()


def policy_node(state: AgentState) -> AgentState:
    with TRACER.start_as_current_span("policy_node", symbol=state.symbol):
        decision = _engine.evaluate(
            nav=state.nav,
            cash=state.cash,
            last_price=state.last_price,
            bars=state.bars,
            positions=state.positions,
            proposal=state.proposal,
            trading_mode=state.trading_mode.value,
            max_name_weight=state.max_name_weight,
        )
        if (
            decision.allowed
            and decision.capped_quantity
            and state.proposal
            and decision.capped_quantity != state.proposal.quantity
        ):
            state.proposal.quantity = decision.capped_quantity
            state.proposal.notional = state.proposal.quantity * state.last_price
        state.policy = decision
        if not decision.allowed or state.proposal is None:
            state.route = RouteDecision.REJECT
            state.reject_reason = ",".join(decision.violations) or "policy_reject"
        elif (
            state.trading_mode.value == "real"
            or decision.requires_human
            or (state.proposal and state.proposal.confidence < state.auto_approve_confidence)
            or state.risk.risk_score >= 0.45
        ):
            state.route = RouteDecision.ESCALATE
        else:
            state.route = RouteDecision.APPROVE
        state.append_trace(
            "policy",
            "Deterministic policy evaluated before any execution node.",
            allowed=decision.allowed,
            violations=decision.violations,
            route=state.route.value if state.route else None,
            var_estimate=decision.var_estimate,
        )
        return state
