import numpy as np

from agents.state import AgentState, RiskAssessment
from governance.policy_engine import _close_returns, _historical_var, _max_drawdown
from observability.tracing import TRACER


def risk_node(state: AgentState) -> AgentState:
    with TRACER.start_as_current_span("risk_node", symbol=state.symbol):
        returns = _close_returns(state.bars)
        vol = float(np.std(returns)) if len(returns) else 0.0
        hist_var = abs(_historical_var(returns, 0.05))
        drawdown = _max_drawdown(state.bars)
        last_ret = float(returns[-1]) if len(returns) else 0.0
        circuit = last_ret <= -0.05
        position = state.positions.get(state.symbol)
        current_value = position.market_value if position else 0.0
        concentration = (current_value / state.nav) if state.nav else 0.0
        flags = []
        if concentration > state.max_name_weight:
            flags.append("already_concentrated")
        if circuit:
            flags.append("circuit_breaker")
        if drawdown > 0.10:
            flags.append("drawdown")
        if not state.research.grounded:
            flags.append("ungrounded_research")
        if state.research.injection_flagged:
            flags.append("injection")
        risk_score = min(
            1.0,
            0.25 * vol * 10
            + 0.25 * drawdown / 0.1
            + 0.25 * (1 if circuit else 0)
            + 0.25 * (0 if state.research.grounded else 1),
        )
        passed = "injection" not in flags and "circuit_breaker" not in flags
        state.risk = RiskAssessment(
            concentration_pct=concentration * 100,
            historical_var_95=hist_var,
            volatility=vol,
            drawdown_pct=drawdown * 100,
            circuit_breaker=circuit,
            risk_score=risk_score,
            flags=flags,
            passed=passed,
        )
        state.append_trace(
            "risk",
            "Scored concentration, VaR, drawdown, and research grounding.",
            risk_score=risk_score,
            flags=flags,
            passed=passed,
        )
        return state
