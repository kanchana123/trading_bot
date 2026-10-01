"""Guarded broker execution. This module must not import LLM clients."""

from agents.state import AgentState, ExecutionResult, RouteDecision
from governance.policy_engine import PolicyEngine
from observability.tracing import TRACER

_engine = PolicyEngine()


class ExecutionDenied(RuntimeError):
    pass


def execution_node(state: AgentState, trade_executor=None) -> AgentState:
    with TRACER.start_as_current_span(
        "execution_node",
        symbol=state.symbol,
        mode=state.trading_mode.value,
    ):
        if state.route != RouteDecision.APPROVE:
            raise ExecutionDenied("execution_node invoked without approve_trade route")
        if state.human_decision not in (None, "approve"):
            raise ExecutionDenied("human rejected this run")
        if state.trading_mode.value == "real" and state.human_decision != "approve":
            raise ExecutionDenied("real trading requires explicit human approval")

        replay = _engine.evaluate(
            nav=state.nav,
            cash=state.cash,
            last_price=state.last_price,
            bars=state.bars,
            positions=state.positions,
            proposal=state.proposal,
            trading_mode=state.trading_mode.value,
            max_name_weight=state.max_name_weight,
        )
        if not replay.allowed or state.proposal is None:
            state.execution = ExecutionResult(
                success=False, error="policy_replay_failed", skipped=True
            )
            state.terminal = True
            state.append_trace("execute_order", "Policy replay blocked execution.", **replay.model_dump())
            return state

        if trade_executor is None:
            state.execution = ExecutionResult(success=False, error="executor_not_configured")
            state.terminal = True
            return state

        result = trade_executor.execute_order(
            portfolio_id=state.portfolio_id,
            strategy_id=state.strategy_id,
            token_details={
                "symbol": state.proposal.symbol,
                "instrumentToken": state.proposal.instrument_token or "0",
                "exchange": state.proposal.exchange,
            },
            transaction_type=state.proposal.side.value,
            quantity=state.proposal.quantity,
            price=state.proposal.limit_price,
            order_type="LIMIT",
            trading_mode=state.trading_mode.value,
        )
        state.execution = ExecutionResult(
            success=bool(result.get("success")),
            order_id=result.get("order_id"),
            broker_order_id=result.get("broker_order_id"),
            error=result.get("error"),
        )
        state.terminal = True
        state.append_trace(
            "execute_order",
            "TradeExecutor invoked after policy replay.",
            success=state.execution.success,
            order_id=state.execution.order_id,
        )
        return state
