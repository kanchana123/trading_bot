"""Interruptible multi-agent graph. LangGraph when installed; local runner always."""

from typing import Callable, Dict, Optional

from agents.nodes.adjudicator import adjudicator_node
from agents.nodes.execution_node import execution_node
from agents.nodes.policy_node import policy_node
from agents.nodes.quant_agent import quant_node
from agents.nodes.research_agent import research_node
from agents.nodes.risk_agent import risk_node
from agents.state import AgentState, RouteDecision
from observability.tracing import TRACER

GRAPH_EDGES = [
    ("start", "research"),
    ("research", "quant"),
    ("quant", "risk"),
    ("risk", "adjudicator"),
    ("adjudicator", "policy"),
    ("policy", "execute_order"),
]


def _persist_events(state: AgentState, audit, seen: int) -> int:
    if audit is None:
        return seen
    for event in state.events[seen:]:
        audit.record_event(state.run_id, event.model_dump())
    return len(state.events)


def run_graph(
    state: AgentState,
    *,
    trade_executor=None,
    audit=None,
    stop_before_execute: bool = True,
) -> AgentState:
    """
    Research → quant → risk → adjudicator → policy → (HITL) → execute.

    interrupt_before=['execute_order'] is the default: escalate and auto-approve
    both pause unless stop_before_execute is False and route is approve_trade.
    """
    with TRACER.start_as_current_span("investment_graph", run_id=state.run_id):
        seen = 0
        if audit:
            audit.record_run(state.to_public_dict(), "running")
        for fn in (research_node, quant_node, risk_node, adjudicator_node, policy_node):
            state = fn(state)
            seen = _persist_events(state, audit, seen)
            if audit:
                audit.record_run(state.to_public_dict(), _status_for(state, mid=True))

        if state.route == RouteDecision.REJECT:
            state.terminal = True
            state.append_trace("end", "Rejected before execution.", reason=state.reject_reason)
            seen = _persist_events(state, audit, seen)
            if audit:
                audit.record_run(state.to_public_dict(), "rejected")
            return state

        if stop_before_execute:
            state.interrupted = True
            state.append_trace(
                "hitl",
                "Paused before execute_order. Operator must confirm.",
                route=state.route.value if state.route else None,
            )
            seen = _persist_events(state, audit, seen)
            if audit:
                audit.record_run(state.to_public_dict(), "awaiting_human")
            return state

        return finish_execution(state, trade_executor=trade_executor, audit=audit)


def finish_execution(state: AgentState, *, trade_executor=None, audit=None) -> AgentState:
    if state.route != RouteDecision.APPROVE:
        state.terminal = True
        state.interrupted = False
        state.append_trace("end", "Human or router did not approve execution.")
        if audit:
            audit.record_run(state.to_public_dict(), "rejected")
        return state
    state.interrupted = False
    state = execution_node(state, trade_executor=trade_executor)
    if audit:
        for event in state.events:
            pass
        latest = state.events[-1].model_dump() if state.events else {}
        if latest:
            audit.record_event(state.run_id, latest)
        audit.record_run(
            state.to_public_dict(),
            "executed" if state.execution and state.execution.success else "execution_failed",
        )
    return state


def resume_graph(
    state: AgentState,
    human_decision: str,
    operator_id: str,
    *,
    trade_executor=None,
    audit=None,
) -> AgentState:
    with TRACER.start_as_current_span(
        "resume_graph",
        run_id=state.run_id,
        decision=human_decision,
        operator_id=operator_id,
    ):
        state.operator_id = operator_id
        state.human_decision = human_decision
        if human_decision != "approve":
            state.route = RouteDecision.REJECT
            state.reject_reason = "human_rejected"
            return finish_execution(state, trade_executor=trade_executor, audit=audit)
        state.route = RouteDecision.APPROVE
        return finish_execution(state, trade_executor=trade_executor, audit=audit)


def build_langgraph(trade_executor=None):
    """Optional LangGraph compile with interrupt_before=['execute_order']."""
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    def wrap(fn: Callable[[AgentState], AgentState]):
        def _inner(payload: Dict) -> Dict:
            current = AgentState.model_validate(payload)
            updated = fn(current)
            return updated.model_dump(mode="json")

        return _inner

    def execute_wrapper(payload: Dict) -> Dict:
        current = AgentState.model_validate(payload)
        updated = execution_node(current, trade_executor=trade_executor)
        return updated.model_dump(mode="json")

    def policy_route(payload: Dict) -> str:
        route = payload.get("route") or RouteDecision.REJECT.value
        if route == RouteDecision.REJECT.value:
            return "reject_trade"
        return "execute_order"

    graph = StateGraph(dict)
    graph.add_node("research", wrap(research_node))
    graph.add_node("quant", wrap(quant_node))
    graph.add_node("risk", wrap(risk_node))
    graph.add_node("adjudicator", wrap(adjudicator_node))
    graph.add_node("policy", wrap(policy_node))
    graph.add_node("execute_order", execute_wrapper)
    graph.add_edge(START, "research")
    graph.add_edge("research", "quant")
    graph.add_edge("quant", "risk")
    graph.add_edge("risk", "adjudicator")
    graph.add_edge("adjudicator", "policy")
    graph.add_conditional_edges(
        "policy",
        policy_route,
        {"reject_trade": END, "execute_order": "execute_order"},
    )
    graph.add_edge("execute_order", END)
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["execute_order"])


def _status_for(state: AgentState, mid: bool = False) -> str:
    if state.terminal:
        return "done"
    if mid:
        return "running"
    return "running"
