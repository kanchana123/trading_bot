from typing import Optional
import hmac
import os

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.graph import GRAPH_EDGES, build_langgraph
from agents.service import InvestmentGraphService
from agents.state import AgentState

router = APIRouter(tags=["v2-agentic"])

_service: Optional[InvestmentGraphService] = None


def get_service() -> InvestmentGraphService:
    if _service is None:
        raise HTTPException(status_code=503, detail="v2 graph service is not initialized")
    return _service


def bind_service(service: InvestmentGraphService) -> None:
    global _service
    _service = service


class GraphRunRequest(BaseModel):
    symbol: str = Field(min_length=3)
    cash: float = Field(default=100000.0, gt=0)
    trading_mode: str = "virtual"
    instrument_token: str = ""
    exchange: str = "NSE_EQ"
    portfolio_id: Optional[int] = None
    strategy_id: Optional[int] = None
    max_name_weight: float = Field(default=0.05, gt=0, le=1)
    bars: Optional[list] = None
    last_price: Optional[float] = None


class HumanDecision(BaseModel):
    operator_id: str = Field(min_length=1)


def _dump(state: AgentState) -> dict:
    return {
        "run_id": state.run_id,
        "status": _status(state),
        "current_node": state.current_node,
        "interrupted": state.interrupted,
        "terminal": state.terminal,
        "route": state.route.value if state.route else None,
        "reject_reason": state.reject_reason,
        "proposal": state.proposal.model_dump(mode="json") if state.proposal else None,
        "policy": state.policy.model_dump(mode="json") if state.policy else None,
        "risk": state.risk.model_dump(mode="json"),
        "research": state.research.model_dump(mode="json"),
        "execution": state.execution.model_dump(mode="json") if state.execution else None,
        "events": [e.model_dump(mode="json") for e in state.events],
        "prompt_version": state.prompt_version,
        "graph_edges": [{"from": a, "to": b} for a, b in GRAPH_EDGES],
    }


def _status(state: AgentState) -> str:
    if state.execution and state.execution.success:
        return "executed"
    if state.terminal and state.route and state.route.value == "reject_trade":
        return "rejected"
    if state.interrupted:
        return "awaiting_human"
    if state.terminal:
        return "done"
    return "running"


@router.get("/meta")
async def v2_meta():
    return {
        "graph_edges": [{"from": a, "to": b} for a, b in GRAPH_EDGES],
        "langgraph_available": build_langgraph() is not None,
        "interrupt_before": ["execute_order"],
        "llm_can_execute": False,
    }


def require_v2_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    expected = os.getenv("TRADING_BOT_API_KEY")
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


@router.post("/runs", status_code=201, dependencies=[Depends(require_v2_key)])
async def start_run(body: GraphRunRequest, service: InvestmentGraphService = Depends(get_service)):
    if body.trading_mode not in ("virtual", "real"):
        raise HTTPException(status_code=400, detail="trading_mode must be virtual or real")
    state = service.start_run(
        symbol=body.symbol,
        cash=body.cash,
        trading_mode=body.trading_mode,
        instrument_token=body.instrument_token,
        exchange=body.exchange,
        portfolio_id=body.portfolio_id,
        strategy_id=body.strategy_id,
        max_name_weight=body.max_name_weight,
        bars=body.bars,
        last_price=body.last_price,
    )
    return _dump(state)


@router.get("/runs")
async def list_runs(service: InvestmentGraphService = Depends(get_service)):
    return service.history()


@router.get("/runs/pending")
async def pending_runs(service: InvestmentGraphService = Depends(get_service)):
    return service.pending()


@router.get("/runs/{run_id}")
async def get_run(run_id: str, service: InvestmentGraphService = Depends(get_service)):
    state = service.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _dump(state)


@router.post("/runs/{run_id}/approve", dependencies=[Depends(require_v2_key)])
async def approve_run(
    run_id: str,
    body: HumanDecision,
    service: InvestmentGraphService = Depends(get_service),
):
    try:
        state = service.approve(run_id, body.operator_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="run not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _dump(state)


@router.post("/runs/{run_id}/reject", dependencies=[Depends(require_v2_key)])
async def reject_run(
    run_id: str,
    body: HumanDecision,
    service: InvestmentGraphService = Depends(get_service),
):
    try:
        state = service.reject(run_id, body.operator_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="run not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _dump(state)


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, service: InvestmentGraphService = Depends(get_service)):
    state = service.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")

    def gen():
        for event in state.events:
            yield f"event: {event.kind}\ndata: {event.model_dump_json()}\n\n"
        yield "event: end\ndata: {\"done\": true}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
