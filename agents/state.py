from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


PROMPT_VERSION = "v2-agentic-2026-10-01"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradingMode(str, Enum):
    VIRTUAL = "virtual"
    REAL = "real"


class RouteDecision(str, Enum):
    APPROVE = "approve_trade"
    REJECT = "reject_trade"
    ESCALATE = "should_escalate_to_human"


class Citation(BaseModel):
    chunk_id: str
    source_uri: str
    form_type: str = "10-K"
    excerpt: str = ""
    score: float = 0.0


class CandidateSignal(BaseModel):
    source: str
    side: Side
    symbol: str
    strength: float = Field(ge=0.0, le=1.0)
    price: float
    quantity: int = Field(ge=1)
    reason: str
    indicators: Dict[str, float] = Field(default_factory=dict)


class ResearchBrief(BaseModel):
    summary: str = ""
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    citations: List[Citation] = Field(default_factory=list)
    grounded: bool = False
    injection_flagged: bool = False


class RiskAssessment(BaseModel):
    concentration_pct: float = 0.0
    historical_var_95: float = 0.0
    volatility: float = 0.0
    drawdown_pct: float = 0.0
    circuit_breaker: bool = False
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    flags: List[str] = Field(default_factory=list)
    passed: bool = True


class OrderProposal(BaseModel):
    symbol: str
    instrument_token: str = ""
    exchange: str = "NSE_EQ"
    side: Side
    quantity: int = Field(ge=1)
    limit_price: float
    stop_loss_pct: float = Field(default=1.0, ge=0.0, le=5.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    citations: List[Citation] = Field(default_factory=list)
    notional: float = 0.0

    @field_validator("rationale")
    @classmethod
    def rationale_present(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Order rationale is required.")
        return value.strip()


class PolicyDecision(BaseModel):
    allowed: bool
    violations: List[str] = Field(default_factory=list)
    capped_quantity: Optional[int] = None
    requires_human: bool = True
    max_notional: float = 0.0
    var_estimate: float = 0.0


class ReasoningTrace(BaseModel):
    node: str
    timestamp: str = Field(default_factory=utc_now)
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class GraphEvent(BaseModel):
    node: str
    timestamp: str = Field(default_factory=utc_now)
    kind: str = "transition"
    message: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    success: bool = False
    order_id: Optional[int] = None
    broker_order_id: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False


class Position(BaseModel):
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    market_value: float = 0.0


class Bar(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class AgentState(BaseModel):
    run_id: str
    thread_id: str
    symbol: str
    instrument_token: str = ""
    exchange: str = "NSE_EQ"
    trading_mode: TradingMode = TradingMode.VIRTUAL
    portfolio_id: Optional[int] = None
    strategy_id: Optional[int] = None
    operator_id: Optional[str] = None
    cash: float = 100000.0
    nav: float = 100000.0
    max_name_weight: float = 0.05
    auto_approve_confidence: float = 0.92
    positions: Dict[str, Position] = Field(default_factory=dict)
    bars: List[Bar] = Field(default_factory=list)
    last_price: float = 0.0
    research: ResearchBrief = Field(default_factory=ResearchBrief)
    signals: List[CandidateSignal] = Field(default_factory=list)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    proposal: Optional[OrderProposal] = None
    policy: Optional[PolicyDecision] = None
    route: Optional[RouteDecision] = None
    interrupted: bool = False
    human_decision: Optional[str] = None
    traces: List[ReasoningTrace] = Field(default_factory=list)
    events: List[GraphEvent] = Field(default_factory=list)
    execution: Optional[ExecutionResult] = None
    prompt_version: str = PROMPT_VERSION
    current_node: str = "start"
    terminal: bool = False
    reject_reason: Optional[str] = None

    def append_trace(self, node: str, message: str, **data: Any) -> None:
        self.traces.append(ReasoningTrace(node=node, message=message, data=data))
        self.events.append(
            GraphEvent(node=node, kind="thought", message=message, payload=data)
        )
        self.current_node = node

    def to_public_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class AgentStateDict(TypedDict, total=False):
    """TypedDict mirror for LangGraph reducers / external checkpointers."""

    run_id: str
    thread_id: str
    symbol: str
    cash: float
    nav: float
    last_price: float
    trading_mode: str
    current_node: str
    interrupted: bool
    terminal: bool
    prompt_version: str
