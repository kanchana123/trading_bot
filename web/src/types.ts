export type GraphEdge = { from: string; to: string };

export type Citation = {
  chunk_id: string;
  source_uri: string;
  form_type: string;
  excerpt: string;
  score: number;
};

export type Proposal = {
  symbol: string;
  side: string;
  quantity: number;
  limit_price: number;
  stop_loss_pct: number;
  confidence: number;
  rationale: string;
  citations: Citation[];
  notional: number;
};

export type Policy = {
  allowed: boolean;
  violations: string[];
  capped_quantity: number | null;
  requires_human: boolean;
  max_notional: number;
  var_estimate: number;
};

export type Risk = {
  concentration_pct: number;
  historical_var_95: number;
  volatility: number;
  drawdown_pct: number;
  circuit_breaker: boolean;
  risk_score: number;
  flags: string[];
  passed: boolean;
};

export type Research = {
  summary: string;
  sentiment: number;
  citations: Citation[];
  grounded: boolean;
  injection_flagged: boolean;
};

export type GraphEvent = {
  node: string;
  timestamp: string;
  kind: string;
  message: string;
  payload: Record<string, unknown>;
};

export type Execution = {
  success: boolean;
  order_id: number | null;
  broker_order_id: string | null;
  error: string | null;
  skipped: boolean;
};

export type RunDump = {
  run_id: string;
  status: string;
  current_node: string;
  interrupted: boolean;
  terminal: boolean;
  route: string | null;
  reject_reason: string | null;
  proposal: Proposal | null;
  policy: Policy | null;
  risk: Risk;
  research: Research;
  execution: Execution | null;
  events: GraphEvent[];
  prompt_version: string;
  graph_edges: GraphEdge[];
};

export type PendingRow = {
  run_id: string;
  thread_id: string;
  symbol: string;
  trading_mode: string;
  status: string;
  prompt_version?: string;
  created_at?: string;
  state?: Partial<RunDump> & Record<string, unknown>;
};

export type AuditRow = {
  run_id: string;
  symbol: string;
  trading_mode: string;
  status: string;
  prompt_version?: string;
  created_at?: string;
  updated_at?: string;
};

export type Health = {
  ok: boolean;
  angel_session: boolean;
  realtime_running: boolean;
  api_key_required: boolean;
  v2?: boolean;
  llm_can_execute?: boolean;
};

export const GRAPH_NODES = [
  "research",
  "quant",
  "risk",
  "adjudicator",
  "policy",
  "execute_order",
] as const;
