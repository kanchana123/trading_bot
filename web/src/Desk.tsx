import type { FormEvent } from "react";
import HowItWorks from "./HowItWorks";
import { GRAPH_NODES } from "./types";
import type { AuditRow, GraphEdge, Health, PendingRow, RunDump } from "./types";

type Props = {
  health: Health | null;
  apiKey: string;
  operator: string;
  onApiKey: (value: string) => void;
  onOperator: (value: string) => void;
  symbol: string;
  cash: string;
  mode: "virtual" | "real";
  onSymbol: (value: string) => void;
  onCash: (value: string) => void;
  onMode: (value: "virtual" | "real") => void;
  onSubmit: (event: FormEvent) => void;
  busy: boolean;
  notice: string;
  noticeOk: boolean;
  edges: GraphEdge[];
  currentNode: string;
  pending: PendingRow[];
  audit: AuditRow[];
  selected: RunDump | null;
  onSelect: (runId: string) => void;
  onDecide: (runId: string, action: "approve" | "reject") => void;
  view: "desk" | "how-it-works";
  onView: (view: "desk" | "how-it-works") => void;
};

function nodeClass(name: string, current: string, selected: RunDump | null) {
  const order = GRAPH_NODES as readonly string[];
  const currentIdx = order.indexOf(current);
  const idx = order.indexOf(name);
  if (name === current) return "node active";
  if (selected?.terminal && idx !== -1) return "node done";
  if (currentIdx > idx && idx !== -1) return "node done";
  return "node";
}

export default function Desk(props: Props) {
  const proposal = props.selected?.proposal;
  const policy = props.selected?.policy;
  const research = props.selected?.research;

  return (
    <div className="desk">
      <header className="top">
        <div>
          <h1>Investment desk</h1>
          <p className="muted">
            Multi-agent graph with deterministic policy and human approval. Models cannot place orders.
          </p>
          <nav className="menu" aria-label="Desk">
            <button
              type="button"
              className={props.view === "desk" ? "menu-btn active" : "menu-btn"}
              onClick={() => props.onView("desk")}
            >
              Desk
            </button>
            <button
              type="button"
              className={props.view === "how-it-works" ? "menu-btn active" : "menu-btn"}
              onClick={() => props.onView("how-it-works")}
            >
              How it works
            </button>
          </nav>
        </div>
        <div className="health">
          <span className={props.health?.ok ? "pill ok" : "pill warn"}>API {props.health?.ok ? "up" : "down"}</span>
          <span className={props.health?.angel_session ? "pill ok" : "pill warn"}>
            Angel {props.health?.angel_session ? "session" : "offline"}
          </span>
          <span className="pill">LLM execute {props.health?.llm_can_execute ? "on" : "off"}</span>
        </div>
        <label>
          API key
          <input
            type="password"
            value={props.apiKey}
            onChange={(e) => props.onApiKey(e.target.value)}
            placeholder="X-API-Key"
          />
        </label>
        <label>
          Operator
          <input value={props.operator} onChange={(e) => props.onOperator(e.target.value)} />
        </label>
      </header>

      {props.view === "how-it-works" ? (
        <HowItWorks />
      ) : (
        <>
      <form className="run-form" onSubmit={props.onSubmit}>
        <label>
          Symbol
          <input value={props.symbol} onChange={(e) => props.onSymbol(e.target.value)} />
        </label>
        <label>
          Cash
          <input value={props.cash} onChange={(e) => props.onCash(e.target.value)} />
        </label>
        <label>
          Mode
          <select value={props.mode} onChange={(e) => props.onMode(e.target.value as "virtual" | "real")}>
            <option value="virtual">virtual (paper)</option>
            <option value="real">real (always HITL)</option>
          </select>
        </label>
        <button type="submit" disabled={props.busy}>
          {props.busy ? "Running graph…" : "Start run"}
        </button>
        <p className={props.noticeOk ? "notice ok" : "notice warn"}>{props.notice}</p>
      </form>

      <section className="rail">
        <h2>State machine</h2>
        <div className="nodes">
          {GRAPH_NODES.map((name) => (
            <div key={name} className={nodeClass(name, props.currentNode, props.selected)}>
              {name.replace("_", " ")}
            </div>
          ))}
        </div>
        {props.edges.length > 0 ? (
          <p className="muted">
            {props.edges.map((e) => `${e.from}→${e.to}`).join(" · ")}
          </p>
        ) : null}
      </section>

      <div className="grid">
        <section>
          <h2>Approval queue</h2>
          {props.pending.length === 0 ? (
            <p className="muted">No runs waiting on an operator.</p>
          ) : (
            <ul className="queue">
              {props.pending.map((row) => (
                <li key={row.run_id}>
                  <button type="button" className="row-btn" onClick={() => props.onSelect(row.run_id)}>
                    <strong>{row.symbol}</strong>
                    <span>{row.trading_mode}</span>
                    <span className="mono">{row.run_id.slice(0, 8)}</span>
                  </button>
                  <div className="row-actions">
                    <button type="button" onClick={() => props.onDecide(row.run_id, "approve")}>
                      Approve
                    </button>
                    <button type="button" className="danger" onClick={() => props.onDecide(row.run_id, "reject")}>
                      Reject
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="inspector">
          <h2>Inspector</h2>
          {!props.selected ? (
            <p className="muted">Select a pending or audit run.</p>
          ) : (
            <div className="stack">
              <p>
                <span className="pill">{props.selected.status}</span>{" "}
                <span className="mono">{props.selected.run_id}</span>
              </p>
              {proposal ? (
                <div>
                  <h3>
                    {proposal.side.toUpperCase()} {proposal.quantity} {proposal.symbol} @ {proposal.limit_price}
                  </h3>
                  <p>Confidence {(proposal.confidence * 100).toFixed(0)}% · notional {proposal.notional.toFixed(2)}</p>
                  <p>{proposal.rationale}</p>
                </div>
              ) : (
                <p className="muted">No order proposal. {props.selected.reject_reason || ""}</p>
              )}
              {policy && (
                <p>
                  Policy {policy.allowed ? "allowed" : "blocked"}
                  {policy.violations.length ? `: ${policy.violations.join(", ")}` : ""}
                </p>
              )}
              {research && (
                <div>
                  <h3>Research {research.grounded ? "grounded" : "ungrounded"}</h3>
                  <p>{research.summary}</p>
                  <ul>
                    {research.citations.map((c) => (
                      <li key={c.chunk_id}>
                        {c.form_type} {c.chunk_id}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <h3>Trace</h3>
              <ol className="trace">
                {props.selected.events.map((event, i) => (
                  <li key={`${event.timestamp}-${i}`}>
                    <span className="mono">{event.node}</span> {event.message}
                  </li>
                ))}
              </ol>
              {props.selected.execution && (
                <p>
                  Execution {props.selected.execution.success ? "filled" : "failed"}
                  {props.selected.execution.order_id != null
                    ? ` · order ${props.selected.execution.order_id}`
                    : ""}
                  {props.selected.execution.error ? ` · ${props.selected.execution.error}` : ""}
                </p>
              )}
            </div>
          )}
        </section>
      </div>

      <section>
        <h2>Audit ledger</h2>
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Symbol</th>
              <th>Mode</th>
              <th>Status</th>
              <th>Prompt</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            {props.audit.map((row) => (
              <tr key={row.run_id} onClick={() => props.onSelect(row.run_id)}>
                <td className="mono">{row.run_id.slice(0, 8)}</td>
                <td>{row.symbol}</td>
                <td>{row.trading_mode}</td>
                <td>{row.status}</td>
                <td>{row.prompt_version}</td>
                <td>{row.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
        </>
      )}
    </div>
  );
}
