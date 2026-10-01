import { FormEvent, useCallback, useEffect, useState } from "react";
import { api, getApiKey, getOperator, isApiUnavailable, setApiKey, setOperator } from "./api";
import Desk from "./Desk";
import "./styles.css";
import type { AuditRow, GraphEdge, Health, PendingRow, RunDump } from "./types";

export default function App() {
  const [apiKey, setApiKeyState] = useState(getApiKey);
  const [operator, setOperatorState] = useState(getOperator);
  const [health, setHealth] = useState<Health | null>(null);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [pending, setPending] = useState<PendingRow[]>([]);
  const [audit, setAudit] = useState<AuditRow[]>([]);
  const [selected, setSelected] = useState<RunDump | null>(null);
  const [symbol, setSymbol] = useState("RELIANCE-EQ");
  const [cash, setCash] = useState("100000");
  const [mode, setMode] = useState<"virtual" | "real">("virtual");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("Ready.");
  const [noticeOk, setNoticeOk] = useState(true);
  const [view, setView] = useState<"desk" | "how-it-works">("desk");

  const show = (msg: string, ok = true) => {
    setNotice(msg);
    setNoticeOk(ok);
  };

  const refresh = useCallback(async () => {
    const [meta, queue, history, healthBody] = await Promise.all([
      api<{ graph_edges: GraphEdge[] }>("/api/v2/meta"),
      api<PendingRow[]>("/api/v2/runs/pending"),
      api<AuditRow[]>("/api/v2/runs"),
      api<Health>("/api/health"),
    ]);
    setEdges(meta.graph_edges || []);
    setPending(queue);
    setAudit(history);
    setHealth(healthBody);
  }, []);

  useEffect(() => {
    let live = true;
    refresh().catch((err: Error) => {
      show(err.message, false);
      if (isApiUnavailable(err)) live = false;
    });
    const timer = window.setInterval(() => {
      if (!live) return;
      refresh().catch((err: Error) => {
        if (isApiUnavailable(err)) live = false;
      });
    }, 8000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const onApiKey = (value: string) => {
    setApiKeyState(value);
    setApiKey(value);
  };
  const onOperator = (value: string) => {
    setOperatorState(value);
    setOperator(value);
  };

  const loadRun = async (runId: string) => {
    const dump = await api<RunDump>(`/api/v2/runs/${runId}`);
    setSelected(dump);
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    try {
      const dump = await api<RunDump>("/api/v2/runs", {
        method: "POST",
        body: JSON.stringify({
          symbol,
          cash: Number(cash),
          trading_mode: mode,
        }),
      });
      setSelected(dump);
      show(
        dump.interrupted
          ? `Paused before execute_order (${dump.status})`
          : `Finished ${dump.status}`
      );
      await refresh();
    } catch (err) {
      show(err instanceof Error ? err.message : String(err), false);
    } finally {
      setBusy(false);
    }
  };

  const onDecide = async (runId: string, action: "approve" | "reject") => {
    setBusy(true);
    try {
      const dump = await api<RunDump>(`/api/v2/runs/${runId}/${action}`, {
        method: "POST",
        body: JSON.stringify({ operator_id: operator || "desk-operator" }),
      });
      setSelected(dump);
      show(`${action} → ${dump.status}`);
      await refresh();
    } catch (err) {
      show(err instanceof Error ? err.message : String(err), false);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Desk
      health={health}
      apiKey={apiKey}
      operator={operator}
      onApiKey={onApiKey}
      onOperator={onOperator}
      symbol={symbol}
      cash={cash}
      mode={mode}
      onSymbol={setSymbol}
      onCash={setCash}
      onMode={setMode}
      onSubmit={onSubmit}
      busy={busy}
      notice={notice}
      noticeOk={noticeOk}
      edges={edges}
      currentNode={selected?.current_node || "research"}
      pending={pending}
      audit={audit}
      selected={selected}
      onSelect={(id) => loadRun(id).catch((err: Error) => show(err.message, false))}
      onDecide={onDecide}
      view={view}
      onView={setView}
    />
  );
}
