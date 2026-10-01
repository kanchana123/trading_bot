import { useEffect, useId, useRef, useState } from "react";

const ARCHITECTURE = `
flowchart TB
  subgraph clients [Clients]
    direction LR
    Desk[React desk /ui/]
    Dash[Dashboard /]
  end

  subgraph fastapi [FastAPI]
    direction LR
    Auth[X-API-Key]
    V1[v1 REST]
    V2[v2 REST + SSE]
  end

  subgraph graph [V2 investment graph]
    direction LR
    R[research] --> Q[quant] --> K[risk] --> A[adjudicator] --> P[policy] --> H[HITL]
    H -->|approve| E[execute_order]
    H -->|reject| X[ended]
  end

  subgraph layers [V2 layers]
    direction LR
    RAG[RAG BM25 + dense]
    GOV[PolicyEngine]
    AUD[AuditLedger]
  end

  subgraph classic [Classic trading]
    direction LR
    BT[Backtest]
    RT[RealtimeTrader]
    ST[Kernel / Bollinger]
  end

  subgraph shared [Shared]
    direction LR
    EX[TradeExecutor]
    DB[(SQLite)]
    Angel[Angel One]
  end

  Desk --> V2
  Dash --> V1
  Dash --> V2
  Auth --> V1
  Auth --> V2
  V2 --> R
  R --> RAG
  Q --> ST
  P --> GOV
  E --> EX
  E --> AUD
  V1 --> BT
  V1 --> RT
  BT --> ST
  RT --> ST
  RT --> EX
  EX --> DB
  AUD --> DB
  EX --> Angel
  BT --> Angel
  RT --> Angel
`;

export default function ArchitectureDiagram() {
  const host = useRef<HTMLDivElement>(null);
  const renderId = useId().replace(/:/g, "");
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let cancelled = false;
    const svgId = `architecture-${renderId}-${Math.random().toString(36).slice(2)}`;
    import("mermaid")
      .then(({ default: mermaid }) => {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "dark",
          themeVariables: {
            darkMode: "true",
            background: "#1b1e27",
            primaryColor: "#12141b",
            primaryTextColor: "#eceef3",
            primaryBorderColor: "#2a2e3a",
            lineColor: "#6ea8fe",
            secondaryColor: "#1b1e27",
            tertiaryColor: "#111318",
            clusterBkg: "#12141b",
            clusterBorder: "#2a2e3a",
            titleColor: "#9aa3b5",
            edgeLabelBackground: "#1b1e27",
            fontFamily: "ui-sans-serif, system-ui, sans-serif",
          },
        });
        return mermaid.render(svgId, ARCHITECTURE);
      })
      .then(({ svg }) => {
        if (!cancelled && host.current) {
          host.current.innerHTML = svg;
          setReady(true);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [renderId]);

  if (error) {
    return <p className="muted">Architecture diagram failed to render: {error}</p>;
  }

  return (
    <figure className="arch">
      {!ready && !error ? <p className="muted">Rendering architecture…</p> : null}
      <div className="arch-diagram" ref={host} />
      <figcaption>
        Models retrieve and propose. Only TradeExecutor writes a fill, after policy replay and human
        approval.
      </figcaption>
    </figure>
  );
}
