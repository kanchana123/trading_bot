import ArchitectureDiagram from "./ArchitectureDiagram";

export default function HowItWorks() {
  return (
    <article className="how">
      <h1>How it works</h1>
      <p className="lede">
        The investment desk runs a governed graph. Models may retrieve text and propose a trade.
        They never send an order. A rule engine and a human sit in front of execution.
      </p>

      <h2>Architecture</h2>
      <p>
        Clients talk only to FastAPI. The v2 graph can retrieve and propose. Classic backtest and
        realtime share the same executor and SQLite store.
      </p>
      <ArchitectureDiagram />
      <h2>What you are looking at</h2>
      <ul>
        <li>
          <strong>Desk</strong> starts a run, shows the node rail, the approval queue, the inspector
          (proposal, research, policy, traces), and the audit ledger.
        </li>
        <li>
          <strong>Start run</strong> kicks research → quant → risk → adjudicator → policy, then
          always pauses before <code>execute_order</code>.
        </li>
        <li>
          <strong>Approve / Reject</strong> is the human checkpoint. Reject ends the run. Approve
          replays policy and then calls TradeExecutor.
        </li>
      </ul>

      <h2>The graph</h2>
      <ol className="steps">
        <li>
          <strong>Research</strong> retrieves fixture 10-K / 10-Q / news chunks for the symbol
          (hybrid BM25 + dense search, then a reranker). The summary must be grounded in those
          excerpts. Prompt-injection style text is screened out.
        </li>
        <li>
          <strong>Quant</strong> runs the existing Kernel momentum and Bollinger band logic on
          completed bars and emits structured signals, not broker calls.
        </li>
        <li>
          <strong>Risk</strong> scores concentration, historical VaR, drawdown, circuit-breaker
          returns, and whether research was grounded.
        </li>
        <li>
          <strong>Adjudicator</strong> merges those inputs into one order proposal (side, quantity,
          limit, rationale, citations) or rejects with a typed reason.
        </li>
        <li>
          <strong>Policy</strong> is deterministic: max about 5% of NAV in one name, stop-loss
          bound, VaR cap, drawdown cap, cash check. A model cannot override this.
        </li>
        <li>
          <strong>HITL</strong> pauses the graph. Real mode always needs an operator. Paper mode
          still pauses in this build so you can inspect the proposal.
        </li>
        <li>
          <strong>Execute</strong> is the only node allowed to call TradeExecutor, and only after
          policy is replayed.
        </li>
      </ol>

      <h2>What approve does (and does not)</h2>
      <p>
        Approve runs <strong>immediately</strong>. It does not wait for the next trading day, and
        it does not place an AMO / GTD / overnight delivery order.
      </p>
      <ul>
        <li>
          <strong>Virtual (paper)</strong> writes a row to SQLite with <code>order_type=virtual</code>.
          Angel is not contacted. “Execution filled · order N” means a paper fill was recorded now.
        </li>
        <li>
          <strong>Real</strong> calls Angel <code>placeOrder</code> now as LIMIT / INTRADAY, if the
          session is live. If the market is closed or the token is wrong, the broker rejects it; the
          app does not queue it for tomorrow’s open.
        </li>
      </ul>

      <h2>How to use the desk</h2>
      <ol className="steps">
        <li>Paste an API key if <code>TRADING_BOT_API_KEY</code> is set. Set your operator id.</li>
        <li>Choose a symbol (fixture research exists for RELIANCE-EQ and ICICIBANK-EQ), cash, and mode.</li>
        <li>Start run. Inspect citations, policy flags, and the proposal in the inspector.</li>
        <li>Approve or reject from the queue. Open any audit row to replay that graph snapshot.</li>
      </ol>
    </article>
  );
}
