import { useCallback, useEffect, useState } from "react";
import { api, errorText } from "./api";
import type { components } from "./schema";

type View = components["schemas"]["AnalyticsOut"];
type Row = components["schemas"]["RowOut"];
type Dataset = "real" | "seeded";

// Real and seeded data are shown in separate panels and never mixed.
export default function Analytics() {
  return (
    <div className="analytics">
      <Panel dataset="real" />
      <Panel dataset="seeded" />
    </div>
  );
}

function Panel({ dataset }: { dataset: Dataset }) {
  const [view, setView] = useState<View | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());

  const load = useCallback(async () => {
    const { data, error } = await api.GET("/api/analytics/{dataset}", { params: { path: { dataset } } });
    if (data) setView(data);
    else setError(errorText(error));
  }, [dataset]);

  useEffect(() => {
    load();
  }, [load]);

  // A narration takes minutes (generator, then the checker): poll while it runs.
  const running = view?.narration?.status === "running";
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => { setNow(Date.now()); load(); }, 4000);
    return () => clearInterval(timer);
  }, [running, load]);

  async function refresh() {
    setError(null);
    const { error } = await api.POST("/api/analytics/{dataset}/refresh", { params: { path: { dataset } } });
    if (error) setError(errorText(error));
    load();
  }

  async function narrate() {
    setError(null);
    const { error } = await api.POST("/api/analytics/{dataset}/narrate", { params: { path: { dataset } } });
    if (error) setError(errorText(error));
    load();
  }

  if (!view) return <section className="card">{error ? <p className="error">{error}</p> : "Loading…"}</section>;
  const rowsById = new Map(view.rows.map((r) => [r.id, r]));
  const totals = view.rows.filter((r) => r.week === null && !r.derived && r.channel !== "all");
  const insights = view.insights.filter((i) => i.status === "insight");
  const thin = view.insights.filter((i) => i.status === "not_enough_data");
  const n = view.narration;

  return (
    <section className={`card panel-${dataset}`}>
      <div className={`dataset-banner ${dataset}`}>{view.label}</div>
      <div className="actions">
        <button onClick={refresh}>{dataset === "real" ? "Refresh real metrics" : "Regenerate seeded data"}</button>
        <span className="muted small">computed {new Date(view.computed_at).toLocaleString()}</span>
      </div>
      {error && <p className="error">{error}</p>}

      <h2>Totals by channel and segment</h2>
      {totals.length === 0 ? <p className="muted small">No data yet.</p> : <TotalsTable rows={totals} />}

      <h2>Insights</h2>
      {insights.length === 0 && <p className="muted small">No insight passes the rules (minimum sample and significance).</p>}
      {insights.map((i, k) => (
        <div key={k} className="insight">
          <span className="badge verdict-pass">{i.rule.replace("_", " ")}</span> {i.statement}
          <div className="small muted">rows: {i.row_ids.map((id) => `m:${id}`).join(", ")}</div>
        </div>
      ))}
      {thin.map((i, k) => (
        <div key={k} className="insight thin">
          <span className="badge">not enough data</span> {i.statement}
        </div>
      ))}

      <h2>Narration</h2>
      <p className="muted small">Written by the generator model from the insights above, then run through the same claim checker as newsletters. Every number must cite a metric row.</p>
      <div className="actions">
        <button className="primary" disabled={running || insights.length === 0} onClick={narrate}>
          {running ? "Narrating…" : "Narrate insights"}
        </button>
        {running && n && <span className="muted small">{Math.round((now - new Date(n.created_at).getTime()) / 1000)}s (a few minutes)</span>}
      </div>
      {n?.status === "failed" && <p className="error">Narration failed: {n.error}</p>}
      {n?.status === "done" && (
        <div className="narration">
          <p className="small">
            Checker verdict: <span className={`badge verdict-${n.verdict}`}>{n.verdict}</span>
            <span className="muted"> · written by {n.model}</span>
          </p>
          {(n.report?.segments as { text: string; verdict: string; citations: string[] }[] | undefined)?.map((s, k) => (
            <div key={k} className={`segment v-${s.verdict}`}>
              <div className="seg-text">{s.text}</div>
              {s.citations.length > 0 && (
                <div className="small muted">
                  {s.citations.map((c) => {
                    const r = rowsById.get(Number(c.replace("m:", "")));
                    return <div key={c}><code>{c}</code> {r ? describe(r) : "(row not found)"}</div>;
                  })}
                </div>
              )}
              {(n.report?.hits as { segment: number | null; rule: string; reason: string }[])
                .filter((h) => h.segment === k)
                .map((h, j) => <div key={j} className="hit"><span className={`badge verdict-block`}>{h.rule}</span> {h.reason}</div>)}
            </div>
          ))}
          {(n.report?.document_hits as { rule: string; reason: string }[] | undefined)?.map((h, j) => (
            <div key={j} className="hit"><span className="badge verdict-block">{h.rule}</span> {h.reason}</div>
          ))}
        </div>
      )}
    </section>
  );
}

function TotalsTable({ rows }: { rows: Row[] }) {
  const keys = [...new Set(rows.map((r) => `${r.channel}|${r.segment}`))].sort();
  const get = (k: string, metric: string) => rows.find((r) => `${r.channel}|${r.segment}` === k && r.metric === metric);
  return (
    <table className="audit">
      <thead><tr><th>Channel</th><th>Segment</th><th>Deliveries</th><th>Clicks</th><th title="Clicks are counted per tracked link, not per reader, so repeat clicks can exceed 100%">Clicks per delivery</th></tr></thead>
      <tbody>
        {keys.map((k) => {
          const [channel, segment] = k.split("|");
          const ctr = get(k, "ctr");
          return (
            <tr key={k}>
              <td>{channel}</td><td>{segment}</td>
              <td>{get(k, "deliveries")?.value ?? 0}</td><td>{get(k, "clicks")?.value ?? 0}</td>
              <td>{ctr ? `${(ctr.value * 100).toFixed(1)}%` : "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function describe(r: Row): string {
  const rate = (r.metric.endsWith("rate") || r.metric.startsWith("ctr")) && !r.metric.includes("ratio");
  const value = rate ? `${(r.value * 100).toFixed(1)}%` : String(r.value);
  return `${r.week ?? "whole window"} · ${r.channel} · ${r.segment} · ${r.metric} = ${value}${r.n != null ? ` (n=${r.n})` : ""}`;
}
