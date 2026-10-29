import { useCallback, useEffect, useState } from "react";
import { api, errorText } from "./api";
import type { components } from "./schema";

type View = components["schemas"]["AnalyticsOut"];
type Row = components["schemas"]["RowOut"];
type Insight = components["schemas"]["InsightOut"];
type Dataset = "real" | "seeded";

const RULE_LABEL: Record<string, string> = {
  channel_gap: "Channels compared",
  trend_drop: "Falling click-through",
  unsubscribe_spike: "Unsubscribe spike",
};
const VERDICT_LABEL: Record<string, string> = {
  pass: "Every number checks out",
  warn: "Checks pass, with warnings",
  block: "Blocked by the checker",
};

// Real and seeded data sit in separate columns and are never mixed.
// The seeded column is hatched so it can't be mistaken for readers' behaviour.
export default function Analytics() {
  return (
    <>
      <header className="page-head">
        <h1>Analytics</h1>
        <p className="muted">
          Only tracked-link clicks and deliveries are real. Insights come from fixed rules with a minimum sample;
          the model only puts them into words, and the checker verifies every number against a metric row.
        </p>
      </header>
      <div className="datasets">
        <Panel dataset="real" />
        <Panel dataset="seeded" />
      </div>
    </>
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

  const title = dataset === "real" ? "Real readers" : "Seeded data";
  if (!view) {
    return (
      <section className={`dataset ${dataset}`}>
        <h2>{title}</h2>
        {error ? <p className="notice error">{error}</p> : <p className="muted">Loading</p>}
      </section>
    );
  }
  const rowsById = new Map(view.rows.map((r) => [r.id, r]));
  const totals = view.rows.filter((r) => r.week === null && !r.derived && r.channel !== "all");
  const insights = view.insights.filter((i) => i.status === "insight");
  const thin = view.insights.filter((i) => i.status === "not_enough_data");
  const n = view.narration;

  return (
    <section className={`dataset ${dataset}`} aria-label={title}>
      <header>
        <div className="row">
          <h2 className="grow">{title}</h2>
          <button onClick={refresh}>{dataset === "real" ? "Recount" : "Regenerate"}</button>
        </div>
        <p>{view.label}. Counted {new Date(view.computed_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}.</p>
      </header>
      {error && <p className="notice error">{error}</p>}

      <div>
        <h3>By channel and readers</h3>
        {totals.length === 0 ? <p className="muted">Nothing delivered yet.</p> : <TotalsTable rows={totals} />}
      </div>

      <div>
        <h3>What the rules found</h3>
        {insights.length === 0 && thin.length === 0 && <p className="muted">No rule has enough data to say anything yet.</p>}
        {insights.map((i, k) => (
          <div key={k} className="insight">
            <span className="kind">{RULE_LABEL[i.rule] ?? i.rule.replace("_", " ")}</span>
            <p>{i.statement}</p>
            <span className="small muted">From rows {i.row_ids.map((id) => `m:${id}`).join(", ")}</span>
          </div>
        ))}
        {thin.map((i, k) => <ThinInsight key={k} insight={i} />)}
      </div>

      <div className="narration">
        <h3>Written summary</h3>
        <p className="small muted">Drafted by {n?.model ?? "the generator"} from the findings above, then checked like a newsletter.</p>
        <div className="row">
          <button className="primary" disabled={running || insights.length === 0} onClick={narrate}>
            {running ? "Writing" : n?.status === "done" ? "Write it again" : "Write a summary"}
          </button>
          {running && n && <span className="small muted">{Math.round((now - new Date(n.created_at).getTime()) / 1000)}s so far. This takes a few minutes.</span>}
          {!running && insights.length === 0 && <span className="small muted">Needs at least one finding.</span>}
        </div>
        {n?.status === "failed" && <p className="notice error">Writing failed: {n.error}</p>}
        {n?.status === "done" && n.verdict && (
          <>
            <p className={`verdict ${n.verdict}`}>{VERDICT_LABEL[n.verdict] ?? n.verdict}</p>
            {(n.report?.segments as { text: string; verdict: string; citations: string[] }[] | undefined)?.map((s, k) => (
              <div key={k} className={`narr-seg ${s.verdict}`}>
                <p>{s.text}</p>
                {s.citations.map((c) => {
                  const r = rowsById.get(Number(c.replace("m:", "")));
                  return <p key={c} className="small muted">{c}: {r ? describe(r) : "row not found"}</p>;
                })}
                {(n.report?.hits as { segment: number | null; rule: string; reason: string }[])
                  .filter((h) => h.segment === k)
                  .map((h, j) => <p key={j} className="note block small"><span className="rule">{h.rule}</span>{h.reason}</p>)}
              </div>
            ))}
            {(n.report?.document_hits as { rule: string; reason: string }[] | undefined)?.map((h, j) => (
              <p key={j} className="note block small"><span className="rule">{h.rule}</span>{h.reason}</p>
            ))}
          </>
        )}
      </div>
    </section>
  );
}

// A finding the rules held back: shows how far the sample is from the minimum.
function ThinInsight({ insight }: { insight: Insight }) {
  const have = insight.n ?? 0;
  return (
    <div className="insight thin">
      <span className="kind">Not enough data yet</span>
      <p className="muted">{insight.statement}</p>
      <div className="sample">
        <div className="bar" role="img" aria-label={`${have} of ${insight.need} needed`}>
          <span style={{ width: `${Math.min(100, (have / insight.need) * 100)}%` }} />
        </div>
        <span className="small muted">{have} of {insight.need}</span>
      </div>
    </div>
  );
}

const CHANNEL: Record<string, string> = { email: "Email", telegram: "Telegram", whatsapp: "WhatsApp" };
const READERS: Record<string, string> = { practitioner: "Practitioners", transitioning: "Moving into AI", all: "All" };

function TotalsTable({ rows }: { rows: Row[] }) {
  const keys = [...new Set(rows.map((r) => `${r.channel}|${r.segment}`))].sort();
  const get = (k: string, metric: string) => rows.find((r) => `${r.channel}|${r.segment}` === k && r.metric === metric);
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th>Channel</th><th>Readers</th><th className="num">Delivered</th><th className="num">Clicks</th>
            <th className="num" title="Clicks are counted per tracked link, not per reader, so repeat clicks can exceed 100%">Clicks per delivery</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => {
            const [channel, segment] = k.split("|");
            const ctr = get(k, "ctr");
            return (
              <tr key={k}>
                <td>{CHANNEL[channel] ?? channel}</td><td>{READERS[segment] ?? segment}</td>
                <td className="num">{get(k, "deliveries")?.value ?? 0}</td>
                <td className="num">{get(k, "clicks")?.value ?? 0}</td>
                <td className="num">{ctr ? `${(ctr.value * 100).toFixed(1)}%` : "none"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function describe(r: Row): string {
  const rate = (r.metric.endsWith("rate") || r.metric.startsWith("ctr")) && !r.metric.includes("ratio");
  const value = rate ? `${(r.value * 100).toFixed(1)}%` : String(r.value);
  const where = [r.week ?? "whole window", CHANNEL[r.channel] ?? r.channel, READERS[r.segment] ?? r.segment].join(", ");
  return `${r.metric} = ${value} (${where}${r.n != null ? `, n = ${r.n}` : ""})`;
}
