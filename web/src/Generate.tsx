import { useEffect, useState } from "react";
import { api, errorText } from "./api";
import type { components } from "./schema";

type Explainer = components["schemas"]["ExplainerOut"];
type Job = components["schemas"]["JobOut"];

// Starts a background generation job and polls it. The free generator can take
// several minutes per call, so this shows elapsed time rather than a spinner.
export default function Generate({ onCreated, onCancel }: { onCreated: (id: number) => void; onCancel: () => void }) {
  const [explainers, setExplainers] = useState<Explainer[]>([]);
  const [explainer, setExplainer] = useState("exp-004");
  const [segment, setSegment] = useState<"practitioner" | "transitioning">("practitioner");
  const [kind, setKind] = useState<"newsletter" | "explainer_summary">("newsletter");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    api.GET("/api/explainers").then(({ data }) => data && setExplainers(data));
  }, []);

  useEffect(() => {
    if (job?.status !== "running") return;
    const timer = setInterval(async () => {
      setNow(Date.now());
      const { data } = await api.GET("/api/jobs/{job_id}", { params: { path: { job_id: job.id } } });
      if (!data) return;
      setJob(data);
      if (data.status === "done" && data.piece_id) onCreated(data.piece_id);
    }, 3000);
    return () => clearInterval(timer);
  }, [job, onCreated]);

  async function start() {
    setError(null);
    const { data, error } = await api.POST("/api/generate", { body: { explainer, segment, kind } });
    if (error) setError(errorText(error));
    else setJob(data);
  }

  const elapsed = job ? Math.round((now - new Date(job.started_at).getTime()) / 1000) : 0;
  return (
    <section className="card">
      <h1>Generate a draft</h1>
      <p className="muted small">
        Grounded in the current corpus only. Every factual sentence must cite a source passage; the draft
        then goes through the same checks and review as any other piece.
      </p>
      <label>
        Explainer
        <select value={explainer} onChange={(e) => setExplainer(e.target.value)} disabled={!!job}>
          {explainers.map((x) => (
            <option key={x.id} value={x.id}>{x.id} · {x.topic} ({x.series})</option>
          ))}
        </select>
      </label>
      <div className="row">
        <label>
          Reader segment
          <select value={segment} onChange={(e) => setSegment(e.target.value as typeof segment)} disabled={!!job}>
            <option value="practitioner">practitioner</option>
            <option value="transitioning">transitioning</option>
          </select>
        </label>
        <label>
          Kind
          <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)} disabled={!!job}>
            <option value="newsletter">newsletter issue</option>
            <option value="explainer_summary">explainer summary</option>
          </select>
        </label>
      </div>
      {error && <p className="error">{error}</p>}
      {job?.status === "running" && <p className="muted">Generating… {elapsed}s (the free model can take a few minutes)</p>}
      {job?.status === "failed" && <p className="error">Generation failed: {job.error}</p>}
      <div className="actions">
        <button className="primary" disabled={job?.status === "running"} onClick={start}>Generate</button>
        <button onClick={onCancel}>Close</button>
      </div>
    </section>
  );
}
