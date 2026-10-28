import { useEffect, useState } from "react";
import { api, errorText, type Chunk, type Hit, type Segment } from "./api";

/** Right-hand pane: what a sentence cites, and the source text behind it.
 *  The parent keys this by sentence, so state starts fresh for each one. */
export default function Evidence({ segment, hits }: { segment: Segment | null; hits: Hit[] }) {
  const chunkIds = segment
    ? [...new Set([segment.support, ...hits.map((h) => h.evidence)].filter(
        (id): id is string => !!id && id.includes("#"),
      ))]
    : [];
  const [open, setOpen] = useState<string | null>(chunkIds[0] ?? null);
  const [chunk, setChunk] = useState<Chunk | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setChunk(null);
    setError(null);
    if (!open) return;
    api.GET("/api/chunks/{chunk_id}", { params: { path: { chunk_id: open } } }).then(({ data, error }) => {
      if (data) setChunk(data);
      else setError(errorText(error));
    });
  }, [open]);

  if (!segment) {
    return <aside className="evidence muted small">Click a sentence to see its sources.</aside>;
  }
  return (
    <aside className="evidence">
      <h2>Sources</h2>
      <p className="small">
        <span className="muted">Cites: </span>
        {segment.citations.length ? segment.citations.map((c) => <code key={c}>{c}</code>) : <em>nothing</em>}
      </p>
      {chunkIds.length > 0 ? (
        <div className="chips">
          {chunkIds.map((id) => (
            <button key={id} className={`chip ${id === open ? "active" : ""}`} onClick={() => setOpen(id)}>
              {id === segment.support ? "source: " : "evidence: "}
              {id}
            </button>
          ))}
        </div>
      ) : (
        <p className="muted small">No source passage linked to this sentence.</p>
      )}
      {error && <p className="error">{error}</p>}
      {chunk && (
        <div className="chunk">
          <div className="chunk-head">
            <span className={`badge ${chunk.status === "superseded" ? "verdict-block" : "verdict-pass"}`}>
              {chunk.status}
            </span>
            <span className="small">{chunk.source_title}</span>
          </div>
          <p className="muted small">{chunk.heading_path}</p>
          <pre className="chunk-text">{chunk.text}</pre>
        </div>
      )}
    </aside>
  );
}
