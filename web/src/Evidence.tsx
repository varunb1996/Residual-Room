import { useEffect, useState } from "react";
import { api, errorText, type Chunk, type Hit, type Segment } from "./api";

/** In the margin beside the selected sentence: what it cites, and the passage behind it. */
export default function SourcePanel({ segment, hits }: { segment: Segment; hits: Hit[] }) {
  // Corpus chunk ids look like "source#0012"; source-level ids have no single passage to show.
  const chunkIds = [...new Set([segment.support, ...hits.map((h) => h.evidence)].filter(
    (id): id is string => !!id && id.includes("#"),
  ))];
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

  if (segment.kind === "exempt") {
    return <p className="small muted">No citation needed: {segment.exempt_type?.replace("_", " ")}.</p>;
  }
  return (
    <div className="source">
      {segment.citations.length === 0 && <p className="small muted">Cites nothing.</p>}
      {chunkIds.length > 1 && (
        <div className="source-pick" role="group" aria-label="Passages">
          {chunkIds.map((id) => (
            <button key={id} aria-pressed={id === open} onClick={() => setOpen(id)}>
              {id === segment.support ? "Supporting passage" : "Evidence"} {id.split("#")[1]}
            </button>
          ))}
        </div>
      )}
      {chunkIds.length === 0 && segment.citations.length > 0 && (
        <p className="small muted">Cites {segment.citations.join(", ")}; no single passage was matched.</p>
      )}
      {error && <p className="small bad-text">{error}</p>}
      {chunk && (
        <>
          <p>
            <strong>{chunk.source_title}</strong>
            {chunk.status === "superseded" && <span className="status-superseded"> (superseded)</span>}
          </p>
          <p className="where">{chunk.heading_path.split(" › ").slice(1).join(" › ") || chunk.heading_path}</p>
          <blockquote>{chunk.text}</blockquote>
        </>
      )}
    </div>
  );
}
