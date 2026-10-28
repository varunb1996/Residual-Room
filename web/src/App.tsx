import { useCallback, useEffect, useState } from "react";
import { api, type PieceSummary } from "./api";
import Generate from "./Generate";
import NewPiece from "./NewPiece";
import PieceView from "./PieceView";

export default function App() {
  const [pieces, setPieces] = useState<PieceSummary[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [creating, setCreating] = useState<"paste" | "generate" | null>(null);
  const opened = useCallback((id: number) => {
    setCreating(null);
    setSelected(id);
  }, []);

  const refresh = useCallback(async () => {
    const { data } = await api.GET("/api/pieces");
    if (data) setPieces(data);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">Residual Room</span>
        <span className="muted">Review</span>
      </header>
      <aside className="sidebar">
        <button className="primary wide" onClick={() => { setCreating("generate"); setSelected(null); }}>
          Generate
        </button>
        <button className="wide paste-btn" onClick={() => { setCreating("paste"); setSelected(null); }}>
          Paste a draft
        </button>
        <ul className="piece-list">
          {pieces.map((p) => (
            <li key={p.id}>
              <button
                className={`piece-item ${p.id === selected ? "active" : ""}`}
                onClick={() => { setSelected(p.id); setCreating(null); }}
              >
                <span className="piece-title">{p.title}</span>
                <span className="piece-meta">
                  <StateBadge state={p.state} /> {p.verdict && <VerdictBadge verdict={p.verdict} />}
                </span>
              </button>
            </li>
          ))}
          {pieces.length === 0 && <li className="muted small">No pieces yet.</li>}
        </ul>
      </aside>
      <main className="main">
        {creating === "paste" && (
          <NewPiece onCreated={(id) => { opened(id); refresh(); }} onCancel={() => setCreating(null)} />
        )}
        {creating === "generate" && (
          <Generate onCreated={(id) => { opened(id); refresh(); }} onCancel={() => setCreating(null)} />
        )}
        {!creating && selected !== null && <PieceView key={selected} id={selected} onChanged={refresh} />}
        {!creating && selected === null && (
          <p className="muted empty">Pick a piece on the left, or paste a new one.</p>
        )}
      </main>
    </div>
  );
}

export function StateBadge({ state }: { state: string }) {
  return <span className={`badge state-${state}`}>{state.replace("_", " ")}</span>;
}

export function VerdictBadge({ verdict }: { verdict: string }) {
  return <span className={`badge verdict-${verdict}`}>{verdict}</span>;
}
