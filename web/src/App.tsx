import { useCallback, useEffect, useState } from "react";
import { api, type PieceSummary } from "./api";
import Analytics from "./Analytics";
import Campaigns from "./Campaigns";
import Generate from "./Generate";
import NewPiece from "./NewPiece";
import PieceView from "./PieceView";

type Screen = "review" | "campaigns" | "analytics";
const SCREENS: { id: Screen; label: string }[] = [
  { id: "review", label: "Review" },
  { id: "campaigns", label: "Campaigns" },
  { id: "analytics", label: "Analytics" },
];

// The queue answers "what needs me?" first; states are grouped by what you do next.
const GROUPS: { label: string; states: string[] }[] = [
  { label: "Needs your review", states: ["in_review"] },
  { label: "Being checked", states: ["checks"] },
  { label: "Drafts", states: ["draft", "rejected"] },
  { label: "Ready to send", states: ["approved", "scheduled"] },
  { label: "Sent", states: ["sent"] },
];

export default function App() {
  const [screen, setScreen] = useState<Screen>("review");
  return (
    <div className="shell">
      <header className="topbar">
        <span className="wordmark">Residual Room</span>
        <nav className="tabs" aria-label="Screens">
          {SCREENS.map((s) => (
            <button key={s.id} className="tab" aria-current={screen === s.id ? "page" : undefined}
              onClick={() => setScreen(s.id)}>
              {s.label}
            </button>
          ))}
        </nav>
      </header>
      {screen === "review" && <Review />}
      {screen === "campaigns" && <main className="page"><Campaigns /></main>}
      {screen === "analytics" && <main className="page"><Analytics /></main>}
    </div>
  );
}

function Review() {
  const [pieces, setPieces] = useState<PieceSummary[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [creating, setCreating] = useState<"paste" | "generate" | null>(null);

  const refresh = useCallback(async () => {
    const { data } = await api.GET("/api/pieces");
    if (data) setPieces(data);
  }, []);
  useEffect(() => {
    refresh();
  }, [refresh]);

  const open = useCallback((id: number) => {
    setCreating(null);
    setSelected(id);
    refresh();
  }, [refresh]);

  return (
    <div className="review">
      <aside className="queue" aria-label="Pieces">
        <div className="queue-actions">
          <button className="primary" onClick={() => { setCreating("generate"); setSelected(null); }}>
            Generate a draft
          </button>
          <button onClick={() => { setCreating("paste"); setSelected(null); }}>Paste a draft</button>
        </div>
        {GROUPS.map((g) => {
          const items = pieces.filter((p) => g.states.includes(p.state));
          if (!items.length) return null;
          return (
            <section key={g.label} className="queue-group">
              <h2>{g.label}<span>{items.length}</span></h2>
              <ul>
                {items.map((p) => (
                  <li key={p.id}>
                    <button className="queue-item" aria-current={p.id === selected ? "true" : undefined}
                      onClick={() => open(p.id)}>
                      <span className={`dot ${p.verdict ?? "none"}`} title={p.verdict ?? "not checked"} />
                      <span className="title">{p.title}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
        {pieces.length === 0 && <p className="queue-empty">No pieces yet. Generate one, or paste a draft.</p>}
      </aside>

      <main className="desk">
        {creating === "generate" && <Generate onCreated={open} onCancel={() => setCreating(null)} />}
        {creating === "paste" && <NewPiece onCreated={open} onCancel={() => setCreating(null)} />}
        {!creating && selected !== null && <PieceView key={selected} id={selected} onChanged={refresh} />}
        {!creating && selected === null && (
          <div className="desk-empty">
            <h1>Pick a piece to review</h1>
            <p>Every sentence is checked against its sources before anything can be sent. Start with what needs your review.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export const STATE_LABEL: Record<string, string> = {
  draft: "Draft",
  checks: "Being checked",
  in_review: "Needs your review",
  approved: "Approved",
  scheduled: "Scheduled",
  sent: "Sent",
  rejected: "Rejected",
};

export const VERDICT_LABEL: Record<string, string> = {
  pass: "All checks pass",
  warn: "Passes, with warnings",
  block: "Blocked",
};
